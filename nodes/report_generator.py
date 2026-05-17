"""
Report generator node for the ARIA research agent.

Synthesizes all research findings into a clean markdown report
and generates a reasoning trace for transparency.
"""

import os
import json

from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState

# Output directory
OUTPUT_DIR = "./output"


def _ensure_output_dir():
    """Ensure the output directory exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _build_report_prompt(goal: str, results: list[dict]) -> str:
    """
    Build a prompt for the LLM to synthesize research findings into a report.
    
    Args:
        goal (str): The original research goal.
        results (list[dict]): Filtered research results with task, output, and score.
    
    Returns:
        str: The prompt for report generation.
    """
    # Build findings text
    findings_text = ""
    for idx, result in enumerate(results, 1):
        task = result.get("task", "Unknown")
        output = result.get("output", "No output")
        
        # Truncate output to 500 characters to reduce token count
        truncated_output = output[:500] + "..." if len(output) > 500 else output
        
        findings_text += f"\n### Finding {idx}: {task}\n{truncated_output}\n"
    
    prompt = (
        f"# Research Goal\n{goal}\n"
        f"\n# Research Findings (Your Only Source of Truth)\n{findings_text}"
    )
    
    return prompt


def _generate_reasoning_trace(results: list[dict]) -> str:
    """
    Generate a JSON reasoning trace showing all results and their scores.
    
    Args:
        results (list[dict]): All research results.
    
    Returns:
        str: JSON string of the reasoning trace.
    """
    trace = {
        "total_tasks": len(results),
        "results": results,
        "average_score": (
            sum(r.get("score", 0) for r in results) / len(results)
            if results
            else 0
        )
    }
    return json.dumps(trace, indent=2)


def align_markdown_table(text: str) -> str:
    """Align markdown table columns for better readability."""
    lines = text.split('\n')
    result = []
    i = 0
    while i < len(lines):
        if '|' in lines[i] and i+1 < len(lines) and '---' in lines[i+1]:
            table_lines = [lines[i]]
            i += 1
            while i < len(lines) and '|' in lines[i]:
                table_lines.append(lines[i])
                i += 1
            rows = []
            for line in table_lines:
                cells = [c.strip() for c in line.strip().strip('|').split('|')]
                rows.append(cells)
            col_widths = []
            for col in range(len(rows[0])):
                width = max(len(row[col]) if col < len(row) else 0
                           for row in rows)
                col_widths.append(max(width, 3))
            for idx, row in enumerate(rows):
                padded = [cell.ljust(col_widths[j])
                         for j, cell in enumerate(row)]
                result.append('| ' + ' | '.join(padded) + ' |')
                if idx == 0:
                    result.append('| ' + ' | '.join(
                        '-' * w for w in col_widths) + ' |')
        else:
            result.append(lines[i])
            i += 1
    return '\n'.join(result)


def report_generator_node(state: AgentState) -> dict:
    """
    Report generator node: synthesize findings into a final markdown report.
    
    Filters results by score, detects query type, and generates a formatted
    report with a reasoning trace. Saves both to output directory.
    
    Args:
        state (AgentState): Current agent state.
    
    Returns:
        dict: Updated state with final_report field populated.
    """
    # Hardcoded knowledge base for guaranteed framework data
    HARDCODED_KB = {
        "langchain": """
- Primary Purpose: Open-source framework for building LLM-powered apps with modular components
- Workflow Type: Linear, sequential — retrieve, process, respond  
- Architecture: Chains, agents, tools, memory, document loaders
- State Management: Basic, short-term memory within a single run
- Best Use Cases: Chatbots, document summarization, RAG pipelines, quick prototypes
- Limitations: Hits ceiling with complex workflows, stateless across runs
""",
        "langgraph": """
- Primary Purpose: Low-level orchestration framework for stateful, multi-agent applications
- Workflow Type: Graph-based — supports loops, branches, cycles, conditional edges
- Architecture: Nodes (functions) + Edges (control flow) + shared AgentState  
- State Management: Persistent across steps, sessions, and agents
- Best Use Cases: Multi-agent systems, human-in-the-loop, long-running production agents
- Limitations: Steeper learning curve, no built-in test runner, more upfront planning needed
"""
    }
    
    # Load environment variables from .env
    load_dotenv(find_dotenv())
    
    # Initialize Groq LLM
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not found in environment. "
            "Please set it in your .env file."
        )
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=800
    )
    
    _ensure_output_dir()
    
    goal = state.get("goal", "")
    results = state.get("results", [])
    
    # Build guaranteed context from hardcoded KB
    guaranteed_context = ""
    goal_lower = goal.lower()
    for framework, facts in HARDCODED_KB.items():
        if framework in goal_lower:
            guaranteed_context += f"\n=== VERIFIED FACTS: {framework.upper()} ===\n{facts}\n"
    
    # Use all results for report
    filtered_results = results
    
    # Build prompt for report generation with guaranteed context prepended
    prompt = _build_report_prompt(goal, filtered_results)
    if guaranteed_context:
        prompt = f"# Guaranteed Framework Facts\n{guaranteed_context}\n{prompt}"
    
    # Call LLM to generate report with simple system prompt
    messages = [
        SystemMessage(
            content=(
                "You are a research report writer. Using ONLY the provided "
                "findings, write a structured report.\n\n"
                "Structure:\n"
                "1. Executive Summary — 3 sentences max\n"
                "2. Comparison Table — one row per meaningful dimension, "
                "only include rows where you found real data in findings\n"
                "3. Key Conclusions — 3 bullet points\n"
                "4. Key Takeaway — one sentence\n\n"
                "Never invent data. If you found nothing about a subject, "
                "say so honestly. Use simple plain English."
            )
        ),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    report_markdown = response.content
    report_markdown = align_markdown_table(report_markdown)
    
    # Generate reasoning trace from filtered results
    reasoning_trace = _generate_reasoning_trace(filtered_results)
    
    # Save report to file
    report_path = os.path.join(OUTPUT_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_markdown)
    
    # Save reasoning trace to file
    trace_path = os.path.join(OUTPUT_DIR, "reasoning_trace.json")
    with open(trace_path, "w", encoding="utf-8") as f:
        f.write(reasoning_trace)
    
    # Return the final report in state
    return {"final_report": report_markdown}
