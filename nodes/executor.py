"""
Executor node for the ARIA research agent.

Executes the current subtask by asking Groq what to search for,
then executes web_search manually. For research tasks, runs dual
searches and combines results.
"""

import os
import re

from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState
from tools.search import web_search


# Hardcoded knowledge base for common frameworks
KNOWLEDGE_BASE = {
    "langchain": {
        "primary purpose": "Open-source framework for building LLM-powered applications using modular components like chains, agents, prompts, and memory",
        "workflow type": "Linear, sequential — retrieve, process, respond",
        "architecture": "Modular components: chains, agents, tools, memory, document loaders",
        "state management": "Basic, short-term memory within a single run",
        "best use cases": "Chatbots, document summarization, RAG pipelines, quick prototypes",
        "limitations": "Hits a ceiling with complex workflows, no native graph-based execution, stateless across runs"
    },
    "langgraph": {
        "primary purpose": "Low-level orchestration framework for building stateful, multi-agent applications as graphs",
        "workflow type": "Graph-based — supports loops, branches, cycles, and conditional edges",
        "architecture": "Nodes (functions) + Edges (control flow) + shared AgentState",
        "state management": "Persistent across steps, sessions, and agents using TypedDict + reducers",
        "best use cases": "Multi-agent systems, human-in-the-loop workflows, long-running agents, production AI",
        "limitations": "Steeper learning curve, no built-in test runner, requires more upfront architecture planning"
    }
}


def _format_knowledge_base_entry(framework: str) -> str:
    """
    Format a knowledge base entry for display.
    
    Args:
        framework (str): The framework name (lowercase).
    
    Returns:
        str: Formatted knowledge base entry.
    """
    if framework not in KNOWLEDGE_BASE:
        return ""
    
    entry = KNOWLEDGE_BASE[framework]
    formatted = f"=== BASELINE KNOWLEDGE: {framework.upper()} ===\n\n"
    
    for key, value in entry.items():
        formatted += f"{key.replace('_', ' ').title()}: {value}\n\n"
    
    return formatted


def _find_framework_in_task(task: str) -> str | None:
    """
    Check if any known framework is mentioned in the task (case insensitive).
    
    Args:
        task (str): The current task description.
    
    Returns:
        str: Framework name (lowercase) if found, None otherwise.
    """
    task_lower = task.lower()
    # Check LangChain BEFORE LangGraph to avoid substring issues
    if "langchain" in task_lower:
        return "langchain"
    if "langgraph" in task_lower:
        return "langgraph"
    return None


def executor_node(state: AgentState) -> dict:
    """
    Executor node: complete the current subtask using web search.
    
    Asks Groq what search query to use for the subtask, then executes
    the web_search directly in Python and returns the results.
    
    Args:
        state (AgentState): Current agent state.
    
    Returns:
        dict: Updated state with new result appended to results and
              current_task_index incremented.
    """
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
        temperature=0.7
    )
    
    # Get current subtask
    subtasks = state.get("subtasks", [])
    current_index = state.get("current_task_index", 0)
    
    if current_index >= len(subtasks):
        # No more subtasks to execute
        return {
            "results": [],
            "current_task_index": current_index
        }
    
    current_subtask = subtasks[current_index]
    
    # Step 1: Ask Groq what search query to use
    system_prompt = (
        "You are a search query generator. Given a research subtask, "
        "output ONE specific search query. Rules:\n"
        "- If the subtask says 'Research LangChain', your query MUST "
        "start with 'LangChain'\n"
        "- If the subtask says 'Research LangGraph', your query MUST "
        "start with 'LangGraph'\n"
        "- Include the specific attribute being researched\n"
        "- Maximum 8 words\n"
        "- Output the query string ONLY, nothing else"
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=current_subtask)
    ]
    
    query_response = llm.invoke(messages)
    search_query = query_response.content.strip()
    
    # Step 2: Bias toward LangChain documentation if researching LangChain
    if "research langchain" in current_subtask.lower():
        search_query = f"LangChain {search_query} site:python.langchain.com OR site:docs.langchain.com"
    
    # Step 3: Check for knowledge base framework mentions
    framework = _find_framework_in_task(current_subtask)
    kb_entry = _format_knowledge_base_entry(framework) if framework else ""
    
    # Step 4: Execute single web search
    result_output = ""
    
    try:
        search_results = web_search.invoke(search_query)
    except Exception as e:
        search_results = f"Search execution failed: {str(e)}"
    
    # Prepend knowledge base entry if framework found, then add search results
    result_output = kb_entry if kb_entry else ""
    result_output += search_results
    
    # Append result to state
    new_result = {
        "task": current_subtask,
        "output": result_output,
        "score": 0  # Critic will fill this in
    }
    
    return {
        "results": [new_result],  # Will be added via operator.add
        "current_task_index": current_index + 1
    }
