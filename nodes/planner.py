"""
Planner node for the ARIA research agent.

Breaks down the research goal into concrete subtasks using Groq LLM.
Integrates with memory context if available.
"""

import os
import re
import time
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState


def _parse_subtasks(response_text: str) -> list[str]:
    """
    Parse a numbered list response from the LLM into a list of subtasks.
    
    Handles various numbering formats: "1.", "1)", etc.
    
    Args:
        response_text (str): The LLM response text.
    
    Returns:
        list[str]: Parsed subtasks.
    """
    # Split by newlines
    lines = response_text.strip().split("\n")
    subtasks = []
    
    for line in lines:
        # Remove leading/trailing whitespace
        line = line.strip()
        
        if not line:
            continue
        
        # Match patterns like "1.", "1)", "1:", etc.
        match = re.match(r"^\d+[\.\)\:]\s*(.+)$", line)
        if match:
            task = match.group(1).strip()
            if task:
                subtasks.append(task)
    
    return subtasks if subtasks else ["Research the topic"]  # Fallback


def planner_node(state: AgentState) -> dict:
    """
    Plan node: breaks the research goal into subtasks.
    
    Consults memory context if available to avoid redundant work.
    Uses ChatGroq to generate a list of 6-8 concrete research subtasks.
    
    Args:
        state (AgentState): Current agent state.
    
    Returns:
        dict: Updated state with subtasks, current_task_index, and replan_count.
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
        temperature=0.3
    )
    
    goal = state.get("goal", "")
    memory_context = state.get("memory_context", "")
    replan_count = state.get("replan_count", 0)
    
    # System prompt for the planner
    system_prompt = """You are a research planning AI. Break the given research goal into 6 concrete subtasks that are each meaningfully different.
Return ONLY a numbered list, nothing else."""
    
    # Add memory context if available
    if memory_context:
        system_prompt += (
            f"\n\nConsider the following past findings to avoid redundant work:\n"
            f"{memory_context}"
        )
    
    # Build user message for LLM planning
    user_message = f"Research goal: {goal}"
    
    # Call LLM with retry logic: up to 3 attempts with 2-second wait between retries
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    max_retries = 3
    retry_delay = 2
    response_text = None
    
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            response_text = response.content
            break  # Success, exit retry loop
        except Exception as e:
            if attempt < max_retries - 1:
                # Not the last attempt, wait and retry
                time.sleep(retry_delay)
            else:
                # All retries exhausted, log error and use fallback
                print(f"LLM call failed after {max_retries} retries in planner: {str(e)}")
                response_text = None
    
    # Parse subtasks from response, or use fallback on error
    if response_text:
        subtasks = _parse_subtasks(response_text)
    else:
        subtasks = ["Research the topic"]  # Fallback: minimal subtask
    
    # Return updated state
    return {
        "subtasks": subtasks,
        "current_task_index": 0,
        "replan_count": replan_count + 1
    }
