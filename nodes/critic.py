"""
Critic node for the ARIA research agent.

Evaluates the quality of each research finding using Groq.
Assigns scores that guide replanning and execution decisions.
"""

import os
import re
import time
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from state import AgentState


def _parse_score(response_text: str) -> int:
    """
    Extract an integer score (0-10) from the LLM response.
    
    Searches for the first integer between 0 and 10 in the response.
    Defaults to 5 if parsing fails.
    
    Args:
        response_text (str): The LLM response text.
    
    Returns:
        int: Parsed score clamped between 0 and 10.
    """
    # Try to find an integer between 0 and 10
    match = re.search(r"\b([0-9]|10)\b", response_text)
    if match:
        score = int(match.group(1))
        return max(0, min(10, score))  # Clamp between 0-10
    return 5  # Default to middle score if parsing fails


def critic_node(state: AgentState) -> dict:
    """
    Critic node: evaluate the quality of the last research result.
    
    Reads the most recent finding from state["results"], sends it to Groq
    for quality evaluation (0-10 scale), and updates the result's score field.
    
    Args:
        state (AgentState): Current agent state.
    
    Returns:
        dict: Updated state with the last result's score field populated.
              Appends the scored result back to results via operator.add.
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
        temperature=0.3  # Lower temperature for consistent scoring
    )
    
    results = state.get("results", [])
    
    # Handle edge case: no results yet
    if not results:
        return {}
    
    # Get the last result
    last_result = results[-1]
    task = last_result.get("task", "Unknown task")
    output = last_result.get("output", "No output")
    
    # Build evaluation prompt
    prompt = (
        f"Rate the quality of this research finding on a scale of 0-10.\n"
        f"Task: {task}\n"
        f"Finding: {output}\n\n"
        f"Consider: Is it specific? Does it answer the task? Is it substantive?\n"
        f"Reply with ONLY a single integer between 0 and 10."
    )
    
    # Call LLM for evaluation with retry logic: up to 3 attempts with 2-second wait between retries
    max_retries = 3
    retry_delay = 2
    response_text = None
    
    for attempt in range(max_retries):
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content
            break  # Success, exit retry loop
        except Exception as e:
            if attempt < max_retries - 1:
                # Not the last attempt, wait and retry
                time.sleep(retry_delay)
            else:
                # All retries exhausted, log error and use fallback
                print(f"LLM call failed after {max_retries} retries in critic: {str(e)}")
                response_text = None
    
    # Parse score from response, or use fallback on error
    if response_text:
        score = _parse_score(response_text)
    else:
        score = 5  # Fallback: neutral score
    
    # Update the last result with the score
    last_result["score"] = score
    
    # Return updated result (appended via operator.add in state)
    # This allows the score to be persisted and accessed by conditional edges
    return {"results": [last_result]}
