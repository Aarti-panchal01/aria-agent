"""
Memory writer node for the ARIA research agent.

Persists completed research findings to ChromaDB for future retrieval
and context injection in planning phases.
"""

from state import AgentState
from memory.chroma_store import save_finding


def memory_writer_node(state: AgentState) -> dict:
    """
    Memory writer node: persist the last result to ChromaDB.
    
    Extracts the most recent research finding from state and saves it
    to the persistent vector store for context retrieval in future runs.
    This is a side-effect node that does not modify state.
    
    Args:
        state (AgentState): Current agent state.
    
    Returns:
        dict: Empty dict (state unchanged, side-effect operation).
    """
    results = state.get("results", [])
    
    # Handle edge case: no results to save
    if not results:
        return {}
    
    # Get the last result
    last_result = results[-1]
    task = last_result.get("task", "Unknown task")
    output = last_result.get("output", "No output")
    
    # Save to ChromaDB
    save_finding(task=task, output=output)
    
    # Return empty dict (no state modifications)
    return {}
