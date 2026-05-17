"""
Memory reader node for the ARIA research agent.

Retrieves relevant past findings from ChromaDB and injects them
as context for the planner to avoid redundant work.
"""

from state import AgentState
from memory.chroma_store import retrieve_relevant


def memory_reader_node(state: AgentState) -> dict:
    """
    Memory reader node: retrieve relevant past findings for the goal.
    
    Queries ChromaDB for findings semantically similar to the research goal
    and returns them as formatted context for injection into planning phase.
    
    Args:
        state (AgentState): Current agent state.
    
    Returns:
        dict: Updated state with memory_context field populated.
    """
    goal = state.get("goal", "")
    
    # Retrieve relevant past findings
    if goal:
        memory_context = retrieve_relevant(query=goal, n=3)
    else:
        memory_context = "No goal provided."
    
    return {"memory_context": memory_context}
