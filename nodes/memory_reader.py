"""
Memory reader node for the ARIA research agent.

Retrieves relevant past findings from ChromaDB and injects them as context
for the planner, along with lookup statistics for the final report.
"""

from config import get_logger
from memory.chroma_store import retrieve_relevant
from state import AgentState

logger = get_logger(__name__)


def memory_reader_node(state: AgentState) -> dict:
    """
    Retrieve relevant past findings for the research goal.

    Args:
        state (AgentState): Current agent state.

    Returns:
        dict: ``memory_context`` (formatted text) and ``memory_stats``
        ({"retrieved", "total"}).
    """
    goal = state.get("goal", "")
    if not goal:
        return {"memory_context": "", "memory_stats": {"retrieved": 0, "total": 0}}

    context, stats = retrieve_relevant(query=goal, n=3)
    logger.info(
        "Memory: retrieved %d of %d stored findings", stats["retrieved"], stats["total"]
    )
    return {"memory_context": context, "memory_stats": stats}
