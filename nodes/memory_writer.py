"""
Memory writer node for the ARIA research agent.

Persists the most recent finding to ChromaDB (with content-hash
deduplication) for retrieval in future runs.
"""

from config import get_logger
from memory.chroma_store import save_finding
from state import AgentState

logger = get_logger(__name__)


def memory_writer_node(state: AgentState) -> dict:
    """
    Persist the last finding to ChromaDB. Side-effect only.

    Args:
        state (AgentState): Current agent state.

    Returns:
        dict: Empty (state unchanged).
    """
    results = state.get("results", [])
    if not results:
        return {}

    last_result = results[-1]
    saved = save_finding(
        task=last_result.get("task", "Unknown task"),
        output=last_result.get("output", "No output"),
    )
    logger.info("Memory writer: %s", "saved new finding" if saved else "skipped duplicate")
    return {}
