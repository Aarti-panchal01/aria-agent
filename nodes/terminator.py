"""
Terminator node for the ARIA research agent.

Decides whether the run is finished. Termination is based purely on plan
progress; the replan budget is enforced by the router (see graph.py).
"""

from config import get_logger
from state import AgentState

logger = get_logger(__name__)


def terminator_node(state: AgentState) -> dict:
    """
    Set ``is_done`` when every subtask has been executed.

    Args:
        state (AgentState): Current agent state.

    Returns:
        dict: ``is_done`` flag.
    """
    subtasks = state.get("subtasks", [])
    current_task_index = state.get("current_task_index", 0)

    is_done = current_task_index >= len(subtasks)
    logger.info(
        "Terminator: task %d/%d, done=%s", current_task_index, len(subtasks), is_done
    )
    return {"is_done": is_done}
