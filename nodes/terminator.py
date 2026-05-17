"""
Terminator node for the ARIA research agent.

Determines whether the agent should continue executing or terminate
based on task completion and replan constraints.
"""

from state import AgentState


def terminator_node(state: AgentState) -> dict:
    """
    Terminator node: determine if the agent should stop or continue.
    
    Checks exit conditions:
    - All subtasks completed (current_task_index >= len(subtasks))
    - Too many replans (replan_count >= 3)
    - Prevents task re-execution by incrementing index if result already exists
    
    Args:
        state (AgentState): Current agent state.
    
    Returns:
        dict: Updated state with is_done flag and optionally incremented index.
    """
    subtasks = state.get("subtasks", [])
    current_task_index = state.get("current_task_index", 0)
    replan_count = state.get("replan_count", 0)
    results = state.get("results", [])
    
    # Check if current task already has a result (prevent re-execution)
    # If we have a result for this task index, move to the next one
    if len(results) >= current_task_index + 1:
        # This task already has a result, increment index to move forward
        current_task_index += 1
    
    # Check termination conditions
    all_tasks_done = current_task_index >= len(subtasks)
    too_many_replans = replan_count >= 3
    
    is_done = all_tasks_done or too_many_replans
    
    return {
        "is_done": is_done,
        "current_task_index": current_task_index
    }
