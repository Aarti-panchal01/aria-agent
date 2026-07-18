"""
LangGraph StateGraph for the ARIA research agent.

Wires the cognitive nodes into a loop with targeted replanning: when the
critic flags a finding as weak (and replan budget remains), the run rewrites
and re-executes only that subtask; otherwise it advances to the next one.
"""

from langgraph.graph import END, StateGraph

from config import MAX_REPLANS, get_logger
from nodes.critic import critic_node
from nodes.executor import executor_node
from nodes.memory_reader import memory_reader_node
from nodes.memory_writer import memory_writer_node
from nodes.planner import planner_node
from nodes.replanner import replanner_node
from nodes.report_generator import report_generator_node
from nodes.terminator import terminator_node
from state import AgentState

logger = get_logger(__name__)


def _latest_result(state: AgentState) -> dict | None:
    """Return the finding for the most-recently-executed subtask."""
    target_index = state.get("current_task_index", 0) - 1
    for r in state.get("results", []):
        if r.get("task_index") == target_index:
            return r
    return None


def _terminator_route(state: AgentState) -> str:
    """
    Route out of the terminator.

    - Done (all subtasks executed) -> report_generator
    - Last finding weak AND replan budget remains -> replanner (targeted)
    - Otherwise -> executor (advance to the next subtask)
    """
    if state.get("is_done", False):
        return "report_generator"

    last = _latest_result(state)
    needs_replan = bool(last and (last.get("critic") or {}).get("replan_needed"))
    if needs_replan and state.get("replan_count", 0) < MAX_REPLANS:
        return "replanner"
    return "executor"


def build_graph():
    """Build and compile the ARIA research agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("memory_reader", memory_reader_node)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("critic", critic_node)
    graph.add_node("memory_writer", memory_writer_node)
    graph.add_node("terminator", terminator_node)
    graph.add_node("replanner", replanner_node)
    graph.add_node("report_generator", report_generator_node)

    graph.set_entry_point("memory_reader")
    graph.add_edge("memory_reader", "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "critic")
    graph.add_edge("critic", "memory_writer")
    graph.add_edge("memory_writer", "terminator")

    graph.add_conditional_edges(
        "terminator",
        _terminator_route,
        {
            "report_generator": "report_generator",
            "replanner": "replanner",
            "executor": "executor",
        },
    )
    graph.add_edge("replanner", "executor")
    graph.add_edge("report_generator", END)

    return graph.compile()


aria_graph = build_graph()
