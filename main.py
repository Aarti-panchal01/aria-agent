"""
Command-line entry point for ARIA (Autonomous Research Intelligence Agent).

Prompts for a research goal, runs the LangGraph agent, and prints/saves the
report. The ``run_research`` helper is reused by the example harness and UI.
"""

from config import get_logger
from graph import aria_graph
from sessions.manager import create_session, save_session
from state import AgentState

logger = get_logger(__name__)

MAX_GOAL_LENGTH = 500


def sanitize_goal(goal: str) -> str:
    """Trim, length-check, and strip control characters from a research goal."""
    goal = goal.strip()
    if not goal:
        raise ValueError("Research goal cannot be empty.")
    if len(goal) > MAX_GOAL_LENGTH:
        raise ValueError(
            f"Research goal exceeds {MAX_GOAL_LENGTH} characters (got {len(goal)})."
        )
    return "".join(c for c in goal if ord(c) >= 32 or c in "\n\t")


def initial_state(
    goal: str, session_id: str = "", enabled_sources: list[str] | None = None
) -> AgentState:
    """Build the initial agent state for a research goal."""
    return {
        "goal": goal,
        "subtasks": [],
        "current_task_index": 0,
        "results": [],
        "memory_context": "",
        "memory_stats": {"retrieved": 0, "total": 0},
        "final_report": "",
        "replan_count": 0,
        "is_done": False,
        "replan_instruction": "",
        "session_id": session_id,
        "enabled_sources": enabled_sources or [],
        "max_replans": 2,
    }


def run_research(
    goal: str, session_id: str = "", enabled_sources: list[str] | None = None
) -> dict:
    """
    Run the full research workflow for a goal and return the final state.

    Args:
        goal (str): A sanitized research goal.
        session_id (str): Optional persistent-session id to tag the run with.
        enabled_sources (list[str] | None): Restrict research to these source
            names; None/empty uses all available sources.

    Returns:
        dict: The final agent state (includes ``final_report``).
    """
    return aria_graph.invoke(initial_state(goal, session_id, enabled_sources))


def main() -> None:
    """Interactive CLI entry point."""
    print("=" * 60)
    print("ARIA - Autonomous Research Intelligence Agent")
    print("=" * 60)

    try:
        goal = sanitize_goal(input("\nEnter your research goal: "))
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    session_id = create_session(goal)
    print(f"\n🎯 Researching: {goal}")
    print(f"🗂️  Session: {session_id}")
    print("⏳ Executing research workflow...\n")

    try:
        final_state = run_research(goal, session_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Research execution failed")
        print(f"\n❌ Error during research execution: {exc}")
        return

    save_session(session_id, final_state)

    print("\n" + "=" * 60)
    print("📊 RESEARCH REPORT")
    print("=" * 60)
    print(final_state.get("final_report", ""))
    print("=" * 60)
    print("\n✅ Research complete!")
    print("📄 Report saved to: ./output/report.md")
    print("📋 Reasoning trace saved to: ./output/reasoning_trace.json")


if __name__ == "__main__":
    main()
