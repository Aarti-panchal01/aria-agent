"""
State definition for ARIA (Autonomous Research Intelligence Agent).

Defines the ``AgentState`` shared across every LangGraph node and the
``merge_results`` reducer that keeps the results list free of duplicates.
"""

from typing import Annotated, TypedDict


def merge_results(old: list[dict], new: list[dict]) -> list[dict]:
    """
    Reducer for the ``results`` channel that merges by stable ``id``.

    Unlike ``operator.add`` (which blindly concatenates and therefore
    duplicated every finding when the critic wrote back its score), this
    reducer treats each finding's ``id`` as a primary key: a write with an
    existing ``id`` *replaces* that finding in place while preserving order.

    This is what makes the critic's score-write and targeted replanning's
    re-execution update a finding instead of appending a copy of it.

    Args:
        old (list[dict]): Findings already in state.
        new (list[dict]): Findings returned by the current node.

    Returns:
        list[dict]: Merged findings, deduplicated by ``id``, order-stable.
    """
    merged: dict[str, dict] = {r["id"]: r for r in old}
    for r in new:
        merged[r["id"]] = r
    return list(merged.values())


class AgentState(TypedDict):
    """
    State schema for the ARIA research agent.

    Attributes:
        goal: The original research objective. Set once, never modified.
        subtasks: The current research plan as a list of subtask strings.
        current_task_index: Index of the subtask being executed next.
        results: Accumulated findings, deduplicated by ``id`` via
            ``merge_results``. Each finding is a dict with keys ``id``,
            ``task_index``, ``task``, ``output``, ``critic`` and ``score``.
        memory_context: Relevant past findings injected before planning.
        memory_stats: Summary of the memory lookup ({"retrieved", "total"}).
        final_report: The final markdown report.
        replan_count: Number of targeted replans performed (capped).
        is_done: Termination flag.
        replan_instruction: Critic's instruction for the next targeted replan.
        session_id: Identifier of the persistent session this run belongs to.
        enabled_sources: Names of research sources to query (empty => all
            available: web, arXiv, Wikipedia, GitHub).
        max_replans: Cap on targeted replans per run; after this many, accept the
            best result for a weak subtask and move on (no infinite loops).
    """

    goal: str
    subtasks: list[str]
    current_task_index: int
    results: Annotated[list[dict], merge_results]
    memory_context: str
    memory_stats: dict
    final_report: str
    replan_count: int
    is_done: bool
    replan_instruction: str
    session_id: str
    enabled_sources: list[str]
    max_replans: int
