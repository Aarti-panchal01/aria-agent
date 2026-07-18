"""
Integration tests that invoke the FULL compiled LangGraph.

These are the tests that would have caught all four original bugs:
duplicated findings, the search-failure TypeError crash, and the malformed
markdown table. Groq, Tavily, and ChromaDB are all mocked.
"""

from contextlib import ExitStack
from unittest.mock import Mock, patch

import pytest

from graph import aria_graph
from main import initial_state
from schemas import CriticScore


class _FakeChat:
    """A stand-in ChatGroq that answers based on the system prompt text."""

    def __init__(self, *args, **kwargs):
        pass

    def _text(self, messages) -> str:
        blob = " ".join(getattr(m, "content", str(m)) for m in messages)
        if "planning AI" in blob:
            return "1. Subtask one\n2. Subtask two\n3. Subtask three"
        if "search query generator" in blob:
            return "a focused search query"
        if "report writer" in blob:
            return "# Report\n\n| Dim | X | Y |\n| --- | --- | --- |\n| Speed | Fast | Slow |\n"
        if "re-planner" in blob:
            return "An improved, more specific subtask"
        return "generic response"

    def invoke(self, messages):
        return Mock(content=self._text(messages))

    def with_structured_output(self, schema):
        return _FakeStructured(self._score)

    # default critic score: good enough, no replan
    _score = CriticScore(
        relevance=8, specificity=8, source_quality=8, completeness=8, overall=8,
        reasoning="Good.", replan_needed=False, replan_instruction="",
    )


class _FakeStructured:
    def __init__(self, score):
        self._score = score

    def invoke(self, messages):
        return self._score


def _run(goal="Compare Redis vs Memcached", chat_cls=_FakeChat, search_fail=False):
    """Invoke the real graph with all external services mocked."""
    from tools.sources.base import SearchResult

    with ExitStack() as stack:
        for mod in ("planner", "executor", "critic", "report_generator", "replanner"):
            stack.enter_context(patch(f"nodes.{mod}.ChatGroq", chat_cls))
        if search_fail:
            stack.enter_context(
                patch("nodes.executor.aggregate_search", side_effect=Exception("boom"))
            )
        else:
            fake = [
                SearchResult(
                    title="X", content="z", url="http://y",
                    source_type="web", relevance_score=0.9,
                )
            ]
            stack.enter_context(patch("nodes.executor.aggregate_search", return_value=fake))
        stack.enter_context(
            patch(
                "nodes.memory_reader.retrieve_relevant",
                return_value=("No past findings.", {"retrieved": 0, "total": 0}),
            )
        )
        stack.enter_context(patch("nodes.memory_writer.save_finding", return_value=True))
        stack.enter_context(patch("nodes.report_generator.open", create=True))
        stack.enter_context(patch("nodes.report_generator.os.makedirs"))
        stack.enter_context(patch.dict("os.environ", {"GROQ_API_KEY": "k", "TAVILY_API_KEY": "k"}))
        return aria_graph.invoke(initial_state(goal))


def test_full_graph_no_duplicates():
    """Every finding has a unique id — the state-duplication bug is gone."""
    state = _run()
    ids = [r["id"] for r in state["results"]]
    assert len(ids) == len(set(ids)), f"duplicate finding ids: {ids}"


def test_full_graph_correct_task_count():
    """One finding per planned subtask (planner returns 3)."""
    state = _run()
    task_indices = sorted(r["task_index"] for r in state["results"])
    assert task_indices == [0, 1, 2], task_indices
    assert len(state["results"]) == 3


def test_search_failure_does_not_crash():
    """A failed search is handled; the run completes without crashing."""
    state = _run(search_fail=True)
    assert state.get("final_report")
    assert len(state["results"]) == 3


def test_report_has_no_malformed_table():
    """The rendered report never contains a separator row in the table body."""
    state = _run()
    lines = [ln for ln in state["final_report"].splitlines() if ln.strip()]
    table_lines = [ln for ln in lines if ln.strip().startswith("|")]
    # exactly one separator row across the whole table
    separators = [ln for ln in table_lines if set(ln.replace("|", "").strip()) <= set("-: ")]
    assert len(separators) == 1, f"expected 1 separator, found {len(separators)}: {separators}"


def test_targeted_replan_reruns_only_failing_task():
    """When the critic flags a finding, replanning re-runs only that subtask."""

    class _ReplanOnceChat(_FakeChat):
        # first critique of task 0 fails, everything else passes
        _calls = {"n": 0}

        def with_structured_output(self, schema):
            self._calls["n"] += 1
            if self._calls["n"] == 1:
                bad = CriticScore(
                    relevance=2, specificity=2, source_quality=2, completeness=2,
                    overall=3, reasoning="Weak.", replan_needed=True,
                    replan_instruction="Use authoritative benchmark sources.",
                )
                return _FakeStructured(bad)
            return _FakeStructured(self._score)

    state = _run(chat_cls=_ReplanOnceChat)
    # Still exactly 3 unique findings (the replan replaced, not appended).
    assert len(state["results"]) == 3
    assert len({r["id"] for r in state["results"]}) == 3
    assert state["replan_count"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
