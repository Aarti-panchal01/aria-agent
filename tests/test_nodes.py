"""
Unit tests for ARIA nodes and helpers (mocked LLM / search / memory).
"""

from unittest.mock import Mock, patch

import pytest

from main import sanitize_goal
from nodes.critic import critic_node
from nodes.planner import _parse_subtasks, planner_node
from nodes.report_generator import align_markdown_table, report_generator_node
from schemas import CriticScore
from state import AgentState, merge_results
from tools.search import web_search

# --- Input sanitization ----------------------------------------------------

def test_input_length_cap():
    with pytest.raises(ValueError):
        sanitize_goal("a" * 600)


def test_input_control_chars():
    cleaned = sanitize_goal("Research LangGraph\x00\x01\x02 workflow\x03\x04 efficiency")
    for ch in ("\x00", "\x01", "\x02", "\x03", "\x04"):
        assert ch not in cleaned
    assert "Research LangGraph" in cleaned and "workflow" in cleaned


# --- merge_results reducer (Bug 1 fix at the unit level) --------------------

def test_merge_results_dedupes_by_id():
    old = [{"id": "a", "score": 0}]
    # Same id written back with a score must REPLACE, not append.
    merged = merge_results(old, [{"id": "a", "score": 8}])
    assert len(merged) == 1
    assert merged[0]["score"] == 8


def test_merge_results_appends_new_ids():
    merged = merge_results([{"id": "a"}], [{"id": "b"}])
    assert [r["id"] for r in merged] == ["a", "b"]


# --- Planner ---------------------------------------------------------------

def test_planner_returns_six_subtasks():
    response = "\n".join(f"{i}. Distinct subtask {i}" for i in range(1, 7))
    state: AgentState = _blank_state("Compare X and Y")

    with patch("nodes.planner.ChatGroq") as mock_groq:
        mock_groq.return_value.invoke.return_value = Mock(content=response)
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            result = planner_node(state)

    assert len(result["subtasks"]) == 6
    assert result["current_task_index"] == 0


def test_planner_fallback_on_llm_failure():
    state = _blank_state("Research something")
    with patch("nodes.planner.ChatGroq") as mock_groq:
        mock_groq.return_value.invoke.side_effect = Exception("LLM down")
        with patch("config.time.sleep"), patch.dict("os.environ", {"GROQ_API_KEY": "k"}):
            result = planner_node(state)
    assert "Research the topic" in result["subtasks"]


def test_parse_subtasks_handles_various_formats():
    subtasks = _parse_subtasks("1. First\n2) Second\n3: Third\n    4. Fourth")
    assert len(subtasks) == 4


# --- Critic (structured, same id, no duplicate) ----------------------------

def test_critic_structured_score_updates_in_place():
    finding = {"id": "task-0", "task_index": 0, "task": "T", "output": "O", "score": 0}
    state = _blank_state("goal")
    state["results"] = [finding]

    score = CriticScore(
        relevance=9, specificity=7, source_quality=8, completeness=8, overall=8,
        reasoning="Solid.", replan_needed=False, replan_instruction="",
    )
    with patch("nodes.critic.ChatGroq") as mock_groq:
        mock_groq.return_value.with_structured_output.return_value.invoke.return_value = score
        with patch.dict("os.environ", {"GROQ_API_KEY": "k"}):
            result = critic_node(state)

    out = result["results"][0]
    assert out["id"] == "task-0"          # same id -> merge updates, no duplicate
    assert out["score"] == 8
    assert out["critic"]["relevance"] == 9
    assert 0 <= out["score"] <= 10


def test_critic_fallback_on_llm_failure():
    finding = {"id": "task-0", "task_index": 0, "task": "T", "output": "O", "score": 0}
    state = _blank_state("goal")
    state["results"] = [finding]
    with patch("nodes.critic.ChatGroq") as mock_groq:
        mock_groq.return_value.with_structured_output.return_value.invoke.side_effect = (
            Exception("down")
        )
        with patch("config.time.sleep"), patch.dict("os.environ", {"GROQ_API_KEY": "k"}):
            result = critic_node(state)
    assert result["results"][0]["score"] == 5


# --- Search (Bug 2 fix: always returns str) --------------------------------

def test_search_returns_str_on_failure():
    with patch("tools.search.TavilySearchResults") as mock_tavily:
        mock_tavily.return_value.invoke.side_effect = Exception("Tavily timeout")
        with patch("tools.search.time.sleep"), patch.dict(
            "os.environ", {"TAVILY_API_KEY": "k"}
        ):
            result = web_search.invoke("query")
    assert isinstance(result, str)
    assert "Search failed" in result


# --- Table renderer (Bug 4 fix) --------------------------------------------

def test_align_markdown_table_no_separator_in_body():
    raw = (
        "| A | B |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n"
        "| 3 | 4 |\n"
    )
    aligned = align_markdown_table(raw)
    body = [ln for ln in aligned.splitlines() if ln.strip()][2:]  # skip header+sep
    assert all("---" not in ln for ln in body), aligned
    assert aligned.count("| ---") <= 1 or aligned.split("\n")[1].count("-") > 0


def test_report_generator_produces_markdown():
    report = "# R\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n"
    state = _blank_state("Research topic")
    state["results"] = [{"id": "task-0", "task_index": 0, "task": "T", "output": "O", "score": 8}]

    with patch("nodes.report_generator.ChatGroq") as mock_groq:
        mock_groq.return_value.invoke.return_value = Mock(content=report)
        with patch.dict("os.environ", {"GROQ_API_KEY": "k"}), patch(
            "builtins.open", create=True
        ), patch("os.makedirs"):
            result = report_generator_node(state)
    assert "final_report" in result
    assert "#" in result["final_report"]


# --- helpers ---------------------------------------------------------------

def _blank_state(goal: str) -> AgentState:
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
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
