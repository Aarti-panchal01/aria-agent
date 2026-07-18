"""Tests for memory relevance filtering."""

from unittest.mock import Mock, patch

from nodes.memory_reader import _NONE_MSG, memory_reader_node


def _state(goal="Cockroach Janta Party"):
    return {"goal": goal}


_CANDIDATE = {"task": "CGP after NEET 2026", "content": "unrelated exam stuff", "similarity": 0.8}


def test_no_candidates_returns_none_message():
    with patch("nodes.memory_reader.retrieve_candidates", return_value=([], 7)):
        out = memory_reader_node(_state())
    assert out["memory_context"] == _NONE_MSG
    assert out["memory_stats"] == {"retrieved": 0, "total": 7}


def test_irrelevant_candidate_filtered_out():
    with patch("nodes.memory_reader.retrieve_candidates", return_value=([_CANDIDATE], 7)), patch(
        "nodes.memory_reader.ChatGroq"
    ) as mock_groq, patch.dict("os.environ", {"GROQ_API_KEY": "k"}):
        mock_groq.return_value.invoke.return_value = Mock(content="No")
        out = memory_reader_node(_state())
    assert out["memory_context"] == _NONE_MSG
    assert out["memory_stats"]["retrieved"] == 0


def test_relevant_candidate_kept():
    good = {"task": "Redis internals", "content": "redis uses hashes", "similarity": 0.82}
    with patch("nodes.memory_reader.retrieve_candidates", return_value=([good], 3)), patch(
        "nodes.memory_reader.ChatGroq"
    ) as mock_groq, patch.dict("os.environ", {"GROQ_API_KEY": "k"}):
        mock_groq.return_value.invoke.return_value = Mock(content="Yes, relevant")
        out = memory_reader_node(_state("Compare Redis vs Memcached"))
    assert "Redis internals" in out["memory_context"]
    assert out["memory_stats"]["retrieved"] == 1
