"""Tests for persistent research sessions."""

from sessions.manager import (
    create_session,
    delete_session,
    list_sessions,
    load_session,
    save_session,
)


def test_session_persistence(tmp_path, monkeypatch):
    """Create → save → load round-trips goal and results exactly."""
    monkeypatch.setenv("ARIA_SESSIONS_DIR", str(tmp_path / "sess"))

    goal = "Compare Redis vs Memcached for caching"
    sid = create_session(goal)

    state = {
        "goal": goal,
        "results": [
            {"id": "task-0", "task_index": 0, "task": "T0", "output": "O0", "score": 8},
            {"id": "task-1", "task_index": 1, "task": "T1", "output": "O1", "score": 7},
        ],
        "is_done": True,
    }
    save_session(sid, state)

    loaded = load_session(sid)
    assert loaded is not None
    assert loaded["goal"] == goal
    assert loaded["results"] == state["results"]

    rows = list_sessions()
    row = next(s for s in rows if s["id"] == sid)
    assert row["goal"] == goal
    assert row["task_count"] == 2
    assert row["status"] == "complete"

    assert delete_session(sid) is True
    assert load_session(sid) is None


def test_load_missing_session_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("ARIA_SESSIONS_DIR", str(tmp_path / "sess"))
    assert load_session("does-not-exist") is None
