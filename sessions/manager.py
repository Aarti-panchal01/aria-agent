"""
Persistent research sessions for ARIA.

Session metadata lives in a small SQLite database; each session's full state
snapshot is a JSON file. ChromaDB (keyed elsewhere) provides the vector memory;
this module tracks the human-facing session list and lets a run be resumed.

The storage root is ``.aria_sessions/`` by default, overridable with the
``ARIA_SESSIONS_DIR`` environment variable (used by tests for isolation).
"""

import json
import os
import shutil
import sqlite3
import uuid
from datetime import datetime

from config import get_logger

logger = get_logger(__name__)


def _base_dir() -> str:
    return os.getenv("ARIA_SESSIONS_DIR", ".aria_sessions")


def _db_path() -> str:
    return os.path.join(_base_dir(), "sessions.db")


def _session_dir(session_id: str) -> str:
    return os.path.join(_base_dir(), session_id)


def _connect() -> sqlite3.Connection:
    os.makedirs(_base_dir(), exist_ok=True)
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            goal        TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            task_count  INTEGER NOT NULL DEFAULT 0,
            status      TEXT NOT NULL DEFAULT 'new'
        )
        """
    )
    return conn


def create_session(goal: str) -> str:
    """Create a new session for ``goal`` and return its id (uuid4)."""
    session_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, goal, created_at, updated_at, task_count, status) "
            "VALUES (?, ?, ?, ?, 0, 'new')",
            (session_id, goal, now, now),
        )
    os.makedirs(_session_dir(session_id), exist_ok=True)
    logger.info("Created session %s for goal: %s", session_id, goal)
    return session_id


def list_sessions() -> list[dict]:
    """Return all sessions (newest first) as dicts."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, goal, created_at, updated_at, task_count, status "
            "FROM sessions ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def save_session(session_id: str, state: dict) -> None:
    """Persist a state snapshot and update session metadata."""
    os.makedirs(_session_dir(session_id), exist_ok=True)
    with open(os.path.join(_session_dir(session_id), "state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)

    task_count = len(state.get("results", []))
    status = "complete" if state.get("is_done") else "partial"
    now = datetime.now().isoformat()
    with _connect() as conn:
        conn.execute(
            "UPDATE sessions SET updated_at=?, task_count=?, status=? WHERE id=?",
            (now, task_count, status, session_id),
        )
    logger.info("Saved session %s (%d tasks, %s)", session_id, task_count, status)


def load_session(session_id: str) -> dict | None:
    """Return the last saved state snapshot for a session, or None."""
    path = os.path.join(_session_dir(session_id), "state.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def delete_session(session_id: str) -> bool:
    """Delete a session's metadata and snapshot. Returns True if it existed."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        existed = cur.rowcount > 0
    shutil.rmtree(_session_dir(session_id), ignore_errors=True)
    if existed:
        logger.info("Deleted session %s", session_id)
    return existed
