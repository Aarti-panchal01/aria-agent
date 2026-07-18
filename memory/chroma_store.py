"""
ChromaDB-based vector store for persisting and retrieving research findings.

Findings are deduplicated by content hash before embedding, and retrieved
in full (no truncation) so memory is substantive rather than cosmetic.
"""

import hashlib
import os
from datetime import datetime

import chromadb

from config import get_logger

logger = get_logger(__name__)

CHROMA_DB_PATH = "./memory/chroma_db"
COLLECTION_NAME = "aria_findings"

os.makedirs(CHROMA_DB_PATH, exist_ok=True)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)


def _get_or_create_collection():
    """Return the findings collection, creating it if needed."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def _content_hash(output: str) -> str:
    """Return a stable md5 hash of a finding's content (used as dedup id)."""
    return hashlib.md5(output.encode("utf-8")).hexdigest()


def _generate_summary(output: str, max_length: int = 100) -> str:
    """Return a short one-line summary of a finding."""
    summary = output[:max_length].replace("\n", " ").strip()
    if len(output) > max_length:
        summary += "..."
    return summary


def save_finding(task: str, output: str) -> bool:
    """
    Save a finding, skipping it if identical content already exists.

    Args:
        task (str): The subtask description.
        output (str): The finding content to store.

    Returns:
        bool: True if a new finding was stored, False if it was a duplicate
        (or an error occurred).
    """
    try:
        collection = _get_or_create_collection()
        doc_id = _content_hash(output)

        existing = collection.get(ids=[doc_id])
        if existing and existing.get("ids"):
            return False  # identical content already embedded

        collection.add(
            ids=[doc_id],
            documents=[output],
            metadatas=[
                {
                    "task": task,
                    "timestamp": datetime.now().isoformat(),
                    "summary": _generate_summary(output),
                }
            ],
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Error saving finding to ChromaDB: %s", exc)
        return False


def retrieve_relevant(query: str, n: int = 3) -> tuple[str, dict]:
    """
    Retrieve the top ``n`` most relevant past findings for ``query``.

    Args:
        query (str): The search query (usually the research goal).
        n (int): Max findings to retrieve.

    Returns:
        tuple[str, dict]: Formatted findings text and stats
        ``{"retrieved": int, "total": int}``.
    """
    try:
        collection = _get_or_create_collection()
        total = collection.count()
        if total == 0:
            return "No past findings in memory yet.", {"retrieved": 0, "total": 0}

        results = collection.query(query_texts=[query], n_results=min(n, total))
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else []

        if not documents:
            return (
                f"No relevant findings found for query: '{query}'",
                {"retrieved": 0, "total": total},
            )

        lines = [f"Relevant Past Findings for '{query}':\n"]
        for idx, (doc, metadata) in enumerate(zip(documents, metadatas, strict=False), 1):
            relevance = ""
            if distances and idx <= len(distances):
                relevance = f" (relevance: {1 - distances[idx - 1]:.2f})"
            lines.append(
                f"Finding {idx}:{relevance}\n"
                f"  Task: {metadata.get('task', 'Unknown Task')}\n"
                f"  Timestamp: {metadata.get('timestamp', 'Unknown Time')}\n"
                f"  Content: {doc}\n"
            )

        return "\n".join(lines), {"retrieved": len(documents), "total": total}
    except Exception as exc:  # noqa: BLE001
        logger.error("Error retrieving findings from ChromaDB: %s", exc)
        return f"Error retrieving findings: {exc}", {"retrieved": 0, "total": 0}


def retrieve_candidates(
    query: str, n: int = 3, threshold: float = 0.75
) -> tuple[list[dict], int]:
    """
    Return candidate past findings whose cosine similarity meets ``threshold``.

    Unlike ``retrieve_relevant`` (which formats whatever is nearest), this
    applies a hard similarity floor so unrelated topics are never returned. The
    caller (memory_reader) additionally runs an LLM relevance check.

    Args:
        query (str): The research goal.
        n (int): Max candidates to fetch before filtering.
        threshold (float): Minimum cosine similarity (0-1) to keep a candidate.

    Returns:
        tuple[list[dict], int]: Candidates ({task, content, similarity,
        timestamp}) above threshold, and the total number of stored findings.
    """
    try:
        collection = _get_or_create_collection()
        total = collection.count()
        if total == 0:
            return [], 0

        results = collection.query(query_texts=[query], n_results=min(n, total))
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else []

        candidates: list[dict] = []
        for idx, (doc, metadata) in enumerate(zip(documents, metadatas, strict=False)):
            similarity = (1 - distances[idx]) if idx < len(distances) else 0.0
            if similarity >= threshold:
                candidates.append(
                    {
                        "task": metadata.get("task", "Unknown Task"),
                        "content": doc,
                        "similarity": round(similarity, 2),
                        "timestamp": metadata.get("timestamp", ""),
                    }
                )
        return candidates, total
    except Exception as exc:  # noqa: BLE001
        logger.error("Error retrieving candidates from ChromaDB: %s", exc)
        return [], 0
