"""
Generate the example research outputs and the results table.

Runs ARIA on a set of diverse topics and writes, for each:
  examples/<slug>/report.md
  examples/<slug>/reasoning_trace.json
…plus examples/RESULTS.md, a table of measured metrics across all runs.

This requires real GROQ_API_KEY and TAVILY_API_KEY. Run it once with your keys
to populate the examples/ directory:

    python scripts/run_examples.py
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_logger  # noqa: E402
from main import run_research, sanitize_goal  # noqa: E402

logger = get_logger("aria.examples")

TOPICS = [
    "Compare Redis vs Memcached for caching",
    "What is retrieval augmented generation",
    "How does transformer attention work",
]

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50]


def _metrics(final_state: dict) -> dict:
    results = final_state.get("results", [])
    scores = [r.get("score", 0) for r in results]
    return {
        "subtasks": len(results),
        "avg_overall_score": round(sum(scores) / len(scores), 2) if scores else 0,
        "replan_cycles": final_state.get("replan_count", 0),
        "memories_retrieved": final_state.get("memory_stats", {}).get("retrieved", 0),
    }


def main() -> None:
    if not (os.getenv("GROQ_API_KEY") and os.getenv("TAVILY_API_KEY")):
        raise SystemExit(
            "Set GROQ_API_KEY and TAVILY_API_KEY (see .env.example) before running."
        )

    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    rows = []

    for topic in TOPICS:
        logger.info("Running example: %s", topic)
        final_state = run_research(sanitize_goal(topic))

        out_dir = os.path.join(EXAMPLES_DIR, _slug(topic))
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write(final_state.get("final_report", ""))
        with open(os.path.join(out_dir, "reasoning_trace.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"goal": topic, "results": final_state.get("results", [])}, f, indent=2
            )

        m = _metrics(final_state)
        m["topic"] = topic
        rows.append(m)

    lines = [
        "# ARIA — Example Results",
        "",
        "Measured across diverse topics (no hardcoded knowledge; every finding "
        "is from live search).",
        "",
        "| Topic | Subtasks | Avg score | Replan cycles | Memories used |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['topic']} | {r['subtasks']} | {r['avg_overall_score']} | "
            f"{r['replan_cycles']} | {r['memories_retrieved']} |"
        )
    with open(os.path.join(EXAMPLES_DIR, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Wrote %d examples + RESULTS.md to %s", len(rows), EXAMPLES_DIR)


if __name__ == "__main__":
    main()
