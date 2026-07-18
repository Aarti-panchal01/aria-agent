"""
Report generator node for the ARIA research agent.

Synthesizes all findings into a markdown report (using ONLY the findings as
source material — no hardcoded facts) and writes a full reasoning trace.
"""

import json
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config import GROQ_MODEL, get_logger, invoke_with_retry, require_groq_key
from state import AgentState

logger = get_logger(__name__)

OUTPUT_DIR = "./output"

_SYSTEM_PROMPT = (
    "You are a research report writer. Using ONLY the provided findings, write "
    "a structured report.\n\n"
    "Structure:\n"
    "1. Executive Summary — 3 sentences max\n"
    "2. Comparison Table — one row per meaningful dimension, only rows where "
    "you found real data in the findings. Emit clean GitHub-flavored markdown "
    "(a header row, one separator row, then data rows)\n"
    "3. Key Conclusions — 3 bullet points\n"
    "4. Key Takeaway — one sentence\n\n"
    "Never invent data. If you found nothing about a subject, say so honestly. "
    "Use simple, plain English."
)


def _is_separator_cells(cells: list[str]) -> bool:
    """True if every cell looks like a markdown table separator (e.g. ---, :--:)."""
    if not cells:
        return False
    return all(c and set(c) <= set("-: ") and "-" in c for c in cells)


def _format_row(row: list[str], widths: list[int]) -> str:
    """Render one table row with each cell left-padded to its column width."""
    return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"


def align_markdown_table(text: str) -> str:
    """
    Pretty-print markdown tables with aligned columns.

    Correctly skips the source separator row (and any stray separator rows in
    the table body) instead of re-emitting them as data — the bug that put a
    ``| --- | --- |`` row inside the old reports.

    Args:
        text (str): Markdown text possibly containing tables.

    Returns:
        str: Markdown with aligned, well-formed tables.
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        has_sep_next = (
            "|" in lines[i]
            and i + 1 < len(lines)
            and _is_separator_cells(
                [c.strip() for c in lines[i + 1].strip().strip("|").split("|")]
            )
        )
        if has_sep_next:
            block = [lines[i]]
            j = i + 2
            while j < len(lines) and "|" in lines[j]:
                block.append(lines[j])
                j += 1

            rows: list[list[str]] = []
            for line in block:
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if _is_separator_cells(cells):
                    continue  # drop the original + any stray separator rows
                rows.append(cells)

            if rows:
                ncols = max(len(r) for r in rows)
                rows = [r + [""] * (ncols - len(r)) for r in rows]
                widths = [max(3, max(len(r[c]) for r in rows)) for c in range(ncols)]

                out.append(_format_row(rows[0], widths))
                out.append("| " + " | ".join("-" * w for w in widths) + " |")
                out.extend(_format_row(r, widths) for r in rows[1:])
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def _unique_sorted(results: list[dict]) -> list[dict]:
    """Return findings sorted by task index (results are already deduped by id)."""
    return sorted(results, key=lambda r: r.get("task_index", 0))


def _build_report_prompt(goal: str, results: list[dict]) -> str:
    """Build the report-generation prompt from goal and findings."""
    findings_text = ""
    for idx, result in enumerate(results, 1):
        output = result.get("output", "No output")
        truncated = output[:500] + "..." if len(output) > 500 else output
        findings_text += f"\n### Finding {idx}: {result.get('task', 'Unknown')}\n{truncated}\n"
    return (
        f"# Research Goal\n{goal}\n"
        f"\n# Research Findings (Your Only Source of Truth)\n{findings_text}"
    )


def _reasoning_trace(goal: str, results: list[dict], memory_stats: dict) -> str:
    """Serialize a full reasoning trace (including per-dimension critic scores)."""
    scores = [r.get("score", 0) for r in results]
    trace = {
        "goal": goal,
        "total_tasks": len(results),
        "average_score": round(sum(scores) / len(scores), 2) if scores else 0,
        "memory_stats": memory_stats,
        "results": results,
    }
    return json.dumps(trace, indent=2)


def report_generator_node(state: AgentState) -> dict:
    """
    Synthesize findings into the final markdown report and reasoning trace.

    Args:
        state (AgentState): Current agent state.

    Returns:
        dict: ``final_report`` (markdown).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    goal = state.get("goal", "")
    results = _unique_sorted(state.get("results", []))
    memory_stats = state.get("memory_stats", {"retrieved": 0, "total": 0})

    llm = ChatGroq(
        model=GROQ_MODEL, api_key=require_groq_key(), temperature=0.2, max_tokens=800
    )
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_build_report_prompt(goal, results)),
    ]

    response = invoke_with_retry(llm, messages, context="report_generator")
    if response is not None:
        report_markdown = response.content
    else:
        report_markdown = (
            f"# Research Report\n\n## Goal\n{goal}\n\n## Findings\n\n"
            f"Encountered an error generating the report. "
            f"Collected {len(results)} findings.\n"
        )

    report_markdown = align_markdown_table(report_markdown)

    # Append an honest memory footer.
    report_markdown += (
        f"\n\n---\n_Retrieved {memory_stats.get('retrieved', 0)} memories from "
        f"{memory_stats.get('total', 0)} stored across previous runs._\n"
    )

    with open(os.path.join(OUTPUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write(report_markdown)
    with open(os.path.join(OUTPUT_DIR, "reasoning_trace.json"), "w", encoding="utf-8") as f:
        f.write(_reasoning_trace(goal, results, memory_stats))

    logger.info("Report generated from %d findings", len(results))
    return {"final_report": report_markdown}
