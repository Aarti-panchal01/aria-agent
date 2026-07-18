"""
Report generator node for the ARIA research agent.

Synthesizes all findings into a markdown report (using ONLY the findings as
source material — no hardcoded facts) and writes a full reasoning trace.
"""

import json
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config import GROQ_MODEL, get_logger, invoke_with_retry, require_groq_key
from state import AgentState

logger = get_logger(__name__)

OUTPUT_DIR = "./output"

# Report-length policy.
MIN_WORDS = 300        # below this, regenerate once with an "expand" instruction
TARGET_WORDS = 600     # asked-for minimum in the prompt

_SYSTEM_PROMPT = (
    "You are a research report writer. Using ONLY the provided findings, write a "
    f"thorough, substantive markdown research report of at least {TARGET_WORDS} words.\n\n"
    "Use exactly these sections, each with a markdown '## ' header:\n"
    "## Executive Summary\n"
    "  2-3 full paragraphs summarizing the topic and the most important insights.\n"
    "## Background\n"
    "  What the topic is, and relevant context or history.\n"
    "## Key Findings\n"
    "  A numbered list. Each item is 2-3 sentences and ENDS with a source "
    "attribution tag taken from the findings, e.g. [Web], [arXiv], [Wikipedia], "
    "[GitHub].\n"
    "## Analysis\n"
    "  What the findings mean, the trade-offs, and the implications.\n"
    "## Comparison Table\n"
    "  A clean GitHub-flavored markdown table (header row, ONE separator row, then "
    "data rows) covering the most meaningful dimensions. Only include rows supported "
    "by the findings.\n"
    "## Conclusion\n"
    "  One paragraph with a clear, decisive takeaway.\n\n"
    "Rules: Never invent data or URLs. If a subject has no data, say so honestly. "
    "Write substantive prose with depth, not filler. Do NOT write a Sources section "
    "— it is appended automatically."
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
        truncated = output[:1200] + "..." if len(output) > 1200 else output
        srcs = ", ".join(result.get("sources", []) or []) or "unknown"
        findings_text += (
            f"\n### Finding {idx}: {result.get('task', 'Unknown')} "
            f"(sources: {srcs})\n{truncated}\n"
        )
    return (
        f"# Research Goal\n{goal}\n"
        f"\n# Research Findings (Your Only Source of Truth)\n{findings_text}"
    )


def _collect_urls(results: list[dict]) -> list[str]:
    """Extract unique source URLs (in order) from the findings' raw output."""
    urls: list[str] = []
    seen: set[str] = set()
    for r in results:
        for match in re.findall(r"URL:\s*(\S+)", r.get("output", "") or ""):
            url = match.strip().rstrip(".,)")
            if url and url not in seen and url.lower() != "no":
                seen.add(url)
                urls.append(url)
    return urls


def _word_count(text: str) -> int:
    return len(text.split())


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
        model=GROQ_MODEL, api_key=require_groq_key(), temperature=0.2, max_tokens=2048
    )
    prompt = _build_report_prompt(goal, results)

    def _generate(expand: bool) -> str | None:
        system = _SYSTEM_PROMPT
        if expand:
            system += (
                "\n\nYour previous draft was too short. Expand every section with "
                "more detail, specific facts from the findings, and deeper analysis. "
                f"The report MUST be at least {TARGET_WORDS} words."
            )
        resp = invoke_with_retry(
            llm,
            [SystemMessage(content=system), HumanMessage(content=prompt)],
            context="report_generator",
        )
        return resp.content if resp is not None else None

    report_markdown = _generate(expand=False)

    # Regenerate once if the report came back too thin.
    if report_markdown and _word_count(report_markdown) < MIN_WORDS:
        logger.info(
            "Report too short (%d words); regenerating with expansion",
            _word_count(report_markdown),
        )
        expanded = _generate(expand=True)
        if expanded and _word_count(expanded) > _word_count(report_markdown):
            report_markdown = expanded

    if not report_markdown:
        report_markdown = (
            f"# Research Report\n\n## Goal\n{goal}\n\n## Findings\n\n"
            f"Encountered an error generating the report. "
            f"Collected {len(results)} findings.\n"
        )

    report_markdown = align_markdown_table(report_markdown)

    # Append an accurate Sources section from the URLs actually retrieved.
    urls = _collect_urls(results)
    if urls:
        report_markdown += "\n\n## Sources\n\n" + "\n".join(f"- {u}" for u in urls) + "\n"

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
