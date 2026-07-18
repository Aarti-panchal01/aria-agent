"""
PDF export for ARIA research reports.

Renders the markdown report into a clean, professionally themed PDF using
reportlab's Platypus: headings, tables, and bullet lists are parsed from the
markdown, with a title header, a metadata block, and a per-page footer.
"""

import re
from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import GROQ_MODEL

_INDIGO = colors.HexColor("#6366f1")
_DARK = colors.HexColor("#1a1a1a")
_GREY = colors.HexColor("#6b7280")


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("AriaTitle", parent=ss["Title"], textColor=_INDIGO, fontSize=22))
    ss.add(ParagraphStyle("AriaMeta", parent=ss["Normal"], textColor=_GREY, fontSize=9, leading=13))
    ss.add(ParagraphStyle("AriaH2", parent=ss["Heading2"], textColor=_DARK, spaceBefore=12))
    ss.add(ParagraphStyle("AriaH3", parent=ss["Heading3"], textColor=_DARK, spaceBefore=8))
    ss.add(ParagraphStyle("AriaBody", parent=ss["BodyText"], alignment=TA_LEFT, leading=15))
    return ss


def _inline(text: str) -> str:
    """Escape XML and convert basic markdown inline styles for Paragraph markup."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    return text


def _is_table_sep(cells: list[str]) -> bool:
    return bool(cells) and all(c and set(c) <= set("-: ") and "-" in c for c in cells)


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _table_flowable(block: list[str], styles) -> Table:
    rows = [_split_row(ln) for ln in block if not _is_table_sep(_split_row(ln))]
    ncols = max(len(r) for r in rows)
    data = [
        [Paragraph(_inline(c), styles["AriaBody"]) for c in (r + [""] * (ncols - len(r)))]
        for r in rows
    ]
    table = Table(data, hAlign="LEFT", colWidths=[(6.5 * inch) / ncols] * ncols)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _INDIGO),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _markdown_to_flowables(md: str, styles) -> list:
    """Parse a markdown subset (headings, tables, bullets, paragraphs) to flowables."""
    lines = md.split("\n")
    flow: list = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flow.append(Spacer(1, 6))
            i += 1
            continue

        # Table block
        if "|" in line and i + 1 < len(lines) and _is_table_sep(_split_row(lines[i + 1])):
            block = [line]
            j = i + 1
            while j < len(lines) and "|" in lines[j]:
                block.append(lines[j])
                j += 1
            flow.append(_table_flowable(block, styles))
            flow.append(Spacer(1, 8))
            i = j
            continue

        # Horizontal rule
        if set(stripped) <= set("-*_") and len(stripped) >= 3:
            flow.append(HRFlowable(width="100%", color=colors.HexColor("#d1d5db")))
            i += 1
            continue

        # Headings
        if stripped.startswith("### "):
            flow.append(Paragraph(_inline(stripped[4:]), styles["AriaH3"]))
            i += 1
            continue
        if stripped.startswith("## "):
            flow.append(Paragraph(_inline(stripped[3:]), styles["AriaH2"]))
            i += 1
            continue
        if stripped.startswith("# "):
            flow.append(Paragraph(_inline(stripped[2:]), styles["AriaH2"]))
            i += 1
            continue

        # Bullet list (consume consecutive bullet lines)
        if stripped[:2] in ("- ", "* "):
            items = []
            while i < len(lines) and lines[i].strip()[:2] in ("- ", "* "):
                items.append(
                    ListItem(Paragraph(_inline(lines[i].strip()[2:]), styles["AriaBody"]))
                )
                i += 1
            flow.append(ListFlowable(items, bulletType="bullet", leftIndent=14))
            continue

        # Plain paragraph
        flow.append(Paragraph(_inline(stripped), styles["AriaBody"]))
        i += 1

    return flow


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(_GREY)
    canvas.drawString(
        0.75 * inch, 0.5 * inch,
        "Generated by ARIA — Autonomous Research Intelligence Agent",
    )
    canvas.drawRightString(7.75 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()


def export_to_pdf(report_text: str, session_id: str, goal: str) -> bytes:
    """
    Export a research report (markdown) to a formatted PDF.

    Args:
        report_text (str): The markdown report body.
        session_id (str): The session id (shown in metadata).
        goal (str): The research goal (shown in metadata).

    Returns:
        bytes: The PDF file contents (starts with b"%PDF").
    """
    styles = _styles()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        topMargin=0.75 * inch,
        bottomMargin=0.9 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        title="ARIA Research Report",
        author="ARIA",
    )

    story: list = [
        Paragraph("🔬 ARIA Research Report", styles["AriaTitle"]),
        Spacer(1, 6),
        Paragraph(
            f"<b>Goal:</b> {_inline(goal)}<br/>"
            f"<b>Session ID:</b> {session_id or 'n/a'}<br/>"
            f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/>"
            f"<b>Model:</b> {GROQ_MODEL}",
            styles["AriaMeta"],
        ),
        Spacer(1, 8),
        HRFlowable(width="100%", color=_INDIGO, thickness=1.2),
        Spacer(1, 12),
    ]
    story.extend(_markdown_to_flowables(report_text or "_No report content._", styles))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
