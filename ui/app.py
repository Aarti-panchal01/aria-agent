"""
Streamlit UI for ARIA — real-time streaming cognitive-loop research interface.

Top: a compact, collapsible live feed of the cognitive loop (last 5 steps, with
a grouped "show all" view) and a thick progress bar. Bottom: the full-width,
styled research report with a word count and one-click PDF export.
"""

import os
import sys
from datetime import datetime

import streamlit as st

# Make the project root importable when launched via `streamlit run ui/app.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph import aria_graph  # noqa: E402
from main import initial_state, sanitize_goal  # noqa: E402
from report.pdf_exporter import export_to_pdf  # noqa: E402
from sessions.manager import (  # noqa: E402
    create_session,
    list_sessions,
    load_session,
    save_session,
)
from state import merge_results  # noqa: E402
from tools.sources.aggregator import available_source_names  # noqa: E402


def _bridge_secrets_to_env() -> None:
    """Copy Streamlit Cloud secrets into os.environ (ARIA reads os.environ)."""
    try:
        for key in ("GROQ_API_KEY", "TAVILY_API_KEY", "GITHUB_TOKEN"):
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:
        pass  # no secrets file locally — .env is used instead


_bridge_secrets_to_env()

st.set_page_config(page_title="ARIA — Research Agent", page_icon="🔬", layout="wide")

_CSS = """
<style>
.block-container { padding-top: 2.2rem; }
/* cognitive feed */
.feed { line-height: 1.35; margin-top: 2px; }
.feed .step { padding: 4px 0; border-bottom: 1px solid #26262b; font-size: 0.92rem; }
.feed .ts { color: #8b8b93; font-size: 0.72rem; margin-left: 8px; }
/* report styling (report is the main markdown block on the page) */
[data-testid="stMarkdownContainer"] h2 {
    color: #f59e0b; border-bottom: 2px solid #6366f1; padding-bottom: 4px; margin-top: 1.3rem;
}
[data-testid="stMarkdownContainer"] h3 { color: #6366f1; }
[data-testid="stMarkdownContainer"] table { border-collapse: collapse; width: 100%; margin: 8px 0; }
[data-testid="stMarkdownContainer"] th, [data-testid="stMarkdownContainer"] td {
    border: 1px solid #3f3f46; padding: 6px 10px;
}
[data-testid="stMarkdownContainer"] thead th { background: #6366f1; color: #fff; }
[data-testid="stMarkdownContainer"] tbody tr:nth-child(even) { background: rgba(99,102,241,0.09); }
[data-testid="stMarkdownContainer"] li { margin-bottom: 4px; }
/* sidebar: compact session buttons, prominent primary Run button */
section[data-testid="stSidebar"] .stButton button { font-size: 12px; }
section[data-testid="stSidebar"] .stButton button[kind="primary"] {
    font-size: 15px; font-weight: 700; padding: 0.55rem;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

st.title("🔬 ARIA — Autonomous Research Intelligence Agent")
st.caption("Plans → executes → critiques → replans → reports. Watch the cognitive loop run live.")


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _ti_for(node: str, payload: dict, acc: dict) -> int:
    """Task index a streamed step belongs to (-1 for setup/final steps)."""
    if node in ("memory_reader", "planner", "report_generator"):
        return -1
    if node == "critic":
        r = payload.get("results", [])
        return r[-1].get("task_index", -1) if r else -1
    if node == "replanner":
        return payload.get("current_task_index", -1)
    # executor / memory_writer: the latest accumulated finding
    r = acc.get("results", [])
    return r[-1].get("task_index", -1) if r else -1


def _cards_for(node: str, payload: dict, acc: dict) -> list[str]:
    """Return step-card labels (no timestamp) for one streamed node update."""
    cards: list[str] = []
    if node == "memory_reader":
        stats = payload.get("memory_stats", {})
        cards.append(
            f"🧠 <b>Memory</b> — {stats.get('retrieved', 0)} relevant of "
            f"{stats.get('total', 0)} stored"
        )
    elif node == "planner":
        cards.append(f"🗺️ <b>Planned</b> {len(payload.get('subtasks', []))} subtasks")
    elif node == "executor":
        results = acc.get("results", [])
        idx = payload.get("current_task_index", 0)
        total = len(acc.get("subtasks", [])) or "?"
        last = results[-1] if results else {}
        task = (last.get("task") or "")[:70]
        cards.append(f'⚡ <b>Task {idx}/{total}</b>: "{task}"')
        out = last.get("output", "")
        srcs = ", ".join(last.get("sources") or []) or "no sources"
        if isinstance(out, str) and out.startswith("Search failed"):
            cards.append("🔍 Search failed (handled gracefully)")
        else:
            n = out.count("URL:") if isinstance(out, str) else 0
            cards.append(f"🔍 {n} result(s) from [{srcs}]")
    elif node == "critic":
        results = payload.get("results", [])
        c = (results[-1].get("critic") or {}) if results else {}
        if c:
            cards.append(
                f"🎯 <b>Critic {c.get('overall')}/10</b> "
                f"(rel {c.get('relevance')}, spec {c.get('specificity')}, "
                f"src {c.get('source_quality')}, comp {c.get('completeness')})"
            )
    elif node == "replanner":
        prev = acc.get("results", [])
        last_score = prev[-1].get("score", "?") if prev else "?"
        cards.append(
            f"♻️ <b>Replanning task {payload.get('current_task_index', 0)}</b> "
            f"(was {last_score}/10)"
        )
    elif node == "memory_writer":
        cards.append("💾 Saved finding to memory")
    elif node == "report_generator":
        cards.append("✅ <b>Research complete</b>")
    return cards


def _merge(acc: dict, payload: dict) -> None:
    """Accumulate streamed deltas into a running state snapshot."""
    for k, v in payload.items():
        if k == "results":
            acc["results"] = merge_results(acc.get("results", []), v)
        else:
            acc[k] = v


def _render_feed(slot, steps: list[dict]) -> None:
    """Render the last 5 steps, newest first, with muted timestamps."""
    rows = "".join(
        f"<div class='step'>{s['label']} <span class='ts'>{s['ts']}</span></div>"
        for s in reversed(steps[-5:])
    )
    slot.markdown(f"<div class='feed'>{rows}</div>", unsafe_allow_html=True)


def _render_progress(slot, done: int, total: int) -> None:
    """Thick custom progress bar with percentage text inside; green when done."""
    pct = int(done / total * 100) if total else 0
    complete = total > 0 and done >= total
    color = "#22c55e" if complete else "#6366f1"
    label = f"{done} of {total} tasks complete" if total else "Starting…"
    slot.markdown(
        f"<div style='background:#1f1f23;border-radius:8px;height:20px;width:100%;"
        f"overflow:hidden;margin:4px 0;'>"
        f"<div style='width:{max(pct, 12)}%;background:{color};height:100%;display:flex;"
        f"align-items:center;justify-content:center;color:#fff;font-size:12px;"
        f"font-weight:600;white-space:nowrap;transition:width .3s;'>{label}</div></div>",
        unsafe_allow_html=True,
    )


def _render_show_all(steps: list[dict], results: list[dict]) -> None:
    """Grouped-by-task full step history; one header per task with its score."""
    by_ti: dict[int, list[dict]] = {}
    for s in steps:
        by_ti.setdefault(s["ti"], []).append(s)
    score_by_ti = {r.get("task_index"): (r.get("critic") or {}).get("overall") for r in results}

    with st.expander("Show all steps", expanded=False):
        for ti in sorted(by_ti):
            if ti < 0:
                st.markdown("**· Setup / final**")
            else:
                sc = score_by_ti.get(ti)
                st.markdown(f"**· Task {ti + 1}**" + (f" — critic {sc}/10" if sc is not None else ""))
            for s in by_ti[ti]:
                st.markdown(
                    f"<div class='step' style='margin-left:14px'>{s['label']} "
                    f"<span class='ts'>{s['ts']}</span></div>",
                    unsafe_allow_html=True,
                )


def _pdf_download_button(report: str, session_id: str, goal: str, key: str) -> None:
    """Render a one-click 'Export as PDF' download button."""
    if not report:
        return
    slug = "".join(c if c.isalnum() else "-" for c in goal[:30]).strip("-") or "report"
    fname = f"aria-research-{slug}-{datetime.now().strftime('%Y%m%d')}.pdf"
    try:
        pdf_bytes = export_to_pdf(report, session_id, goal)
        st.download_button(
            "⬇️ Export as PDF", data=pdf_bytes, file_name=fname,
            mime="application/pdf", key=key,
        )
    except Exception as exc:  # noqa: BLE001
        st.caption(f"PDF export unavailable: {exc}")


def _render_report(report: str, session_id: str, goal: str, key: str) -> None:
    """Render the full-width report with word count and PDF export."""
    st.divider()
    st.subheader("📊 Research report")
    if not report:
        st.info("No report was produced.")
        return
    st.markdown(report)
    st.caption(f"Report: ~{len(report.split())} words")
    _pdf_download_button(report, session_id, goal, key=key)


def run(goal: str, session_id: str | None = None) -> None:
    """Stream a research run: compact live feed on top, full report below."""
    try:
        clean_goal = sanitize_goal(goal)
    except ValueError as exc:
        st.error(str(exc))
        return

    sid = session_id or create_session(clean_goal)
    st.session_state["active_session"] = sid
    st.caption(f"Session `{sid[:8]}` · {clean_goal} · started {_now()}")

    st.subheader("🧠 Cognitive loop (live)")
    prog_slot = st.empty()
    feed_slot = st.empty()
    _render_progress(prog_slot, 0, 0)

    steps: list[dict] = []
    acc: dict = {}
    enabled_sources = st.session_state.get("enabled_sources") or None

    try:
        for update in aria_graph.stream(initial_state(clean_goal, sid, enabled_sources)):
            for node, payload in update.items():
                if not isinstance(payload, dict):
                    continue
                _merge(acc, payload)
                ti = _ti_for(node, payload, acc)
                for label in _cards_for(node, payload, acc):
                    steps.append({"ti": ti, "label": label, "ts": _now()})
            _render_feed(feed_slot, steps)
            total = len(acc.get("subtasks", []))
            done = min(acc.get("current_task_index", 0), total) if total else 0
            _render_progress(prog_slot, done, total)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Run failed: {exc}")
        return

    total = len(acc.get("subtasks", []))
    _render_progress(prog_slot, total, total)

    acc.setdefault("goal", clean_goal)
    acc["session_id"] = sid
    save_session(sid, acc)

    results = sorted(acc.get("results", []), key=lambda r: r.get("task_index", 0))
    _render_show_all(steps, results)
    _render_report(acc.get("final_report", ""), sid, clean_goal, key="dl-run")


def show_saved_session(session_id: str) -> None:
    """Render a previously saved session and offer to continue it."""
    saved = load_session(session_id)
    if not saved:
        st.info("This session has no saved state yet. Click **Run ARIA** to start it.")
        return
    st.caption(
        f"Session `{session_id[:8]}` · {saved.get('goal', '')} · "
        f"{len(saved.get('results', []))} findings"
    )
    if st.button("🔄 Continue this session"):
        run(saved.get("goal", ""), session_id)
        return
    _render_report(saved.get("final_report", ""), session_id, saved.get("goal", ""), key="dl-saved")


# --- Sidebar controls ------------------------------------------------------

with st.sidebar:
    st.header("Run")
    if st.button("➕ New Research", use_container_width=True):
        for k in ("active_session", "prefill_goal"):
            st.session_state.pop(k, None)
        st.rerun()

    goal_input = st.text_area(
        "Research goal",
        value=st.session_state.get("prefill_goal", ""),
        placeholder="Compare Redis vs Memcached for caching",
        height=100,
    )
    start = st.button("🚀 Run ARIA", type="primary", use_container_width=True)
    st.caption("Requires GROQ_API_KEY and TAVILY_API_KEY (env, .env, or Streamlit secrets).")

    st.divider()
    st.subheader("🔎 Research Sources")
    _available = set(available_source_names())
    _labels = {"web": "Web (Tavily)", "arxiv": "arXiv", "wikipedia": "Wikipedia", "github": "GitHub"}
    selected_sources: list[str] = []
    for name, label in _labels.items():
        avail = name in _available
        if st.checkbox(
            label if avail else f"{label} — key missing",
            value=avail, disabled=not avail, key=f"src-{name}",
        ) and avail:
            selected_sources.append(name)
    st.session_state["enabled_sources"] = selected_sources

    st.divider()
    st.subheader("📚 Previous Sessions")
    sessions = list_sessions()
    if not sessions:
        st.caption("No saved sessions yet.")
    for s in sessions[:20]:
        icon = "✅" if s["status"] == "complete" else "🟡"
        label = f"{icon} {s['goal'][:34]} · {s['created_at'][:10]}"
        if st.button(label, key=f"sess-{s['id']}", use_container_width=True):
            st.session_state["active_session"] = s["id"]
            st.session_state["prefill_goal"] = s["goal"]
            st.rerun()


if start:
    run(goal_input)
elif st.session_state.get("active_session"):
    show_saved_session(st.session_state["active_session"])
else:
    st.info("Enter a research goal in the sidebar and click **🚀 Run ARIA**.")
