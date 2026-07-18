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
/* Global — dark "techy" aesthetic (deep navy + indigo/violet accents) */
.stApp {
    background-color: #0b0f19;
    background-image: radial-gradient(circle at 15% -10%, #171e30 0%, rgba(11,15,25,0) 45%),
                      radial-gradient(circle at 100% 0%, #1a1533 0%, rgba(11,15,25,0) 40%);
    color: #e5e7eb;
}
.main .block-container { max-width: 900px; padding: 2rem 2rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0e1320;
    border-right: 1px solid #1f2637;
}
section[data-testid="stSidebar"] .stMarkdown { font-size: 13px; color: #cbd5e1; }

/* Sidebar footer — flows right after content (no forced gap), still at bottom */
.sidebar-footer {
    margin-top: 1.25rem;
    padding: 0.75rem 0 0.5rem 0;
    border-top: 1px solid #1f2637;
    font-size: 11px;
    color: #64748b;
}
.sidebar-footer a { color: #818cf8; text-decoration: none; font-weight: 500; }

/* Typography */
h1 { font-size: 1.8rem !important; font-weight: 700 !important; color: #f3f4f6 !important; }
h2 { font-size: 1.2rem !important; font-weight: 600 !important; color: #cbd5e1 !important; }
h3 { font-size: 1rem !important; font-weight: 600 !important; color: #a5b4fc !important; }
p, span, label, li { color: #cbd5e1; }

/* Buttons — gradient indigo→violet with subtle glow */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 0.5rem 1rem !important;
    width: 100% !important;
    box-shadow: 0 2px 12px rgba(99,102,241,0.28) !important;
    transition: opacity 0.2s, box-shadow 0.2s !important;
}
.stButton > button:hover { opacity: 0.92 !important; box-shadow: 0 3px 18px rgba(139,92,246,0.4) !important; }
.stDownloadButton > button {
    background: transparent !important; color: #a5b4fc !important;
    border: 1px solid #6366f1 !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 13px !important;
}
input, textarea, .stTextArea textarea {
    background-color: #131826 !important; color: #e5e7eb !important;
    border: 1px solid #2a3244 !important;
}

/* Cognitive loop step cards */
.step-card {
    background: #131826; border: 1px solid #1f2637; border-radius: 8px;
    padding: 8px 12px; margin: 4px 0; font-size: 13px; color: #e5e7eb;
    display: flex; justify-content: space-between; align-items: center; gap: 12px;
}
.step-card.plan { border-left: 3px solid #64748b; }
.step-card.execute { border-left: 3px solid #10b981; }
.step-card.critic { border-left: 3px solid #818cf8; }
.step-card.replan { border-left: 3px solid #f59e0b; }
.step-card.complete { border-left: 3px solid #10b981; background: #0f2318; }
.step-card .ts { color: #64748b; font-size: 11px; white-space: nowrap; }

/* Report card + markdown */
[data-testid="stMarkdownContainer"] h2 {
    border-bottom: 1px solid #2a3244; padding-bottom: 4px; margin-top: 1.3rem;
}
[data-testid="stMarkdownContainer"] table { border-collapse: collapse; width: 100%; margin: 8px 0; }
[data-testid="stMarkdownContainer"] th, [data-testid="stMarkdownContainer"] td {
    border: 1px solid #2a3244; padding: 6px 10px;
}
[data-testid="stMarkdownContainer"] thead th { background: #6366f1; color: #fff; }
[data-testid="stMarkdownContainer"] tbody tr:nth-child(even) { background: #131826; }
[data-testid="stMarkdownContainer"] li { margin-bottom: 4px; }
[data-testid="stMarkdownContainer"] a { color: #a5b4fc; }
div[data-testid="stExpander"] { border: 1px solid #1f2637; border-radius: 8px; }

/* Badges */
.badge { display:inline-block; background:#1e213a; color:#a5b4fc; border-radius:999px;
    padding:2px 10px; font-size:12px; font-weight:600; }
.badge.live { background:#0f2318; color:#34d399; }
.wordcount { float:right; background:#131826; border:1px solid #2a3244; border-radius:999px;
    padding:2px 10px; font-size:12px; color:#94a3b8; }

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important; border-radius: 4px !important;
}

/* Empty state */
.empty-state { text-align:center; color:#cbd5e1; margin-top:3rem; }
.empty-state .big { font-size:3rem; }
.empty-state h2 { border:none !important; color:#f3f4f6 !important; }
.empty-state p { color:#94a3b8; max-width:620px; margin:0.5rem auto; font-size:15px; }

/* Sidebar branding + labels */
.brand { font-size:1.4rem; font-weight:700; color:#f3f4f6; }
.brand-sub { font-size:12px; color:#94a3b8; margin-top:-4px; }
.brand-ver { font-size:11px; color:#64748b; }
.side-label { font-size:12px; color:#64748b; font-weight:600; text-transform:uppercase;
    letter-spacing:0.04em; margin:2px 0; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

st.title("🔬 ARIA — Autonomous Research Intelligence Agent")
st.caption("Plans → executes → critiques → replans → reports. Watch the cognitive loop run live.")


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _fmt_elapsed(delta) -> str:
    secs = int(delta.total_seconds())
    return f"{secs // 60}m {secs % 60}s"


_KIND = {
    "memory_reader": "plan",
    "planner": "plan",
    "executor": "execute",
    "critic": "critic",
    "replanner": "replan",
    "memory_writer": "plan",
    "report_generator": "complete",
}


def _kind_for(node: str) -> str:
    return _KIND.get(node, "plan")


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


def _card_html(s: dict) -> str:
    return (
        f"<div class='step-card {s.get('kind', 'plan')}'>"
        f"<span>{s['label']}</span><span class='ts'>{s['ts']}</span></div>"
    )


def _render_feed(slot, steps: list[dict]) -> None:
    """Render the last 5 steps as styled cards, newest first."""
    slot.markdown("".join(_card_html(s) for s in reversed(steps[-5:])), unsafe_allow_html=True)


def _render_progress(slot, done: int, total: int, elapsed: str = "", complete: bool = False) -> None:
    """Thick progress bar; indigo while running, green + elapsed when complete."""
    pct = int(done / total * 100) if total else 0
    color = "#10b981" if complete else "#6366f1"
    if complete:
        label = f"Complete · {total} tasks · {elapsed}"
        pct = 100
    elif total:
        label = f"Researching… {done} of {total} tasks"
    else:
        label = "Starting…"
    slot.markdown(
        f"<div style='background:#1f2637;border-radius:6px;height:20px;width:100%;"
        f"overflow:hidden;margin:4px 0;'>"
        f"<div style='width:{max(pct, 14)}%;background:{color};height:100%;display:flex;"
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

    with st.expander(f"Show all {len(steps)} steps", expanded=False):
        for ti in sorted(by_ti):
            if ti < 0:
                st.markdown("**· Setup / final**")
            else:
                sc = score_by_ti.get(ti)
                st.markdown(f"**· Task {ti + 1}**" + (f" — critic {sc}/10" if sc is not None else ""))
            st.markdown("".join(_card_html(s) for s in by_ti[ti]), unsafe_allow_html=True)


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


def _render_report(
    report: str, session_id: str, goal: str, key: str, memory_stats: dict | None = None
) -> None:
    """Render the report inside a clean card: title + word-count badge, body, footer."""
    st.write("")
    if not report:
        st.info("No report was produced.")
        return
    wc = len(report.split())
    with st.container(border=True):
        st.markdown(
            f"<h2>{goal}<span class='wordcount'>~{wc} words</span></h2>",
            unsafe_allow_html=True,
        )
        st.markdown(report)
        st.divider()
        left, right = st.columns([2, 1])
        with left:
            ms = memory_stats or {}
            st.caption(
                f"Retrieved {ms.get('retrieved', 0)} memories from "
                f"{ms.get('total', 0)} previous runs"
            )
        with right:
            _pdf_download_button(report, session_id, goal, key=key)
        with st.expander("📋 Copy markdown"):
            st.code(report, language="markdown")


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

    header_slot = st.empty()
    header_slot.markdown(
        "### 🧠 Cognitive Loop &nbsp; <span class='badge live'>● Live</span>",
        unsafe_allow_html=True,
    )
    prog_slot = st.empty()
    feed_slot = st.empty()
    _render_progress(prog_slot, 0, 0)

    steps: list[dict] = []
    acc: dict = {}
    enabled_sources = st.session_state.get("enabled_sources") or None
    started = datetime.now()

    try:
        for update in aria_graph.stream(initial_state(clean_goal, sid, enabled_sources)):
            for node, payload in update.items():
                if not isinstance(payload, dict):
                    continue
                _merge(acc, payload)
                ti = _ti_for(node, payload, acc)
                for label in _cards_for(node, payload, acc):
                    steps.append({"ti": ti, "label": label, "ts": _now(), "kind": _kind_for(node)})
            _render_feed(feed_slot, steps)
            total = len(acc.get("subtasks", []))
            done = min(acc.get("current_task_index", 0), total) if total else 0
            _render_progress(prog_slot, done, total)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Run failed: {exc}")
        return

    total = len(acc.get("subtasks", []))
    elapsed = _fmt_elapsed(datetime.now() - started)
    header_slot.markdown(
        "### 🧠 Cognitive Loop &nbsp; <span class='badge'>✓ Complete</span>",
        unsafe_allow_html=True,
    )
    _render_progress(prog_slot, total, total, elapsed=elapsed, complete=True)

    acc.setdefault("goal", clean_goal)
    acc["session_id"] = sid
    save_session(sid, acc)

    results = sorted(acc.get("results", []), key=lambda r: r.get("task_index", 0))
    _render_show_all(steps, results)
    _render_report(
        acc.get("final_report", ""), sid, clean_goal, key="dl-run",
        memory_stats=acc.get("memory_stats"),
    )


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
    _render_report(
        saved.get("final_report", ""), session_id, saved.get("goal", ""), key="dl-saved",
        memory_stats=saved.get("memory_stats"),
    )


# --- Sidebar controls ------------------------------------------------------

with st.sidebar:
    if st.button("➕ New Research", use_container_width=True):
        for k in ("active_session", "prefill_goal", "history_show"):
            st.session_state.pop(k, None)
        st.rerun()

    goal_input = st.text_area(
        "What do you want to research?",
        value=st.session_state.get("prefill_goal", ""),
        placeholder="e.g. How does transformer attention work?",
        height=100,
    )
    start = st.button("🚀 Run ARIA", type="primary", use_container_width=True)

    # Initialize source toggles ONCE so user (de)selections persist across reruns.
    _available = set(available_source_names())
    _labels = {"web": "Web", "arxiv": "arXiv", "wikipedia": "Wikipedia", "github": "GitHub"}
    for _name in _labels:
        _key = f"src-{_name}"
        if _key not in st.session_state:
            st.session_state[_key] = _name in _available

    # Live line under the Run button — updates immediately as sources are toggled.
    _active = [_labels[n] for n in _labels if st.session_state.get(f"src-{n}") and n in _available]
    st.markdown(
        "<div class='brand-sub'>Searching: "
        + (" · ".join(_active) if _active else "no sources selected")
        + "</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("<div class='side-label'>Sources</div>", unsafe_allow_html=True)
    for name, label in _labels.items():
        avail = name in _available
        st.checkbox(label if avail else f"{label} (no key)", key=f"src-{name}", disabled=not avail)
    st.session_state["enabled_sources"] = [
        n for n in _labels if st.session_state.get(f"src-{n}") and n in _available
    ]

    st.divider()
    st.markdown("<div class='side-label'>History</div>", unsafe_allow_html=True)
    sessions = list_sessions()
    if not sessions:
        st.caption("No sessions yet.")
    show_n = st.session_state.get("history_show", 5)
    for s in sessions[:show_n]:
        dot = "🟢" if s["status"] == "complete" else "🟡"
        label = f"{dot} {s['goal'][:35]} · {s['created_at'][:10]}"
        if st.button(label, key=f"sess-{s['id']}", use_container_width=True):
            st.session_state["active_session"] = s["id"]
            st.session_state["prefill_goal"] = s["goal"]
            st.rerun()
    if len(sessions) > show_n:
        if st.button(f"View all ({len(sessions)})", key="view-all", use_container_width=True):
            st.session_state["history_show"] = len(sessions)
            st.rerun()

    st.markdown(
        '<div class="sidebar-footer">Built by '
        '<a href="https://aarti-tech-portfolio.vercel.app" target="_blank">Aarti Panchal</a>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    """Perplexity-style empty state with clickable suggestion chips."""
    st.markdown(
        "<div class='empty-state'>"
        "<div class='big'>🔬</div>"
        "<h2>Research anything.</h2>"
        "<p>ARIA plans your research into subtasks, searches across web + arXiv + "
        "Wikipedia + GitHub, critiques each finding, replans the weak ones, and "
        "writes a structured report.</p>"
        "<p style='color:#9ca3af'>Try one:</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    suggestions = [
        "How does RAG work?",
        "Compare Redis vs Memcached",
        "What is LangGraph?",
        "Explain transformer attention",
    ]
    cols = st.columns(len(suggestions))
    for col, text in zip(cols, suggestions, strict=False):
        with col:
            if st.button(text, key=f"sug-{text[:14]}"):
                st.session_state["prefill_goal"] = text
                st.rerun()


if start:
    run(goal_input)
elif st.session_state.get("active_session"):
    show_saved_session(st.session_state["active_session"])
else:
    render_empty_state()
