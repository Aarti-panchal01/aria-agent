"""
Streamlit UI for ARIA — real-time streaming cognitive-loop research interface.

The left panel streams each cognitive step as it happens (planner → executor →
critic → replanner → memory → report), with timestamps and a progress bar; the
final report renders on the right. The streaming IS the demo.

Run with:
    streamlit run ui/app.py
"""

import os
import sys
from datetime import datetime

import streamlit as st

# Make the project root importable when launched via `streamlit run ui/app.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph import aria_graph  # noqa: E402
from main import initial_state, sanitize_goal  # noqa: E402
from state import merge_results  # noqa: E402


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

st.title("🔬 ARIA — Autonomous Research Intelligence Agent")
st.caption(
    "Plans → executes → critiques → replans → reports. "
    "Watch the cognitive loop run live."
)


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _cards_for(node: str, payload: dict, acc: dict) -> list[str]:
    """Return timestamped step-card lines for one streamed node update."""
    ts = _now()
    cards: list[str] = []

    if node == "memory_reader":
        stats = payload.get("memory_stats", {})
        cards.append(
            f"🧠 **Memory** — retrieved {stats.get('retrieved', 0)} of "
            f"{stats.get('total', 0)} past findings · `{ts}`"
        )
    elif node == "planner":
        subs = payload.get("subtasks", [])
        cards.append(f"🗺️ **Planned** {len(subs)} subtasks · `{ts}`")
    elif node == "executor":
        results = acc.get("results", [])
        idx = payload.get("current_task_index", 0)
        total = len(acc.get("subtasks", [])) or "?"
        last = results[-1] if results else {}
        task = (last.get("task") or "")[:80]
        cards.append(f'⚡ **Executing task {idx}/{total}:** "{task}" · `{ts}`')
        out = last.get("output", "")
        n = out.count("Title:") if isinstance(out, str) else 0
        if isinstance(out, str) and out.startswith("Search failed"):
            cards.append(f"🔍 Search failed (handled gracefully) · `{ts}`")
        else:
            cards.append(f"🔍 Search returned {n} result(s) · `{ts}`")
    elif node == "critic":
        results = payload.get("results", [])
        c = (results[-1].get("critic") or {}) if results else {}
        if c:
            cards.append(
                f"🎯 **Critic scored** {c.get('overall')}/10 "
                f"(relevance:{c.get('relevance')}, specificity:{c.get('specificity')}, "
                f"source:{c.get('source_quality')}, completeness:{c.get('completeness')}) · `{ts}`"
            )
    elif node == "replanner":
        prev = acc.get("results", [])
        last_score = prev[-1].get("score", "?") if prev else "?"
        idx = payload.get("current_task_index", 0)
        cards.append(f"♻️ **Replanning task {idx}:** score was {last_score}/10 · `{ts}`")
    elif node == "memory_writer":
        cards.append(f"💾 Writing to memory… · `{ts}`")
    elif node == "report_generator":
        cards.append(f"✅ **Research complete** · `{ts}`")

    return cards


def _merge(acc: dict, payload: dict) -> None:
    """Accumulate streamed deltas into a running state snapshot."""
    for k, v in payload.items():
        if k == "results":
            acc["results"] = merge_results(acc.get("results", []), v)
        else:
            acc[k] = v


def run(goal: str) -> None:
    """Stream a research run: live feed on the left, report on the right."""
    try:
        clean_goal = sanitize_goal(goal)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.markdown(f"**Session goal:** {clean_goal} · _started {_now()}_")
    feed_col, report_col = st.columns([1, 1], gap="large")

    with feed_col:
        st.subheader("🧠 Cognitive loop (live)")
        progress = st.progress(0.0, text="Starting…")
        feed = st.empty()
    with report_col:
        st.subheader("📊 Research report")
        report_slot = st.empty()
        report_slot.info("The report will appear here when the run completes.")

    steps: list[str] = []
    acc: dict = {}

    try:
        for update in aria_graph.stream(initial_state(clean_goal)):
            for node, payload in update.items():
                if isinstance(payload, dict):
                    _merge(acc, payload)
                    steps.extend(_cards_for(node, payload, acc))

            feed.markdown("\n\n".join(f"- {s}" for s in reversed(steps)))

            total = len(acc.get("subtasks", []))
            done = min(acc.get("current_task_index", 0), total) if total else 0
            frac = (done / total) if total else 0.0
            progress.progress(frac, text=f"{done}/{total or '?'} tasks")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Run failed: {exc}")
        return

    progress.progress(1.0, text="Complete")
    report = acc.get("final_report", "")
    results = sorted(acc.get("results", []), key=lambda r: r.get("task_index", 0))

    with report_col:
        report_slot.empty()
        if report:
            st.markdown(report)
            with st.expander("Copy raw markdown"):
                st.code(report, language="markdown")
        st.session_state["last_report"] = report
        st.session_state["last_goal"] = clean_goal
        st.session_state["last_results"] = results

    with st.sidebar:
        st.header("Reasoning trace")
        for r in results:
            c = r.get("critic") or {}
            with st.expander(f"Task {r.get('task_index', '?')}: {r.get('task', '')[:50]}"):
                if c:
                    st.write({k: c.get(k) for k in (
                        "overall", "relevance", "specificity",
                        "source_quality", "completeness", "reasoning")})
                st.caption(f"query: {r.get('query', '')}")


# --- Sidebar controls ------------------------------------------------------

with st.sidebar:
    st.header("Run")
    goal_input = st.text_area(
        "Research goal",
        placeholder="Compare Redis vs Memcached for caching",
        height=100,
    )
    start = st.button("Run ARIA", type="primary", use_container_width=True)
    st.divider()
    st.caption("Requires GROQ_API_KEY and TAVILY_API_KEY (env, .env, or Streamlit secrets).")

if start:
    run(goal_input)
else:
    st.info("Enter a research goal in the sidebar and click **Run ARIA**.")
