"""
Streamlit UI for ARIA — live cognitive-loop research interface.

Streams the LangGraph execution node-by-node so the user watches ARIA plan,
execute, critique, replan, and report in real time (not behind a spinner).

Run with:
    streamlit run ui/app.py
"""

import os
import sys

import streamlit as st

# Make the project root importable when launched via `streamlit run ui/app.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph import aria_graph  # noqa: E402
from main import initial_state, sanitize_goal  # noqa: E402

st.set_page_config(page_title="ARIA — Research Agent", page_icon="🔬", layout="wide")

st.title("🔬 ARIA — Autonomous Research Intelligence Agent")
st.caption(
    "Plans → executes → critiques → replans → reports. "
    "Watch the cognitive loop run live."
)

with st.sidebar:
    st.header("Run")
    goal = st.text_area(
        "Research goal",
        placeholder="Compare Redis vs Memcached for caching",
        height=100,
    )
    start = st.button("Run ARIA", type="primary", use_container_width=True)
    st.divider()
    st.markdown(
        "**Requires** `GROQ_API_KEY` and `TAVILY_API_KEY` in your environment "
        "or `.env` file."
    )
    trace_box = st.container()


def _describe_step(node: str, payload: dict) -> str:
    """Human-readable one-liner for a streamed node update."""
    if node == "memory_reader":
        stats = payload.get("memory_stats", {})
        return f"🧠 Memory: retrieved {stats.get('retrieved', 0)} of {stats.get('total', 0)} past findings"
    if node == "planner":
        subs = payload.get("subtasks", [])
        return f"🗺️ Planned {len(subs)} subtasks"
    if node == "executor":
        idx = payload.get("current_task_index", 0)
        return f"🔎 Executed subtask {idx} (web search)"
    if node == "critic":
        results = payload.get("results", [])
        if results and results[-1].get("critic"):
            c = results[-1]["critic"]
            return (
                f"⚖️ Critic scored {c['overall']}/10 "
                f"(relevance {c['relevance']}, specificity {c['specificity']}, "
                f"sources {c['source_quality']}, completeness {c['completeness']})"
            )
        return "⚖️ Critic scored the finding"
    if node == "replanner":
        return f"♻️ Replanning weak subtask (attempt {payload.get('replan_count', '?')})"
    if node == "memory_writer":
        return "💾 Saved finding to memory"
    if node == "terminator":
        return "🏁 Done" if payload.get("is_done") else "➡️ Continuing"
    if node == "report_generator":
        return "📝 Report generated"
    return f"• {node}"


def run() -> None:
    """Execute the graph, streaming progress and rendering the final report."""
    try:
        clean_goal = sanitize_goal(goal)
    except ValueError as exc:
        st.error(str(exc))
        return

    progress = st.empty()
    steps: list[str] = []
    final_state: dict = {}

    with st.status("Running ARIA…", expanded=True) as status:
        for update in aria_graph.stream(initial_state(clean_goal)):
            for node, payload in update.items():
                steps.append(_describe_step(node, payload))
                progress.markdown("\n\n".join(f"- {s}" for s in steps))
                if isinstance(payload, dict):
                    final_state.update(payload)
        status.update(label="Research complete ✅", state="complete")

    report = final_state.get("final_report", "")
    if report:
        st.subheader("📊 Research Report")
        st.code(report, language="markdown")  # gives a built-in copy button
        st.markdown(report)

    results = sorted(final_state.get("results", []), key=lambda r: r.get("task_index", 0))
    with trace_box:
        st.header("Reasoning trace")
        for r in results:
            c = r.get("critic") or {}
            with st.expander(f"Task {r.get('task_index', '?')}: {r.get('task', '')[:60]}"):
                if c:
                    st.write(
                        {
                            "overall": c.get("overall"),
                            "relevance": c.get("relevance"),
                            "specificity": c.get("specificity"),
                            "source_quality": c.get("source_quality"),
                            "completeness": c.get("completeness"),
                            "reasoning": c.get("reasoning"),
                        }
                    )
                st.caption(f"query: {r.get('query', '')}")


if start:
    run()
else:
    st.info("Enter a research goal in the sidebar and click **Run ARIA**.")
