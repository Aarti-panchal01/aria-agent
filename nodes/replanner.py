"""
Replanner node for the ARIA research agent.

Targeted replanning: instead of throwing away the whole plan, this rewrites
ONLY the subtask that the critic flagged as weak, using the critic's
specific instruction, then rewinds the task index so the executor re-runs
just that one subtask.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config import GROQ_MODEL, get_logger, invoke_with_retry, require_groq_key
from state import AgentState

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a research re-planner. A subtask produced a weak result. Rewrite "
    "that single subtask so a fresh web search will return better information, "
    "following the critic's instruction. Return ONLY the rewritten subtask as "
    "one line, nothing else."
)


def replanner_node(state: AgentState) -> dict:
    """
    Rewrite the most-recently-executed subtask and rewind to re-run it.

    Args:
        state (AgentState): Current agent state.

    Returns:
        dict: Updated ``subtasks``, rewound ``current_task_index`` (to the
        failing task), incremented ``replan_count``, and a cleared
        ``replan_instruction``.
    """
    subtasks = list(state.get("subtasks", []))
    target_index = state.get("current_task_index", 1) - 1  # last executed subtask
    replan_count = state.get("replan_count", 0)
    instruction = state.get("replan_instruction", "")

    if target_index < 0 or target_index >= len(subtasks):
        # Nothing sensible to replan; move on without changing the plan.
        return {"replan_instruction": ""}

    old_task = subtasks[target_index]

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Original subtask:\n{old_task}\n\n"
                f"Critic instruction:\n{instruction}"
            )
        ),
    ]
    response = invoke_with_retry(llm(), messages, context="replanner")
    improved = response.content.strip() if response else old_task
    if not improved:
        improved = old_task

    subtasks[target_index] = improved
    logger.info(
        "Replanned task %d (attempt %d): %s -> %s",
        target_index,
        replan_count + 1,
        old_task,
        improved,
    )

    return {
        "subtasks": subtasks,
        "current_task_index": target_index,
        "replan_count": replan_count + 1,
        "replan_instruction": "",
    }


def llm():
    """Construct the replanner LLM (kept as a helper for easy test patching)."""
    return ChatGroq(model=GROQ_MODEL, api_key=require_groq_key(), temperature=0.4)
