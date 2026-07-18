"""
Critic node for the ARIA research agent.

Scores the most recent finding across multiple dimensions using structured
output (Pydantic), updates that finding in place (same id, no duplicate),
and surfaces a targeted replan instruction when quality is low.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config import (
    GROQ_MODEL,
    REPLAN_THRESHOLD,
    get_logger,
    invoke_with_retry,
    require_groq_key,
)
from schemas import CriticScore
from state import AgentState

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a meticulous research critic. Score the given research finding "
    "against the subtask it was meant to answer, across the requested "
    "dimensions (0-10 each). Set replan_needed to true only if the finding is "
    "too weak to use; when true, give a specific replan_instruction describing "
    "a better search angle. Be honest and calibrated."
)


def _fallback_score() -> dict:
    """Neutral critic result used when the LLM call fails."""
    return {
        "relevance": 5,
        "specificity": 5,
        "source_quality": 5,
        "completeness": 5,
        "overall": 5,
        "reasoning": "Critic LLM unavailable; assigned neutral fallback score.",
        "replan_needed": False,
        "replan_instruction": "",
    }


def critic_node(state: AgentState) -> dict:
    """
    Critic node: score the last finding and decide whether to replan.

    Args:
        state (AgentState): Current agent state.

    Returns:
        dict: The updated finding (same id) plus ``replan_instruction``.
    """
    results = state.get("results", [])
    if not results:
        return {}

    last_result = dict(results[-1])  # copy; keep the same id so merge updates it
    task = last_result.get("task", "Unknown task")
    output = last_result.get("output", "No output")

    llm = ChatGroq(model=GROQ_MODEL, api_key=require_groq_key(), temperature=0.2)
    structured_llm = llm.with_structured_output(CriticScore)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Subtask:\n{task}\n\nFinding:\n{output}"),
    ]

    score_obj = invoke_with_retry(structured_llm, messages, context="critic")
    if isinstance(score_obj, CriticScore):
        critic_dict = score_obj.model_dump()
    else:
        critic_dict = _fallback_score()

    last_result["critic"] = critic_dict
    last_result["score"] = critic_dict["overall"]

    replan_instruction = ""
    if critic_dict["overall"] < REPLAN_THRESHOLD and critic_dict.get("replan_needed"):
        replan_instruction = critic_dict.get("replan_instruction", "") or (
            "Search for more specific, authoritative sources on this subtask."
        )

    logger.info(
        "Critic scored task %s: overall=%d (relevance=%d specificity=%d "
        "source=%d completeness=%d) replan=%s",
        last_result.get("task_index"),
        critic_dict["overall"],
        critic_dict["relevance"],
        critic_dict["specificity"],
        critic_dict["source_quality"],
        critic_dict["completeness"],
        bool(replan_instruction),
    )

    return {"results": [last_result], "replan_instruction": replan_instruction}
