"""
Planner node for the ARIA research agent.

Breaks the research goal into concrete, distinct subtasks, optionally
informed by relevant findings retrieved from memory.
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config import GROQ_MODEL, get_logger, invoke_with_retry, require_groq_key
from state import AgentState
from tools.sources.aggregator import available_source_names

logger = get_logger(__name__)

NUM_SUBTASKS = 6

_SOURCE_LABELS = {
    "web": "general web search",
    "arxiv": "arXiv academic papers",
    "wikipedia": "Wikipedia articles",
    "github": "GitHub repositories",
}


def _parse_subtasks(response_text: str) -> list[str]:
    """
    Parse a numbered-list LLM response into a list of subtask strings.

    Handles ``1.``, ``1)`` and ``1:`` numbering styles.

    Args:
        response_text (str): The LLM response text.

    Returns:
        list[str]: Parsed subtasks (falls back to a single generic task).
    """
    subtasks = []
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^\d+[\.\)\:]\s*(.+)$", line)
        if match:
            task = match.group(1).strip()
            if task:
                subtasks.append(task)
    return subtasks if subtasks else ["Research the topic"]


def planner_node(state: AgentState) -> dict:
    """
    Plan node: break the research goal into distinct subtasks.

    Consults memory context if available to avoid redundant work.

    Args:
        state (AgentState): Current agent state.

    Returns:
        dict: ``subtasks`` and a reset ``current_task_index``.
    """
    llm = ChatGroq(model=GROQ_MODEL, api_key=require_groq_key(), temperature=0.3)

    goal = state.get("goal", "")
    memory_context = state.get("memory_context", "")

    # Constrain planning to sources ARIA can actually query.
    enabled = state.get("enabled_sources") or available_source_names()
    source_desc = ", ".join(_SOURCE_LABELS.get(s, s) for s in enabled) or "general web search"

    system_prompt = (
        f"You are a research planning AI. Break the given research goal into "
        f"{NUM_SUBTASKS} concrete subtasks that are each meaningfully different "
        f"and together give broad coverage of the topic.\n"
        f"ARIA can ONLY gather information from these sources: {source_desc}.\n"
        f"Plan tasks that are answerable using ONLY these sources. NEVER suggest "
        f"academic databases (JSTOR, EBSCO, ProQuest, Scopus), paywalled tools, "
        f"surveys, interviews, lab experiments, or any source not in that list.\n"
        f"Return ONLY a numbered list, nothing else."
    )
    if memory_context:
        system_prompt += (
            f"\n\nConsider these past findings to avoid redundant work:\n{memory_context}"
        )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Research goal: {goal}"),
    ]

    response = invoke_with_retry(llm, messages, context="planner")
    subtasks = _parse_subtasks(response.content) if response else ["Research the topic"]

    logger.info("Planner produced %d subtasks", len(subtasks))
    return {"subtasks": subtasks, "current_task_index": 0}
