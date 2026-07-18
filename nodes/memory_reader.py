"""
Memory reader node for the ARIA research agent.

Retrieves past findings from ChromaDB, but only injects ones that are BOTH
above a similarity threshold AND pass an LLM relevance check against the current
goal — so an unrelated prior run never pollutes the context.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config import GROQ_MODEL, get_logger, invoke_with_retry, require_groq_key
from memory.chroma_store import retrieve_candidates
from state import AgentState

logger = get_logger(__name__)

_NONE_MSG = "No relevant memories from previous runs."

_RELEVANCE_SYSTEM = (
    "You judge relevance. Given a research goal and a past research finding, "
    "answer with ONLY 'Yes' or 'No' — is the past finding relevant to the goal?"
)


def _is_relevant(llm, goal: str, memory_text: str) -> bool:
    """Ask the LLM whether a past finding is relevant to the current goal."""
    messages = [
        SystemMessage(content=_RELEVANCE_SYSTEM),
        HumanMessage(
            content=(
                f"Research goal: {goal}\n\n"
                f"Past finding:\n{memory_text[:500]}\n\n"
                f"Is this past finding relevant to the research goal? Answer Yes or No."
            )
        ),
    ]
    resp = invoke_with_retry(llm, messages, context="memory_relevance")
    if resp is None:
        return False  # on failure, err toward NOT injecting stale context
    return resp.content.strip().lower().startswith("y")


def memory_reader_node(state: AgentState) -> dict:
    """
    Retrieve and relevance-filter past findings for the research goal.

    Args:
        state (AgentState): Current agent state.

    Returns:
        dict: ``memory_context`` and ``memory_stats`` ({"retrieved", "total"}).
    """
    goal = state.get("goal", "")
    if not goal:
        return {"memory_context": "", "memory_stats": {"retrieved": 0, "total": 0}}

    candidates, total = retrieve_candidates(query=goal, n=3, threshold=0.75)
    if not candidates:
        logger.info("Memory: no candidates above threshold (of %d stored)", total)
        return {"memory_context": _NONE_MSG, "memory_stats": {"retrieved": 0, "total": total}}

    llm = ChatGroq(model=GROQ_MODEL, api_key=require_groq_key(), temperature=0.0)
    relevant = [c for c in candidates if _is_relevant(llm, goal, c["content"])]

    if not relevant:
        logger.info(
            "Memory: %d candidates found but none passed relevance check (of %d stored)",
            len(candidates),
            total,
        )
        return {"memory_context": _NONE_MSG, "memory_stats": {"retrieved": 0, "total": total}}

    lines = [f"Relevant Past Findings for '{goal}':\n"]
    for idx, c in enumerate(relevant, 1):
        lines.append(
            f"Finding {idx} (relevance {c['similarity']}):\n"
            f"  Task: {c['task']}\n"
            f"  Content: {c['content'][:400]}\n"
        )
    logger.info(
        "Memory: injected %d relevant of %d candidates (%d stored)",
        len(relevant),
        len(candidates),
        total,
    )
    return {
        "memory_context": "\n".join(lines),
        "memory_stats": {"retrieved": len(relevant), "total": total},
    }
