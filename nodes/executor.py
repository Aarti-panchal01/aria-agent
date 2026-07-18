"""
Executor node for the ARIA research agent.

For the current subtask, asks the LLM for a focused search query, runs a
real Tavily web search, and records the finding. No hardcoded knowledge:
every finding comes from live search.
"""

import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config import GROQ_MODEL, get_logger, invoke_with_retry, require_groq_key
from state import AgentState
from tools.search import web_search

logger = get_logger(__name__)

_QUERY_SYSTEM_PROMPT = (
    "You are a search query generator. Given a research subtask, output ONE "
    "specific, focused web search query.\n"
    "Rules:\n"
    "- Include the specific entities and attribute being researched\n"
    "- Maximum 8 words\n"
    "- Output the query string ONLY, nothing else"
)


def _result_id_for_index(state: AgentState, task_index: int) -> str:
    """
    Return a stable id for the finding at ``task_index``.

    Re-uses the existing finding's id when this task is being re-executed
    (e.g. after a targeted replan) so the merge reducer updates it in place
    rather than appending a duplicate. Otherwise mints a fresh uuid4.
    """
    for r in state.get("results", []):
        if r.get("task_index") == task_index:
            return r["id"]
    return str(uuid.uuid4())


def executor_node(state: AgentState) -> dict:
    """
    Executor node: research the current subtask via live web search.

    Args:
        state (AgentState): Current agent state.

    Returns:
        dict: A single finding (merged by id) and the advanced task index.
    """
    llm = ChatGroq(model=GROQ_MODEL, api_key=require_groq_key(), temperature=0.7)

    subtasks = state.get("subtasks", [])
    current_index = state.get("current_task_index", 0)

    if current_index >= len(subtasks):
        return {"current_task_index": current_index}

    current_subtask = subtasks[current_index]

    # Step 1: ask the LLM for a focused search query.
    messages = [
        SystemMessage(content=_QUERY_SYSTEM_PROMPT),
        HumanMessage(content=current_subtask),
    ]
    response = invoke_with_retry(llm, messages, context="executor")
    search_query = response.content.strip() if response else current_subtask[:100]

    # Step 2: run the search (web_search always returns a str).
    search_results = web_search.invoke(search_query)

    finding = {
        "id": _result_id_for_index(state, current_index),
        "task_index": current_index,
        "task": current_subtask,
        "query": search_query,
        "output": search_results,
        "critic": None,
        "score": 0,
    }
    logger.info("Executed subtask %d/%d: %s", current_index + 1, len(subtasks), current_subtask)

    return {"results": [finding], "current_task_index": current_index + 1}
