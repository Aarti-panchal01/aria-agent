"""
Search tools for the ARIA research agent.

Provides a Tavily-backed web search tool that ALWAYS returns a string,
so downstream string handling can never raise ``TypeError``.
"""

import os
import time

from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

from config import MAX_RETRIES, RETRY_DELAY_SECONDS, get_logger

logger = get_logger(__name__)


@tool
def web_search(query: str) -> str:
    """
    Search the web using the Tavily API.

    Returns the top results formatted as text. On any failure (missing key,
    API error after retries, or no results) this returns an explanatory
    string rather than a dict — callers can safely concatenate the result.

    Args:
        query (str): The search query.

    Returns:
        str: Formatted search results, or a "Search failed: ..." message.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return "Search failed: TAVILY_API_KEY not found in environment."

    tavily = TavilySearchResults(max_results=3, tavily_api_key=tavily_api_key)

    for attempt in range(MAX_RETRIES):
        try:
            tavily_results = tavily.invoke(query)

            results = []
            for r in tavily_results:
                results.append(
                    f"Title: {r.get('title', 'No Title')}\n"
                    f"URL: {r.get('url', 'No URL')}\n"
                    f"Content: {r.get('content', 'No Content')[:600]}\n"
                )

            if not results:
                return f"No search results found for query: '{query}'"

            return "Search Results:\n\n" + "\n---\n".join(results)

        except Exception as exc:  # noqa: BLE001 - retry on any transport error
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error("Tavily search failed after %d retries: %s", MAX_RETRIES, exc)
                return f"Search failed after {MAX_RETRIES} retries: {exc}"
