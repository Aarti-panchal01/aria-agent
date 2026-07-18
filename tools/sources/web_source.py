"""Web search source (Tavily)."""

import os

from config import get_logger
from tools.sources.base import SearchResult

logger = get_logger(__name__)


class WebSource:
    name = "web"
    description = "General web search via Tavily (requires TAVILY_API_KEY)"

    def is_available(self) -> bool:
        return bool(os.getenv("TAVILY_API_KEY"))

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        from langchain_community.tools.tavily_search import TavilySearchResults

        key = os.getenv("TAVILY_API_KEY")
        if not key:
            return []
        try:
            raw = TavilySearchResults(max_results=max_results, tavily_api_key=key).invoke(query)
        except Exception as exc:  # noqa: BLE001
            logger.error("Web (Tavily) search failed: %s", exc)
            return []

        results = []
        for r in raw or []:
            results.append(
                SearchResult(
                    title=r.get("title", "No Title"),
                    content=(r.get("content", "") or "")[:600],
                    url=r.get("url", ""),
                    source_type="web",
                    relevance_score=float(r.get("score", 0.5) or 0.5),
                )
            )
        return results
