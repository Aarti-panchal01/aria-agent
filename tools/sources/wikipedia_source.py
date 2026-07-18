"""Wikipedia encyclopedic source (no API key required)."""

from config import get_logger
from tools.sources.base import SearchResult

logger = get_logger(__name__)


class WikipediaSource:
    name = "wikipedia"
    description = "Encyclopedic summaries from Wikipedia"

    def is_available(self) -> bool:
        return True  # public API, no key needed

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        import wikipedia

        try:
            titles = wikipedia.search(query, results=max_results)
        except Exception as exc:  # noqa: BLE001
            logger.error("Wikipedia search failed: %s", exc)
            return []

        results = []
        for title in titles:
            try:
                page = wikipedia.page(title, auto_suggest=False)
            except Exception:  # noqa: BLE001 - disambiguation / missing page: skip
                continue
            results.append(
                SearchResult(
                    title=page.title,
                    content=(page.summary or "")[:600],
                    url=page.url,
                    source_type="wikipedia",
                    relevance_score=0.6,
                )
            )
        return results
