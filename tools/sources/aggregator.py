"""
Multi-source aggregator.

Queries every enabled/available research source in parallel (a thread pool —
the source libraries are synchronous/blocking, so threads give real parallelism
where asyncio would not), then merges, deduplicates by URL, and ranks by
relevance.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from config import get_logger
from tools.sources.arxiv_source import ArxivSource
from tools.sources.base import SearchResult
from tools.sources.github_source import GitHubSource
from tools.sources.web_source import WebSource
from tools.sources.wikipedia_source import WikipediaSource

logger = get_logger(__name__)

# Registry of all known sources (instances).
_REGISTRY = [WebSource(), ArxivSource(), WikipediaSource(), GitHubSource()]

_ICON = {"web": "🌐", "arxiv": "🔬", "wikipedia": "📖", "github": "🐙"}


def all_sources() -> list:
    """Return every registered source instance."""
    return list(_REGISTRY)


def available_source_names() -> list[str]:
    """Names of sources that can currently run (keys present etc.)."""
    return [s.name for s in _REGISTRY if s.is_available()]


def get_sources(enabled: list[str] | None = None) -> list:
    """
    Resolve which sources to query.

    Args:
        enabled: If given, restrict to these source names. Sources that are not
            available (missing key) are always excluded.

    Returns:
        list of source instances to query.
    """
    chosen = []
    for s in _REGISTRY:
        if enabled is not None and s.name not in enabled:
            continue
        if not s.is_available():
            continue
        chosen.append(s)
    return chosen


def aggregate_search(
    query: str, enabled: list[str] | None = None, max_results: int = 3
) -> list[SearchResult]:
    """Query all resolved sources in parallel, dedup by URL, rank by relevance."""
    sources = get_sources(enabled)
    if not sources:
        return []

    collected: list[SearchResult] = []
    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        futures = {pool.submit(s.search, query, max_results): s for s in sources}
        for fut in as_completed(futures):
            src = futures[fut]
            try:
                collected.extend(fut.result(timeout=45) or [])
            except Exception as exc:  # noqa: BLE001
                logger.error("Source %s failed: %s", src.name, exc)

    seen: set[str] = set()
    deduped: list[SearchResult] = []
    for r in sorted(collected, key=lambda x: x.relevance_score, reverse=True):
        key = r.url or f"{r.source_type}:{r.title}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def format_results(results: list[SearchResult]) -> str:
    """Format results as attributed text: '[ARXIV] title … [WEB] title …'."""
    if not results:
        return "No results found from any source."
    blocks = ["Search Results:\n"]
    for r in results:
        icon = _ICON.get(r.source_type, "•")
        blocks.append(
            f"{icon} [{r.source_type.upper()}] {r.title}\nURL: {r.url}\n{r.content}\n"
        )
    return "\n---\n".join(blocks)
