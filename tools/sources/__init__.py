"""Multi-source research: web (Tavily), arXiv, Wikipedia, GitHub."""

from tools.sources.aggregator import (
    aggregate_search,
    all_sources,
    available_source_names,
    format_results,
    get_sources,
)
from tools.sources.base import ResearchSource, SearchResult

__all__ = [
    "aggregate_search",
    "all_sources",
    "available_source_names",
    "format_results",
    "get_sources",
    "ResearchSource",
    "SearchResult",
]
