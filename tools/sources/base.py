"""
Shared interface for ARIA research sources.

A source turns a query into a list of ``SearchResult`` objects. Concrete
sources (web, arXiv, Wikipedia, GitHub) implement the ``ResearchSource``
protocol so the aggregator can treat them uniformly.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single normalized result from any research source."""

    title: str
    content: str
    url: str
    source_type: str  # "arxiv" | "wikipedia" | "github" | "web"
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)


@runtime_checkable
class ResearchSource(Protocol):
    """Protocol every research source implements."""

    name: str
    description: str

    def is_available(self) -> bool:
        """True if this source can run (e.g. required API key present)."""
        ...

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Return up to ``max_results`` results for ``query``."""
        ...
