"""arXiv academic paper source (no API key required)."""

from config import get_logger
from tools.sources.base import SearchResult

logger = get_logger(__name__)


class ArxivSource:
    name = "arxiv"
    description = "Academic papers and preprints from arXiv"

    def is_available(self) -> bool:
        return True  # public API, no key needed

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        import arxiv

        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            results = []
            for paper in arxiv.Client().results(search):
                authors = ", ".join(a.name for a in paper.authors[:3])
                results.append(
                    SearchResult(
                        title=paper.title,
                        content=f"Authors: {authors}\n{(paper.summary or '')[:600]}",
                        url=paper.entry_id,
                        source_type="arxiv",
                        relevance_score=0.7,
                    )
                )
            return results
        except Exception as exc:  # noqa: BLE001
            logger.error("arXiv search failed: %s", exc)
            return []
