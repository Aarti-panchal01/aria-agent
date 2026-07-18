"""Tests for research sources and the multi-source aggregator (all mocked)."""

from unittest.mock import Mock, patch

from tools.sources.aggregator import aggregate_search, format_results, get_sources
from tools.sources.arxiv_source import ArxivSource
from tools.sources.base import SearchResult
from tools.sources.github_source import GitHubSource
from tools.sources.web_source import WebSource
from tools.sources.wikipedia_source import WikipediaSource


def test_web_source_parses_results():
    fake = [{"title": "T", "content": "C" * 10, "url": "http://x", "score": 0.9}]
    with patch("tools.sources.web_source.os.getenv", return_value="key"), patch(
        "langchain_community.tools.tavily_search.TavilySearchResults"
    ) as mock_tav:
        mock_tav.return_value.invoke.return_value = fake
        out = WebSource().search("q", 3)
    assert len(out) == 1
    assert out[0].source_type == "web"
    assert out[0].url == "http://x"


def test_arxiv_source_parses_results():
    paper = Mock(title="Attention Is All You Need", summary="S" * 20, entry_id="http://arxiv/1")
    author = Mock()
    author.name = "Ashish Vaswani"  # note: Mock(name=...) is reserved, so set attr explicitly
    paper.authors = [author]
    with patch("arxiv.Search"), patch("arxiv.Client") as mock_client:
        mock_client.return_value.results.return_value = [paper]
        out = ArxivSource().search("transformers", 3)
    assert len(out) == 1
    assert out[0].source_type == "arxiv"
    assert "arxiv" in out[0].url


def test_wikipedia_source_parses_results():
    page = Mock(title="RAG", summary="Retrieval augmented generation", url="http://wiki/RAG")
    with patch("wikipedia.search", return_value=["RAG"]), patch(
        "wikipedia.page", return_value=page
    ):
        out = WikipediaSource().search("RAG", 3)
    assert len(out) == 1
    assert out[0].source_type == "wikipedia"


def test_github_source_parses_results():
    repo = Mock(
        full_name="langchain-ai/langgraph",
        description="graph agents",
        html_url="http://gh/lg",
        stargazers_count=1000,
        language="Python",
    )
    fake_repos = Mock()
    fake_repos.__getitem__ = lambda self, s: [repo]
    with patch("tools.sources.github_source.os.getenv", return_value=None), patch(
        "github.Github"
    ) as mock_gh:
        mock_gh.return_value.search_repositories.return_value = fake_repos
        out = GitHubSource().search("langgraph", 3)
    assert len(out) == 1
    assert out[0].source_type == "github"
    assert out[0].url == "http://gh/lg"


def test_source_failure_returns_empty_list():
    with patch("tools.sources.web_source.os.getenv", return_value="key"), patch(
        "langchain_community.tools.tavily_search.TavilySearchResults"
    ) as mock_tav:
        mock_tav.return_value.invoke.side_effect = Exception("boom")
        assert WebSource().search("q") == []


def test_aggregate_dedupes_by_url_and_ranks():
    a = SearchResult(title="A", content="c", url="http://dup", source_type="web", relevance_score=0.9)
    b = SearchResult(title="B", content="c", url="http://dup", source_type="arxiv", relevance_score=0.4)
    c = SearchResult(title="C", content="c", url="http://uniq", source_type="wikipedia", relevance_score=0.6)

    class _S:
        name = "web"
        description = "x"

        def is_available(self):
            return True

        def search(self, query, max_results=5):
            return [a, b, c]

    with patch("tools.sources.aggregator.get_sources", return_value=[_S()]):
        out = aggregate_search("q")
    urls = [r.url for r in out]
    assert urls == ["http://dup", "http://uniq"]  # deduped, ranked by relevance
    assert out[0].relevance_score == 0.9  # higher-scored dup kept


def test_get_sources_respects_enabled_and_availability():
    names = {s.name for s in get_sources(enabled=["arxiv", "wikipedia"])}
    assert names == {"arxiv", "wikipedia"}  # both keyless => available


def test_format_results_has_source_attribution():
    r = SearchResult(title="T", content="C", url="http://x", source_type="arxiv", relevance_score=0.7)
    text = format_results([r])
    assert "[ARXIV]" in text
    assert "http://x" in text
    assert format_results([]) == "No results found from any source."
