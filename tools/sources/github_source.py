"""GitHub repository source (public search; GITHUB_TOKEN optional for rate limits)."""

import os

from config import get_logger
from tools.sources.base import SearchResult

logger = get_logger(__name__)


class GitHubSource:
    name = "github"
    description = "Public repositories from GitHub (GITHUB_TOKEN optional)"

    def is_available(self) -> bool:
        return True  # public search works unauthenticated (token raises rate limits)

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        try:
            from github import Auth, Github

            token = os.getenv("GITHUB_TOKEN")
            gh = Github(auth=Auth.Token(token)) if token else Github()
            repos = gh.search_repositories(query=query)

            results = []
            for repo in repos[:max_results]:
                desc = repo.description or "(no description)"
                results.append(
                    SearchResult(
                        title=repo.full_name,
                        content=f"{desc}\n⭐ {repo.stargazers_count} · {repo.language or 'n/a'}"[:600],
                        url=repo.html_url,
                        source_type="github",
                        relevance_score=0.5,
                    )
                )
            return results
        except Exception as exc:  # noqa: BLE001
            logger.error("GitHub search failed: %s", exc)
            return []
