# Changelog

All notable changes to ARIA are documented here.
This project follows [Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-07-18

A correctness-and-credibility overhaul. The core loop is now provably correct,
the demo generalizes to any topic, and the promised UI ships.

### Fixed
- **State duplication (critical).** `results` used `operator.add`, so the
  critic's score write-back appended a *copy* of every finding — the reasoning
  trace showed each task twice. Introduced a `merge_results` reducer keyed by a
  stable finding `id`; writes now update in place. (`fix(state)`)
- **Search-failure crash.** `web_search` returned a `dict` on failure while the
  executor concatenated it as a string (`str + dict` → `TypeError`). Search now
  always returns `str`. (`fix(tools)`)
- **Malformed report tables.** `align_markdown_table` re-emitted the source
  separator row as table data (`| --- | --- |` inside the body). It now skips
  separator rows. (`fix(report)`)

### Removed
- **Hardcoded knowledge base.** The pre-written `KNOWLEDGE_BASE` / `HARDCODED_KB`
  facts for "langchain"/"langgraph" (which staged the demo) are gone. Every
  finding now comes from live web search. (`fix(executor)`)

### Added
- **Structured multi-dimension critic.** Replaced the single-integer regex score
  with a Pydantic `CriticScore` (relevance, specificity, source quality,
  completeness, overall, reasoning) via `with_structured_output`. (`feat(critic)`)
- **Targeted replanning.** Low-scoring subtasks are rewritten and re-executed
  individually instead of discarding the whole plan. (`feat(critic)`)
- **Memory dedup + full retrieval.** Findings are deduplicated by content hash
  before embedding; retrieval no longer truncates to 150 chars. Memory stats are
  reported. (`feat(memory)`)
- **Integration tests.** Full-graph tests assert no duplicate findings, correct
  task count, no crash on search failure, and well-formed tables. (`test(integration)`)
- **Streamlit UI** with a live cognitive-loop view. (`feat(ui)`)
- **Project hygiene:** `pyproject.toml`, GitHub Actions CI (pytest + ruff),
  centralized `config.py` with logging (replacing `print`), and a single
  `load_dotenv` at startup. (`chore`)

## [0.1.0] — 2025

Initial release: LangGraph research agent with planner, executor, regex critic,
ChromaDB memory, and markdown report generation.
