# ARIA v0.3.0 — Upgrade Notes

This upgrade turns ARIA from a CLI research agent into a deployable, multi-source
research product with a live UI, persistent sessions, and PDF export.

## What was added

1. **Streamlit Cloud deploy config** — `.streamlit/config.toml` (indigo/dark
   theme), `.streamlit/secrets.toml.example`, and a README deploy guide.
2. **Real-time streaming UI** (`ui/app.py`) — the cognitive loop streams live:
   a left feed of timestamped step cards (plan → execute → critic scores →
   replan → memory → done), a progress bar (`tasks_completed / total`), and the
   report on the right. Uses `aria_graph.stream()`.
3. **Persistent sessions** (`sessions/manager.py`) — named research sessions in
   SQLite (`.aria_sessions/sessions.db`) with JSON state snapshots. Resume any
   past session from the sidebar.
4. **PDF export** (`report/pdf_exporter.py`) — one-click, themed PDF of any
   report (markdown → headings/tables/bullets, metadata block, per-page footer).
5. **Multi-source research** (`tools/sources/`) — web (Tavily), **arXiv**,
   **Wikipedia**, and **GitHub**, queried in parallel, deduplicated by URL,
   ranked by relevance, with `[SOURCE]` attribution in every finding. Pick
   sources via sidebar checkboxes.

Tests grew from 17 → 29 (sessions, PDF, per-source, aggregator, updated
integration). `ruff` clean; CI runs on 3.10 and 3.12.

## How to deploy to Streamlit Cloud (step by step)

1. Ensure this repo is on GitHub (branch `main`).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with
   GitHub.
3. **New app** → pick this repo, branch `main`, **main file path `ui/app.py`**.
4. **Advanced settings → Secrets** → paste (see
   `.streamlit/secrets.toml.example`):
   ```toml
   GROQ_API_KEY = "..."
   TAVILY_API_KEY = "..."
   # GITHUB_TOKEN = "..."   # optional
   ```
5. **Deploy**. The theme and `requirements.txt` are picked up automatically.

## How to use each new source

Sources are chosen with the sidebar **Research Sources** checkboxes; the CLI
uses all available sources by default.

| Source | Key needed | Best for |
| --- | --- | --- |
| 🌐 Web (Tavily) | `TAVILY_API_KEY` | Current events, blogs, docs |
| 🔬 arXiv | none | Academic papers, ML/science topics |
| 📖 Wikipedia | none | Definitions, background, overviews |
| 🐙 GitHub | none (`GITHUB_TOKEN` raises rate limit) | Libraries, implementations, tools |

- A source with a missing key is shown greyed-out and skipped automatically.
- Every finding is tagged with its origin, e.g. `[ARXIV]`, `[WIKIPEDIA]`.
- Sources are queried **in parallel** (thread pool) per subtask, then merged and
  deduplicated by URL.

## What Aarti must do manually

These require your accounts / a display and can't be done from CI:

- [ ] **Deploy**: go to [share.streamlit.io](https://share.streamlit.io) and
      deploy the app (main file `ui/app.py`).
- [ ] **Add secrets** on Streamlit Cloud: `GROQ_API_KEY`, `TAVILY_API_KEY`.
- [ ] **Optional**: add `GITHUB_TOKEN` (Cloud secrets or local `.env`) to raise
      the GitHub source's rate limit.
- [ ] **Fill the results table**: run `python scripts/run_examples.py` with your
      keys to generate `examples/` + `examples/RESULTS.md`, then paste the
      numbers into the README results table. (Note: Groq's free tier rate-limits
      hard — expect ~5 min per topic; run when you have quota headroom.)
- [ ] **Record a demo GIF** of a live run and save it to `assets/demo.gif`, then
      reference it in the README.

## Notes / honest limitations

- Multi-source parallelism uses a **thread pool**, not `asyncio` — the source
  libraries (arxiv/wikipedia/PyGithub) are synchronous, so threads give real
  parallelism where asyncio would not.
- ChromaDB memory remains local/embedded (see `SECURITY.md`); sessions store
  state snapshots as JSON, not full LangGraph checkpoints, so "resume" continues
  the same goal with accumulated memory rather than replaying mid-graph.
- See `LIMITATIONS.md` for the full list.
