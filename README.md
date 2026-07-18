# ARIA — Autonomous Research Intelligence Agent

> An LLM agent that **plans, executes, critiques, and replans its own research** —
> then writes a sourced report.

![CI](https://github.com/Aarti-panchal01/aria-agent/actions/workflows/ci.yml/badge.svg)
&nbsp;·&nbsp; Python 3.10+ &nbsp;·&nbsp; LangGraph · ChromaDB · Tavily · Groq &nbsp;·&nbsp; MIT

---

## What makes ARIA different

Most "agents" plan once and execute in a straight line. ARIA closes the loop:

- It **scores every finding across four dimensions** — relevance, specificity,
  source quality, and completeness — using structured output, not a regex.
- It **replans surgically**: when a subtask's finding is weak, ARIA rewrites and
  re-runs *only that subtask* with a targeted instruction, instead of throwing
  away the whole plan.
- It **remembers**: findings are embedded in ChromaDB (deduplicated by content
  hash) and retrieved on future runs to avoid redundant work.

The result is a research loop whose quality is *self-corrected*, with a complete
JSON reasoning trace for every decision.

> **No staged demos.** Earlier versions injected hardcoded facts for a single
> topic. That is gone — every finding now comes from live web search, and ARIA
> is tested to generalize (see `examples/`).

---

## How it works

```
memory_reader → planner → executor → critic → memory_writer → terminator ─┬─▶ report_generator → END
                             ▲                                            │
                             └──────────── replanner ◀────────────────────┘
                                    (only when the critic flags a weak finding)
```

1. **Memory Reader** — pulls relevant past findings from ChromaDB.
2. **Planner** — breaks the goal into distinct subtasks.
3. **Executor** — generates a focused query and runs a live Tavily search.
4. **Critic** — scores the finding (structured `CriticScore`) and decides if a
   replan is needed.
5. **Replanner** — rewrites just the failing subtask and re-executes it.
6. **Memory Writer** — persists the finding (content-hash dedup).
7. **Report Generator** — synthesizes a markdown report + reasoning trace.

State flows through a typed `AgentState` whose `results` channel uses a custom
`merge_results` reducer (keyed by a stable finding `id`) so scores update
findings in place — no duplicates.

---

## Results

Run ARIA on three deliberately unrelated topics to demonstrate generalization:

```bash
python scripts/run_examples.py   # requires your GROQ + TAVILY keys
```

This populates `examples/` with a report + reasoning trace per topic and an
`examples/RESULTS.md` table:

| Topic | Subtasks | Avg critic score | Replan cycles | Memories used |
| --- | --- | --- | --- | --- |
| Compare Redis vs Memcached | _run to fill_ | _run to fill_ | _run to fill_ | _run to fill_ |
| What is retrieval augmented generation | _run to fill_ | _run to fill_ | _run to fill_ | _run to fill_ |
| How does transformer attention work | _run to fill_ | _run to fill_ | _run to fill_ | _run to fill_ |

> These cells are intentionally left for you to fill from a real run — the
> numbers should be *measured*, not asserted. `scripts/run_examples.py` writes
> them for you.

---

## Setup

```bash
git clone https://github.com/Aarti-panchal01/aria-agent
cd aria-agent

python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

pip install -e ".[dev,ui]"       # installs ARIA + tests + Streamlit UI
```

Configure API keys (both have free tiers):

```bash
cp .env.example .env
# GROQ_API_KEY   → https://console.groq.com
# TAVILY_API_KEY → https://app.tavily.com
```

---

## Run

**CLI:**

```bash
aria                 # console script (installed via pyproject)
# or: python main.py
```

**Web UI — watch the cognitive loop live:**

```bash
streamlit run ui/app.py
```

The UI streams each node as it fires ("Planned 6 subtasks" → "Critic scored
8/10 (relevance 9…)" → "Replanning weak subtask…") and renders the final report
with a copy button and a per-task reasoning trace in the sidebar.

---

## Deploy to Streamlit Cloud

ARIA ships with a ready-to-deploy Streamlit app.

1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. Go to [share.streamlit.io](https://share.streamlit.io) and **connect your
   GitHub account**.
3. Create a new app pointing at this repo, branch `main`, **main file
   `ui/app.py`**.
4. Under **App → Settings → Secrets**, add your keys (see
   [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example)):
   ```toml
   GROQ_API_KEY = "..."
   TAVILY_API_KEY = "..."
   # GITHUB_TOKEN = "..."   # optional, raises the GitHub source's rate limit
   ```
5. Deploy. The theme in [`.streamlit/config.toml`](.streamlit/config.toml) is
   applied automatically.

Streamlit Cloud injects those secrets into the environment, and ARIA reads its
keys from `os.environ`, so no code changes are needed.

---

## Tech stack

| Layer | Technology |
| --- | --- |
| LLM | Groq + Llama 3.1 8B Instant |
| Agent framework | LangGraph (`StateGraph`, custom reducer, conditional edges) |
| Structured output | Pydantic (`CriticScore`) |
| Web search | Tavily API |
| Memory | ChromaDB (local vector store, content-hash dedup) |
| UI | Streamlit |

---

## Project structure

```
aria-agent/
├── main.py                 # CLI entry point + run_research() helper
├── config.py               # env loading, logging, retry policy, constants
├── schemas.py              # Pydantic CriticScore
├── graph.py                # LangGraph wiring + targeted-replan routing
├── state.py                # AgentState + merge_results reducer
├── nodes/                  # planner, executor, critic, replanner, memory_*, terminator, report_generator
├── memory/chroma_store.py  # persistent vector memory
├── tools/search.py         # Tavily web search (always returns str)
├── ui/app.py               # Streamlit live cognitive-loop UI
├── scripts/run_examples.py # generate examples/ + RESULTS.md
├── tests/                  # unit + full-graph integration tests
└── pyproject.toml
```

---

## Development

```bash
pytest -v            # unit + integration tests
ruff check .         # lint
```

CI runs both on every push (see `.github/workflows/ci.yml`). The integration
tests exercise the **full compiled graph** and assert: no duplicate findings,
one finding per subtask, no crash on search failure, and well-formed tables.

See [`LIMITATIONS.md`](LIMITATIONS.md) for an honest account of what ARIA does
not do well, and [`CHANGELOG.md`](CHANGELOG.md) for the v0.1 → v0.2 upgrade.

---

## Citation

```bibtex
@software{panchal2026aria,
  author = {Panchal, Aarti},
  title  = {ARIA: Autonomous Research Intelligence Agent},
  year   = {2026},
  url    = {https://github.com/Aarti-panchal01/aria-agent}
}
```

## Author

**Aarti Panchal** — B.Tech AI/ML, PES University (2024–2028)
[Portfolio](https://aarti-panchal.site/) · [LinkedIn](https://linkedin.com/in/aarti-panchal-93196a319)

## License

MIT — see [LICENSE](LICENSE).
