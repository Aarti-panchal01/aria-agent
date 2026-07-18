# Examples

Real ARIA runs on diverse topics — proof that the agent generalizes now that
the hardcoded knowledge base is gone.

## How to generate

These artifacts are produced from **live** Groq + Tavily calls, so they are not
committed pre-baked (that would be the exact staged-demo problem v0.2 removed).
Generate them yourself with your own free API keys:

```bash
cp .env.example .env   # then fill in GROQ_API_KEY and TAVILY_API_KEY
python scripts/run_examples.py
```

This runs ARIA on three deliberately unrelated topics and writes:

```
examples/
├── compare-redis-vs-memcached-for-caching/
│   ├── report.md
│   └── reasoning_trace.json
├── what-is-retrieval-augmented-generation/
│   ├── report.md
│   └── reasoning_trace.json
├── how-does-transformer-attention-work/
│   ├── report.md
│   └── reasoning_trace.json
└── RESULTS.md          # metrics table across all three runs
```

`RESULTS.md` reports, per topic: subtask count, average critic score, replan
cycles, and memories used — the numbers to cite in the README's results section.

## About the two committed samples

`compare-redis-vs-memcached-for-caching/` and
`what-is-retrieval-augmented-generation/` are **real** outputs from a live run
(no hardcoded knowledge — every finding came from live search, and all six
subtasks are present with unique ids, demonstrating generalization).

Honest caveat: that run hit Groq's free-tier rate limits hard, so most of the
**critic scores in these two traces are fallback values (5/10)** — the critic's
structured-output calls were throttled (HTTP 400/429) and fell back to a neutral
score (`"reasoning": "Critic LLM unavailable..."`). The reports themselves are
genuine; only the per-finding critic scores are degraded. Re-run
`python scripts/run_examples.py` with quota headroom to regenerate these with
real critic scores and to produce the third topic + `RESULTS.md`.
