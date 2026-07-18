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
