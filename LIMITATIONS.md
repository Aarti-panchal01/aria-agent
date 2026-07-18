# Limitations

An honest account of what ARIA does *not* do well yet. Naming these is
deliberate — trust a system that knows its own edges.

## Research quality is bounded by web search
ARIA synthesizes only what Tavily returns. For niche, paywalled, or very recent
topics, results can be thin. It reads search-result snippets (top 3, ~600 chars
each), not full articles — it does not crawl or read PDFs.

## The critic is an LLM, not ground truth
Scores come from `llama-3.1-8b-instant` judging text. They are calibrated and
multi-dimensional, but a small model can still be over-generous or miss subtle
inaccuracy. The critic checks *quality signals*, not factual correctness against
an authority.

## Replanning is bounded and greedy
Targeted replanning retries a weak subtask up to `MAX_REPLANS` (3) times total
across the run. If one subtask keeps failing, it consumes the budget and ARIA
moves on with the best result it has rather than looping forever.

## Fixed plan width
The planner always produces ~6 subtasks regardless of topic breadth. Very narrow
or very broad goals are not adaptively resized.

## Memory is single-user and local
ChromaDB is stored on disk with its default embedding function. There is no
multi-user isolation, no re-ranking, and no eviction — memory grows unbounded
across runs (deduplicated by exact content hash only).

## Single provider / single model
Everything runs on Groq + Llama 3.1 8B. There is no model routing, no fallback
provider, and no cost/latency budgeting.

## Not hardened for production
No rate-limit backoff beyond a fixed retry, no persistent run history/DB, no
auth on the UI, and no observability beyond logs and the JSON reasoning trace.
