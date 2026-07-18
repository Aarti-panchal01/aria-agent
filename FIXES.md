# FIXES — output-quality & UI pass

Fixes to ARIA's research quality and interface, building on v0.3.0.

## 1. Planner — no more unavailable tools (`fix(planner)`)
- The planner now receives the list of sources ARIA can actually query
  (web/arXiv/Wikipedia/GitHub) and is instructed to plan only tasks answerable
  from them — with an explicit ban on JSTOR, EBSCO, ProQuest, Scopus, paywalled
  tools, surveys, interviews, and lab work.
- Added a **`max_replans` state field (default 2)**; the router caps targeted
  replans at that number, then accepts the best result and moves on. No infinite
  loops. (`nodes/planner.py`, `state.py`, `main.py`, `graph.py`)

## 2. Report — substantial, multi-section (`feat(report)`)
- Rewrote the report prompt to produce: **Executive Summary** (2-3 paragraphs),
  **Background**, **Key Findings** (numbered, each 2-3 sentences ending with a
  `[Web]/[arXiv]/[Wikipedia]/[GitHub]` source tag), **Analysis**, **Comparison
  Table**, and **Conclusion**.
- Target length ≥ 600 words; if the draft is **< 300 words it regenerates once**
  with an explicit "expand with more detail and analysis" instruction.
- A **Sources** section is appended deterministically from the URLs actually
  retrieved (accurate, never hallucinated).
- `max_tokens` raised 800 → 2048 so long reports aren't truncated.
  (`nodes/report_generator.py`)

## 3. Memory — relevance filtering (`fix(memory)`)
- Retrieval now applies a **cosine-similarity floor of 0.75**
  (`retrieve_candidates` in `memory/chroma_store.py`).
- Survivors then pass an **LLM Yes/No relevance check** against the current
  goal; only genuinely relevant memories are injected.
- If nothing qualifies, the context is **"No relevant memories from previous
  runs."** instead of forcing unrelated context. (`nodes/memory_reader.py`)
- This stops a memory about, say, "CGP after NEET 2026" leaking into a run about
  "Cockroach Janta Party."

## 4. UI — compact layout & styling (`feat(ui)`)
- **Top/bottom layout** replaces the cramped side-by-side columns: cognitive
  loop on top, full-width report below.
- **Live feed**: shows the **last 5 steps** by default with small muted
  timestamps; a **"Show all steps"** expander groups the full history **by task**
  with each task's final critic score.
- **Progress bar**: custom 20px-thick bar with the label **inside** it
  ("4 of 6 tasks complete") — indigo while running, green when complete.
- **Report styling** (CSS): amber `h2` headers with an indigo underline,
  bordered tables with alternating row shading, roomier bullets; a
  **word count** ("Report: ~847 words") under the report.
- **Spacing**: 4px between feed steps, reduced top padding, tighter session
  header.
- **Sidebar**: 12px session-list buttons; a prominent full-width primary
  **🚀 Run ARIA** button.
- **PDF export**: replaced the box-glyph title with plain **"ARIA Research
  Report"**; strengthened table borders (visible grid + indigo box/header rule);
  the report's **Sources** section now flows into the PDF.
  (`ui/app.py`, `report/pdf_exporter.py`)

## Verification
- `pytest -v` — **32 passing** (added memory-relevance tests; updated the
  integration test to the new memory + source seams).
- `ruff check .` — clean.
- `streamlit run ui/app.py` — boots without error.
- CI green on Python 3.10 and 3.12.

## Note on process
These phases were built **sequentially**, not in parallel: they share files
(`state.py`, `main.py`, `ui/app.py`, node modules) and Phase 5 verifies the
output of Phases 1-4, so parallel agents editing the same files would have raced
and produced a broken push.
