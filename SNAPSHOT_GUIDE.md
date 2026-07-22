# vFinal — code + data snapshot

This folder is a curated snapshot of the core, current files from the Lake County contract
pipeline project, prepared for submission/review. It includes the runtime data needed to
actually execute the tools (`contracts.db`, `chroma_db/`, the current ground-truth
spreadsheet), and deliberately excludes old file versions, one-off triage/scratch scripts,
intermediate CSV artifacts, the raw PDF corpus, and `.env` secrets — see "What's
deliberately not here" below.

This file is about the snapshot itself. `README.md` in this same folder is the project's
own README (product architecture, decisions, eval numbers, limitations) — read that one
for the product, this one for how the files here fit together.

## How the pieces connect

```
extraction_schema.json ─┐
extraction_system_prompt.txt ─┤
                              ▼
                    extract_contracts.py  ──┐
                    build_existence_table.py │  write to contracts.db
                    build_vector_store.py    │  (SQLite) + chroma_db/
                              │              │
                              ▼              │
                    date_utils.py ◄──────────┤ (shared date parsing,
                    renewal_confidence.py ◄──┘  imported by build_master_table.py
                              │                  and compare_ground_truth.py)
                              ▼
                    build_master_table.py ──► contracts_master table
                              │
                              ▼
                    query_router.py ──► chat_app.py (Streamlit UI)

compare_ground_truth.py ──► reads contracts.db + Berkshire_Ground_Truth_Labels_v14.xlsx (both included)
eval_retrieval.py       ──► calls query_router.answer_question() directly, no UI needed
```

**This snapshot is runnable, not just readable** — `contracts.db`, `chroma_db/`, and
`Berkshire_Ground_Truth_Labels_v14.xlsx` are all included (see "Data files included"
below). The one thing still required to actually execute anything is a `.env` with
`ANTHROPIC_API_KEY`/`VOYAGE_API_KEY` — not included here (secrets), so whoever runs this
needs to supply their own.

## File-by-file

### Extraction pipeline
- **`extract_contracts.py`** — Tier 1/1.5 extraction (base agreements, SOWs, modifications,
  amendments). Reads each PDF (digital-text via pypdf, or rasterized pages via PyMuPDF for
  scanned docs), sends it to Claude with `extraction_schema.json`'s tool-use schema and
  `extraction_system_prompt.txt`'s instructions, writes 19 fields (`{field}_value`/
  `{field}_page` pairs) per document into the `extractions` table.
- **`extraction_schema.json`** — the Claude tool-use schema defining the 19 extracted
  fields and their enums. Consumed by `extract_contracts.py`.
- **`extraction_system_prompt.txt`** — the 13-numbered-rule system prompt steering
  extraction behavior (citation requirements, date-resolution rules, enum tie-breakers).
  Consumed by `extract_contracts.py`. CLAUDE.md flags rule 2 (citation-mandatory) as having
  been accidentally weakened by external prompt updates more than once — diff before
  overwriting this file.
- **`build_existence_table.py`** — Tier 2/4 existence-only pass (renewal letters,
  extensions, leases, vendor disclosures, award notices, rate adjustments, and oversized
  Tier 1/1.5 docs excluded for cost/effort). No extraction, no API calls — just identity
  fields (filename, contract_id, doc_role, tier) into the `existence_only` table.
- **`date_utils.py`** — shared date/duration parsing (`normalize_date`, `parse_duration`,
  `parse_notice_period_days`, `parse_escalation_pct`). Single source of truth imported by
  both `build_master_table.py` (computed status columns) and `compare_ground_truth.py`
  (scoring), so the two never drift out of sync on how a date string is interpreted.
- **`renewal_confidence.py`** — filename-only heuristic (`classify_renewal_filename`,
  `best_tier_per_contract`, `precise_renewal_expiration`) that tiers renewal_letter/
  extension filenames into high/moderate/low confidence and computes a precise implied
  renewal-expiration date for high-confidence matches. No document-content extraction, no
  API calls. Imported by `build_master_table.py`.
- **`build_master_table.py`** — rebuilds the unified `contracts_master` table from
  `extractions` + `existence_only` (+ the legacy `rate_adjustments` table), then layers on
  computed columns: `parsed_effective_date`, `parsed_expiration_date`, `computed_status`,
  `renewal_action_needed`, `renewal_confidence`, `renewal_precise_expiration`,
  `renewal_precise_still_active`. This is the table every other file in this folder reads
  from — nothing downstream touches `extractions`/`existence_only` directly.
- **`build_vector_store.py`** — builds the ChromaDB clause store from the 123 Tier 1/1.5
  docs: section-based chunking (falls back to page-based), a one-time vision transcription
  pass for scanned docs, resumable (`--resume`) and crash-resilient (per-document
  incremental writes, retry with backoff on transient network errors).

### Evaluation
- **`compare_ground_truth.py`** — field-by-field scoring of `contracts_master` against
  `Berkshire_Ground_Truth_Labels_v14.xlsx` (the `GT_FILE` constant points at it; both are
  included in this snapshot, so this file runs as-is with a valid `.env`). Splits results
  into a **value test** (GT has a real answer — the harder,
  more informative accuracy number) and a **recognition test** (GT explicitly says
  not_stated — did the pipeline correctly recognize absence, or hallucinate a value).
  Flags fields where the recognition-test rows dominate the total ("base-rate driven") so a
  high blended number can't be mistaken for genuine extraction skill. Uses `date_utils.py`
  for date-equivalence matching and a dedicated dollar-amount extractor (matches
  `$`-prefixed figures on the raw, pre-normalization string, so a real value isn't lost to
  a blanket "starts with not_stated" normalization rule).
- **`eval_retrieval.py`** — a 10-case eval (5 SQL-path, 5 vector-path) against the live
  `query_router.answer_question()` pipeline, each with an explicit programmatic pass
  criterion (not a subjective read). Includes a deliberate "fails safely" case (a
  known-weak retrieval query that should hedge rather than hallucinate) and a router
  rejection case (gibberish input should decline to call a tool, not guess).

### Query & chat
- **`query_router.py`** — the core query logic. One Claude tool-use call decides SQL-path
  vs. vector-path and produces the query in the same call; a second call synthesizes the
  final answer with citations. Contains `INTENT_NORMALIZATION` (maps common vague
  phrasings to verified-correct SQL), the vendor-hint metadata pre-filter for vector search
  (`_filenames_for_vendor_hint`, `run_vector_path`), SQL safety validation
  (`_validate_readonly_sql`), a bounded single-retry on SQL execution error, and a
  top-level `try/except` in `answer_question()` so no exception ever reaches the caller.
  Both `chat_app.py` and `eval_retrieval.py` call into this file; it has no UI dependency
  of its own and runs standalone via its CLI entry point.
- **`chat_app.py`** — the Streamlit chat UI. Calls `query_router.answer_question()`,
  renders SQL results as Plotly bar charts or tables (`render_bar_chart`,
  `render_extras`) rather than raw LLM prose, and renders vector results with inline
  citations. Sidebar has two button groups: structured "Try asking:" queries and a
  separate "You can also retrieve contract clauses:" group for vector-path queries.

### Documentation
- **`CLAUDE.md`** — the project's full session-handoff document: current state, complete
  architecture, every fix made and why, known open issues, file inventory. The single most
  detailed source of *why* things look the way they do in this codebase — read it before
  the code if you only have time for one.
- **`README.md`** — the project-facing README (architecture, key decisions, evaluation
  numbers, known limitations, top improvements if continued). Shorter and more slide-ready
  than CLAUDE.md.

## Data files included

- **`contracts.db`** (444K) — SQLite, all tables (`extractions`, `existence_only`,
  `contracts_master`, `rate_adjustments`). Needed by `query_router.py`, `chat_app.py`,
  `compare_ground_truth.py`, and `eval_retrieval.py` — essentially everything except the
  document-classification/existence scripts when run from scratch.
- **`chroma_db/`** (31M) — the ChromaDB persistent store, collection `contract_clauses`,
  1,457 chunks. Needed for any vector-path query (`query_router.py`'s vector path,
  `chat_app.py`'s clause-retrieval sidebar buttons, `eval_retrieval.py`'s vector-path
  cases).
- **`Berkshire_Ground_Truth_Labels_v14.xlsx`** (24K) — the current hand-labeled ground
  truth, matching the `GT_FILE` constant in `compare_ground_truth.py`.

All three are small enough to include without meaningfully changing the size of this
folder (~32M total) — leaving them out would mean nothing here actually runs.

## What's deliberately not here

- **The raw PDF corpus (`contracts/`, ~549M, 389 documents)** — not needed to run any tool
  in this folder; only matters for rebuilding extraction/the vector store from scratch
  (`extract_contracts.py`, `build_vector_store.py`) or manually spot-checking a specific
  PDF against its extracted fields. Left out for size; available in the main project if
  needed.
- **Secrets:** `.env` (API keys) — required to actually execute anything here, but never
  bundled. Supply your own `ANTHROPIC_API_KEY`/`VOYAGE_API_KEY`.
- **Superseded/historical:** older ground-truth spreadsheet versions
  (`Berkshire_Ground_Truth_Labels_v2/v4/v5/v7/v9/v10/v11/v12/v13.xlsx` and the unversioned
  original), `contracts.db.v10_backup`, `RETIRED_*` files (the descoped Tier 3
  rate-adjustment pipeline, preserved in the main project but not current), and
  `STATUS_REPORT.md` (an earlier session's handoff doc, explicitly marked historical in
  CLAUDE.md).
- **One-off triage/scratch:** `triage_corpus.py`, `family_grouping.py`,
  `test_vector_retrieval.py` (a lighter, informal precursor to `eval_retrieval.py`), and
  every intermediate CSV (`master_tiered_corpus.csv`, `tier*_existence_only.csv`,
  `pilot_tier*.csv`, `corpus_triage.csv`, `family_*.csv`, `suggested_sample_100*.csv`,
  `vision_test_sample.csv`, `vision_test.db`).
- **`tiered_extraction_approach.md`** — still current in the main project (CLAUDE.md cites
  it as the authoritative source for the Tier 3 and oversized-document descoping
  rationale), but left out of this code-focused snapshot since it's rationale
  documentation rather than code. Worth pulling in separately if the reviewer wants the
  full reasoning behind the tiering decisions.
