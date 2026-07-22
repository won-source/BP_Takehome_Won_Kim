# Lake County Contract Extraction & Analysis Pipeline

## Architecture

349 of 389 corpus documents entered the pipeline (40 excluded — wrong document type, not size — via filename-regex classification failure or correct routing to solicitation/bid categories out of scope). Tiered extraction: 123 documents (Tier 1/1.5) received full 19-field LLM extraction; 226 documents (Tier 2/4) received existence-only identity confirmation, zero API calls. Structured output lives in SQLite (`contracts.db`, browsable with standard SQLite tooling — e.g. the `sqlite3` CLI or Datasette, though neither is a bundled dependency of this project); a ChromaDB vector store (1,457 chunks, built from the 123 fully-extracted documents) supports clause-level retrieval. A Streamlit chat UI routes queries across SQL, vector, or a rejection path for out-of-scope questions.

## Key decisions

* 19 fields were chosen to answer specific portfolio questions (assignment exposure, renewal risk, spend commitment), not to maximize extraction coverage.
* Document classification and renewal-confirmation both use filename-pattern matching rather than content inspection — zero-cost, deterministic, and documented as a scoping choice rather than a silent limitation.
* 4 fields (liability_cap, governing_law, initial_term, out_of_scope_defined) were removed from the schema after verifying zero variance or redundancy across the labeled set, not by assumption.
* Oversized documents (10 scanned PDFs >50 pages, 6 digital PDFs >60 pages, up to 405 pages) were manually triaged out of Tier 1/1.5 into Tier 2 (existence-only) before extraction ran, using a page-count/text-extractability CSV (`triage_corpus.py`) reviewed by hand — not an automated runtime cutoff. See Known limitations for what this means in practice.

## Evaluation

Extraction accuracy: 81.9% automated / 85.8% manual-verified across 127 scored fields, drawn from the 12 Tier 1/1.5 documents (of 17 hand-labeled total — the other 5 are Tier 2/4 existence-only docs, which get an identity check, not field-by-field scoring). Retrieval eval: 10/10 passed on a hand-designed test suite (5 SQL-path, 5 vector-path), including a documented safe-failure case (CDW modification query — retrieval gap, correctly hedged rather than hallucinated).

## Known limitations

* DocuSign AcroForm-based documents (4 in corpus) have form-field text invisible to plain-text extraction.
* `sole_source_vs_competitive_bid` (60% accuracy, 3/5 scored) has two root-caused failure modes, not one systemic bug: one genuine extraction miss (`141301_CDW_Signed.pdf` — page 1 explicitly states the SOW "shall be governed by the OMNIA Cobb County... Agreement," a clear cooperative-purchasing-vehicle signal the pipeline read past and returned `not_stated` for) and one defensible GT/pipeline disagreement (`2025_12_16_Contract_25306...pdf` — ground truth inferred `competitive_bid` from "Consultant submitted a proposal," but the document never uses the words RFP, competitive bid, or solicitation; the pipeline's conservative not_stated is arguably correct given no explicit procurement-method statement exists in the text).
* Oversized-document exclusion (see Architecture) was a one-time manual CSV triage before extraction, not a runtime, per-document `extraction_status` flag — the `extraction_status` column exists in the schema but is only ever populated with `error` (file not found or an uncaught exception during extraction), never a distinct oversized-skip value. A document that slipped past triage would be extracted like any other, with vision-page capture silently capped at 35 pages (`page_truncated` flag) rather than skipped.
* Vector store is not independently browsable — accessed only via the chat UI's retrieval path, with citations.
* `computed_status` and the renewal-precise columns are a snapshot as of the last `build_master_table.py` run, not live-computed — while the chat UI's "expiring soon" query does its own date filtering live (`date('now')`), the underlying active/expired classification itself is frozen at build time. Four of the six precise renewal-expiration dates land in October 2026; without a rebuild after that point, those contracts will keep reporting as active past their real deadline. Rebuild (`py build_master_table.py --db contracts.db`) to refresh.
* Duration-derived expiration dates (e.g. "12 weeks from...") are always anchored to the contract's effective date, even when the source text names a different anchor event. One case in the corpus (`21144_Fully_Executed_Agreement.pdf`, "12 weeks from receipt of notice to proceed") is anchored to the signature date instead of the actual notice-to-proceed date — no impact today since the contract is expired either way, but it's the general failure mode for any duration tied to a milestone other than execution.

## Top improvements, if continued

1. Letter-text regex extraction for renewal dates (currently derived from filename patterns; day-level precision would improve with direct text parsing).
2. Fix the `141301_CDW_Signed.pdf`-class extraction miss for `sole_source_vs_competitive_bid` (explicit procurement-vehicle language present in the text but not picked up) — the remaining gap in that field is a labeling-inference disagreement on one document, not a missing capability.
3. Renewal-reminder email automation (90/60/30-day thresholds) — logic designed, not wired to a live email service in this environment.
