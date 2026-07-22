# Lake County Contract Extraction & Analysis Pipeline

## Architecture

349 of 389 corpus documents entered the pipeline. 40 were excluded for document type reasons: filename-regex classification failures, or correct routing to out-of-scope solicitation/bid categories. Tiered extraction: 123 documents (Tier 1/1.5) received full 19-field LLM extraction; 226 documents (Tier 2/4) received existence-only identity confirmation, zero API calls. Structured output lives in SQLite (`contracts.db`, included in this repository; open directly with the `sqlite3` CLI, DB Browser for SQLite, or Datasette). A ChromaDB vector store (1,457 chunks, built from the 123 fully-extracted documents) supports clause-level retrieval. A Streamlit chat UI routes queries across SQL, vector, or a rejection path for out-of-scope questions.

## Setup and Running

**Install dependencies:**
```
pip install -r requirements.txt
```

**Create a `.env` file** in the project root with your own API keys:
```
ANTHROPIC_API_KEY=your_key_here
VOYAGE_API_KEY=your_key_here
```

**Run the chat UI:**
```
py -m streamlit run chat_app.py
```
On Windows, bare `python`/`streamlit` may resolve to a stub rather than the real install. Use `py -m streamlit` if `streamlit run` alone doesn't launch.

**Open the database directly:** `contracts.db` is included in this repository. Open with the `sqlite3` CLI, DB Browser for SQLite, or Datasette. No setup beyond the file itself.

**Run the evaluation scripts:**
```
py compare_ground_truth.py
py eval_retrieval.py
```

**Optional: rebuild derived tables.** Only needed for full production use.
```
py build_master_table.py --db contracts.db
```

## Key decisions

* The 19 fields were chosen specifically to answer portfolio questions like assignment exposure, renewal risk, and spend commitment.
* Document classification and renewal confirmation both use filename-pattern matching. This is zero-cost and deterministic; it's documented here as a deliberate scoping choice.
* 4 fields (liability_cap, governing_law, initial_term, out_of_scope_defined) were removed from the schema after verifying zero variance or redundancy across the labeled set.
* Oversized documents (10 scanned PDFs >50 pages, 6 digital PDFs >60 pages, up to 405 pages) were manually triaged out of Tier 1/1.5 into Tier 2 (existence-only) before extraction ran, using a page-count/text-extractability CSV (`triage_corpus.py`) reviewed by hand as a one-time step. See Known limitations for what this means in practice.

## Evaluation

Extraction accuracy: 81.9% automated / 85.8% manual-verified across 127 scored fields, drawn from the 12 Tier 1/1.5 documents (of 17 hand-labeled total: 4 Tier 2/4 existence-only docs plus 1 rate_adjustment doc scored on a separate specialized schema). Retrieval eval: 10/10 passed on a hand-designed test suite (5 SQL-path, 5 vector-path), including one documented safe-failure case: a CDW modification query hit a retrieval gap, and the model correctly said so instead of guessing.

## Known limitations

* DocuSign AcroForm-based documents (4 in corpus) have form-field text invisible to plain-text extraction.
* `sole_source_vs_competitive_bid` (60% accuracy, 3/5 scored) has two separate, root-caused failure modes. One is a genuine extraction miss (`141301_CDW_Signed.pdf`: page 1 explicitly states the SOW "shall be governed by the OMNIA Cobb County... Agreement," a clear cooperative-purchasing-vehicle signal the pipeline read past and returned `not_stated` for). The other is a defensible GT/pipeline disagreement (`2025_12_16_Contract_25306...pdf`: ground truth inferred `competitive_bid` from "Consultant submitted a proposal," but the document never uses the words RFP, competitive bid, or solicitation; the pipeline's conservative `not_stated` is arguably correct given no explicit procurement-method statement exists in the text).
* Oversized-document exclusion (see Architecture) happened as a one-time manual CSV triage before extraction. The `extraction_status` column exists in the schema, but it's only ever populated with `error` (file not found or an uncaught exception during extraction); it never gets a distinct oversized-skip value. A document that slipped past triage would be extracted like any other, with vision-page capture capped at 35 pages (`page_truncated` flag).
* Vector store isn't independently browsable. It's only accessed through the chat UI's retrieval path, with citations.
* `computed_status` and the renewal-precise columns are a snapshot from the last `build_master_table.py` run. The chat UI's "expiring soon" query does its own date filtering live (`date('now')`), but the underlying active/expired classification is frozen at build time. Four of the six precise renewal-expiration dates land in October 2026. Without a rebuild after that point, those contracts will keep reporting as active past their real deadline. Rebuild with `py build_master_table.py --db contracts.db` to refresh.
* Duration-derived expiration dates (e.g. "12 weeks from...") are always anchored to the contract's effective date, even when the source text names a different anchor event. One case in the corpus (`21144_Fully_Executed_Agreement.pdf`, "12 weeks from receipt of notice to proceed") is anchored to the signature date instead of the actual notice-to-proceed date. This has no impact today since the contract is expired either way, but it's the general failure mode for any duration tied to a milestone other than execution.
* `renewal_precise_expiration` is inaccurate by 1-17 days for 4 of the 6 contracts currently computing as still-active via renewal (Burke, Stanley, Ciorba, Clark Dietz, Burns & McDonnell; Burke's is correct by coincidence). Root cause: the column assumes each sub-agreement renews on its own originally-extracted base-agreement date. Lake County actually renews all sub-agreements under one `contract_id` on a shared anniversary schedule. This doesn't affect any active/inactive determination, only exact-date precision if queried directly. It isn't fixed; correct dates were applied manually to the exec summary slide instead.

## Top improvements, if continued

**Features:**

* Normalize `contract_value` into a structured annual/period figure. It's currently free-text; this is needed for a live dashboard and any future ERP cross-reference.
* Distinguish signature/execution date from contractual effective date, especially on modifications.
* Renewal-reminder email automation (90/60/30-day thresholds), with logic designed but no live email integration in this environment.
* Letter-text date extraction for all files, replacing filename-derived renewal dates with direct text parsing for day-level precision (generalizes the fix for the `renewal_precise_expiration` limitation above).
* Teams/workplace chatbot integration, scoped out for time.

**Evaluation:**

* Close remaining accuracy gaps to push past 85%, including the DocuSign AcroForm issue and any other root-caused gaps surfaced during a broader field-by-field review.
* Automate the eval suite as a pre-deploy check, so any change to extraction logic reruns `compare_ground_truth.py` and `eval_retrieval.py` before shipping.
* Expand the retrieval eval beyond the current 10 hand-designed cases toward a larger, less curated sample.

**Production monitoring:**

* Schedule periodic ground-truth re-runs and spot-audits.
* Monitor existing confidence signals over time, with active alerting on distance-threshold pass-rate and truncation/error-rate trends.
* Add a live dashboard of current contracts.
