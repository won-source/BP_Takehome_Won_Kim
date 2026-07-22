"""
Query router: dispatches a natural-language question to one of two paths.

- STRUCTURED/AGGREGATE questions (e.g. "which contracts have assignment consent
  requirements", "total contract value expiring this year") -> SQL against
  contracts_master. Claude writes the actual SQL query itself, given the table
  schema; the query is validated read-only before running.
- SEMANTIC/CLAUSE-LEVEL questions (e.g. "what does the indemnification clause
  say", "what changed in the CDW modification") -> vector search against the
  ChromaDB clause store built by build_vector_store.py.

One Claude tool-use call decides the path AND produces the SQL/search query in
the same call (cheap, single round-trip). A second call synthesizes the final
natural-language answer from whichever path's raw results, with citations.

Scoped to the full corpus: contracts_master has 349 rows (123 Tier 1/1.5 docs
with full 19-field extraction, 226 Tier 2/4 existence-only records). The
ChromaDB clause store covers all 123 Tier 1/1.5 docs.

Every failure point (router can't classify, SQL errors out, vector search
finds nothing relevant) returns a clean user-facing message instead of an
exception -- see the FALLBACK_* constants and answer_question()'s top-level
try/except.
"""

import argparse
import contextlib
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import anthropic
import chromadb
import voyageai

from extract_contracts import _load_dotenv

_load_dotenv()

MODEL = "claude-sonnet-4-5"
DB_FILE = "contracts.db"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "contract_clauses"

# ChromaDB's default distance space returns ~0.9-0.95 for genuinely relevant
# chunks and ~1.05-1.2+ for irrelevant/nonsense queries against this
# collection (empirically checked: "indemnification clause" -> best 0.926;
# "asdfjkl random nonsense" -> best 1.085). 1.0 sits in the gap between them.
VECTOR_DISTANCE_THRESHOLD = 1.0
MAX_SQL_RETRIES = 1  # total attempts = 1 + this

FALLBACK_ROUTER_UNCLASSIFIED = (
    "I wasn't able to determine how to answer that from the contract data. "
    "Try rephrasing, or ask about specific contract fields, vendors, or dates."
)
FALLBACK_VECTOR_NO_RESULTS = "I couldn't find relevant contract clauses for that question."
FALLBACK_SQL_ERROR = (
    "I couldn't retrieve contract data for that question -- the query didn't run successfully. "
    "Try rephrasing, or ask about specific contract fields, vendors, or dates."
)
FALLBACK_UNHANDLED = (
    "Something went wrong answering that question. Try rephrasing, or ask about "
    "specific contract fields, vendors, or dates."
)

CONTRACTS_MASTER_SCHEMA = """
Table: contracts_master (one row per document, all tiers; 349 rows total --
  123 Tier 1/1.5 rows with full field extraction, 226 Tier 2/4 existence-only rows)
  filename TEXT, contract_id TEXT, doc_role TEXT, extraction_tier TEXT,
  text_quality TEXT, source_pages TEXT, notes TEXT, modification_type TEXT,
  -- For Tier 1/1.5 rows (base_agreement, sow, modification, amendment), the following
  -- 19 fields are populated as {field}_value / {field}_page pairs. For Tier 2/4 rows
  -- (renewal_letter, lease, vendor_disclosure, award_notice, rate_adjustment -- existence
  -- only) these are all NULL.
  counterparty_name_value TEXT, counterparty_name_page INTEGER,
  org_role_value TEXT, org_role_page INTEGER,                          -- enum: buyer/seller/lessor/lessee/not_stated
  effective_date_value TEXT, effective_date_page INTEGER,               -- free text date (last signature date)
  expiration_date_value TEXT, expiration_date_page INTEGER,
  renewal_mechanism_value TEXT, renewal_mechanism_page INTEGER,         -- enum: none/option_to_renew/auto_renew_unless_cancelled/not_stated
  contract_value_value TEXT, contract_value_page INTEGER,
  fee_structure_value TEXT, fee_structure_page INTEGER,                -- enum: fixed_fee/time_and_materials/per_unit/rate_based/not_stated
  payment_terms_value TEXT, payment_terms_page INTEGER,
  annual_price_escalation_type_value TEXT, annual_price_escalation_type_page INTEGER,  -- enum: none/automatic_stepup/cap_on_requested_increase/not_stated
  annual_price_escalation_detail_value TEXT, annual_price_escalation_detail_page INTEGER,
  termination_for_convenience_value TEXT, termination_for_convenience_page INTEGER,  -- Y/N/not_stated
  notice_period_value TEXT, notice_period_page INTEGER,
  assignment_consent_required_value TEXT, assignment_consent_required_page INTEGER,  -- Y/N/not_stated
  indemnity_present_value TEXT, indemnity_present_page INTEGER,        -- Y/N/not_stated
  insurance_required_value TEXT, insurance_required_page INTEGER,      -- Y/N/not_stated
  sole_source_vs_competitive_bid_value TEXT, sole_source_vs_competitive_bid_page INTEGER,  -- enum: competitive_bid/sole_source/cooperative_purchasing_vehicle/not_applicable/not_stated
  signer_title_value TEXT, signer_title_page INTEGER,
  fiscal_year_appropriation_contingent_value TEXT, fiscal_year_appropriation_contingent_page INTEGER,  -- Y/N/not_stated
  modification_summary_value TEXT, modification_summary_page INTEGER,
  -- modification_type enum: rate_increase/rate_schedule_expansion/term_extension/scope_change/vendor_or_party_change/assignment/termination/other/not_applicable

  -- Computed columns (post-processing over the raw prose fields above, NOT re-extracted).
  -- ALWAYS use these for date/status filtering and comparison -- effective_date_value and
  -- expiration_date_value are rich prose (e.g. "one year from effective date, with option
  -- to renew for four additional one-year periods") and are NOT directly comparable/filterable
  -- as dates. Use the raw _value fields only when the user wants the actual contract language,
  -- not for WHERE clauses involving dates.
  parsed_effective_date TEXT,   -- YYYY-MM-DD or NULL if unparseable
  parsed_expiration_date TEXT,  -- YYYY-MM-DD or NULL if unparseable/not stated/only a duration with no parseable effective date to anchor it
  computed_status TEXT,         -- 'active' / 'expired' / 'expired_but_renewal_on_file' / 'not_yet_effective' / 'expiration_not_stated' / 'unknown'
      -- 'expired_but_renewal_on_file' means the base agreement's own dates say expired, but a
      -- renewal_letter/extension exists on file for the same contract_id -- continuation is
      -- likely but the actual renewed end date isn't known. Treat as a DISTINCT case from both
      -- 'active' and 'expired' -- don't silently fold it into either when answering.
      -- 'expiration_not_stated' means the effective date is known (the contract started, isn't in
      -- the future) but no expiration date could be resolved at all -- a clean, legitimate fact
      -- about the document (it may genuinely not state a fixed end date), NOT an error. Present
      -- this as its own category; don't call it "unknown" or imply something went wrong.
      -- 'unknown' means the EFFECTIVE date itself couldn't be resolved -- a real data gap where
      -- we don't know anything about this contract's timeline, distinct from the case above.
  renewal_action_needed INTEGER,      -- 1 if expiring within 90 days AND renewal_mechanism is
      -- auto_renew_unless_cancelled or option_to_renew (the "at risk of unintended auto-renewal" signal).
      -- CAVEAT: computed off parsed_expiration_date only -- for a contract that's already
      -- expired_but_renewal_on_file, this never fires even when renewal_precise_expiration
      -- shows a real deadline coming up soon, since it never looks at that column. Use the
      -- COALESCE(renewal_precise_expiration, parsed_expiration_date) pattern from the
      -- "expiring soon" intent below instead of this column alone for that class of question.
  parsed_notice_period_days INTEGER,  -- integer day count parsed from notice_period_value
  parsed_escalation_pct REAL,         -- first percentage figure parsed from annual_price_escalation_detail_value
  renewal_confidence TEXT,      -- 'high' / 'moderate' / 'low' / NULL. Filename-only heuristic over the
      -- contract_id's renewal_letter/extension documents (NOT extracted from document content). NULL means
      -- no renewal_letter/extension exists on file for this contract_id at all.
      -- 'high': a renewal filename has an explicit year-range (e.g. "..._25_26.pdf") that covers today.
      -- 'moderate': a renewal filename has an embedded date within roughly the last 18 months, no explicit range.
      -- 'low': a renewal exists on file but the filename doesn't support a confident "still current" read
      --   (stale range, unparseable, etc.) -- treat as needing manual verification, not as active.
      -- Prefer renewal_precise_still_active (below) over this column when deciding whether an
      -- 'expired_but_renewal_on_file' contract counts as "active" -- it's a specific computed date,
      -- not just a coarse year-covers-today check. Fall back to renewal_confidence IN ('high','moderate')
      -- only when renewal_precise_still_active is NULL (no 'high'-tier renewal letter to anchor a date to).
  renewal_confidence_basis TEXT, -- human-readable explanation of what filename/pattern drove the tier, for citation
  renewal_precise_expiration TEXT,     -- YYYY-MM-DD or NULL. Only populated for 'expired_but_renewal_on_file'
      -- rows that have a 'high'-tier renewal letter: the base agreement's own expiration month/day, applied
      -- to the renewal letter's year_end (the most recent one on file, if there are several). A real,
      -- specific implied date -- e.g. base expired 2024-10-11, renewal letter says "25_26" -> implied
      -- 2026-10-11 -- NOT a guess or a coarse year-range check.
  renewal_precise_still_active INTEGER -- 1 if renewal_precise_expiration > today, 0 if it already passed
      -- (a 'high'-tier renewal on file doesn't guarantee it's STILL active today -- the specific
      -- anniversary date may have already come and gone this year), NULL if no 'high'-tier renewal letter
      -- exists for this contract (renewal_confidence 'moderate'/'low'/NULL cases). Use this, not
      -- renewal_confidence alone, whenever a precise active/not-active answer is needed rather than a tier.
"""

ROUTER_TOOLS = [
    {
        "name": "query_via_sql",
        "description": (
            "Use for structured/aggregate questions answerable by querying the contracts_master "
            "table directly -- filtering, counting, listing by a field value, aggregating dollar "
            "amounts, etc. Write a single read-only SELECT query. Always include the relevant "
            "_page column(s) alongside any _value column(s) you select, so the answer can cite "
            "a source page.\n\n"
            "For a GROUP BY / aggregate-count question (e.g. 'status breakdown', 'contracts by "
            "vendor'): filename doesn't correspond to a single row and should be omitted -- select "
            "ONLY the grouping column and the count/aggregate column (exactly 2 columns), so the "
            "result can be charted.\n\n"
            "For a question that returns a LIST of contracts: when returning a list of contracts, "
            "SELECT counterparty_name_value, contract_value_value, computed_status, "
            "expiration_date_value rather than filename as the primary columns -- these are what a "
            "human actually wants to see, not a filename. Filename should only be included if the "
            "user specifically asks for it. Also include whichever specific field the question is "
            "actually about (e.g. assignment_consent_required_value for an assignment-consent "
            "question) alongside those primary columns. Always LIMIT results to 15 rows unless the "
            "user asks for a complete list, and include the true total as a scalar subquery column "
            "(e.g. `(SELECT COUNT(*) FROM contracts_master WHERE <same filter>) AS total_count`) -- "
            "NOT a second statement, only one SELECT is allowed -- so the response can say 'showing "
            "15 of X total'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "A single read-only SELECT statement against contracts_master."},
                "reasoning": {"type": "string", "description": "One sentence on why this is a structured question."},
            },
            "required": ["sql", "reasoning"],
        },
    },
    {
        "name": "query_via_vector",
        "description": (
            "Use for semantic/clause-level questions that need actual contract clause text or a "
            "modification's prose summary, not a structured field lookup -- e.g. 'what does the "
            "indemnification clause say', 'what changed in the X amendment', 'show me the assignment "
            "language in contract Y'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "search_query": {"type": "string", "description": "The text to embed and search the clause vector store with."},
                "vendor_or_contract_hint": {
                    "type": "string",
                    "description": (
                        "If the question names a specific vendor/company/contract (e.g. 'Tyler "
                        "Technologies', 'the CDW modification'), put that name here so the search can "
                        "be pre-filtered to that party's documents -- generic clause boilerplate "
                        "(e.g. 'termination') otherwise tends to dominate similarity over a company "
                        "name and can bury the right document. Omit for generic clause questions with "
                        "no specific party named."
                    ),
                },
                "reasoning": {"type": "string", "description": "One sentence on why this needs clause-level retrieval."},
            },
            "required": ["search_query", "reasoning"],
        },
    },
]

INTENT_NORMALIZATION = """
Common user intents and their correct SQL translation -- use these when the user's phrasing matches the intent, even if they don't use database column names:

- "what's the status of our contracts" / "contract status breakdown" / "how many are active" (as a general question) → GROUP BY computed_status WHERE computed_status IS NOT NULL AND computed_status != 'unknown' (the 'unknown' bucket is a data-coverage artifact -- a date that couldn't be parsed -- not a meaningful status category for a deal team, so exclude it from the default breakdown. If the user explicitly asks about unknown-status or unparseable-date contracts specifically, answer that directly and do NOT apply this exclusion.)
NOTE ON JOINS: contracts_master already has its own doc_role column -- never JOIN to
extractions for a doc_role or tier filter (that table shares nearly all the same
*_value column names, so an unqualified column reference in a joined query raises
"ambiguous column name"). Filter directly on cm.doc_role / cm.extraction_tier.

- "what's the status of our base contracts" / "base contract status" / "status of base agreements" → use the SAME corrected Active definition as "active contract breakdown" below (computed_status='active' OR (computed_status='expired_but_renewal_on_file' AND renewal_precise_still_active=1)), not the raw computed_status value alone -- raw computed_status='active' undercounts (11) because it misses the 6 expired-but-actually-renewed base agreements that resolve to still-active via renewal_precise_still_active. Scope to extraction_tier IN ('1','1.5') (excludes the 16 oversized Tier-2 base_agreement rows, which have no extracted status data anyway). Use this exact 4-bucket query so every base-status view in the app agrees with the "17 active" headline:
  SELECT 'Active' AS status, COUNT(*) AS count FROM contracts_master WHERE doc_role = 'base_agreement' AND extraction_tier IN ('1','1.5') AND (computed_status = 'active' OR (computed_status = 'expired_but_renewal_on_file' AND renewal_precise_still_active = 1))
  UNION ALL
  SELECT 'Uncontracted', COUNT(*) FROM contracts_master WHERE doc_role = 'base_agreement' AND extraction_tier IN ('1','1.5') AND computed_status = 'expired'
  UNION ALL
  SELECT 'Expiration not stated', COUNT(*) FROM contracts_master WHERE doc_role = 'base_agreement' AND extraction_tier IN ('1','1.5') AND computed_status = 'expiration_not_stated'
  UNION ALL
  SELECT 'Renewal status uncertain', COUNT(*) FROM contracts_master WHERE doc_role = 'base_agreement' AND extraction_tier IN ('1','1.5') AND computed_status = 'expired_but_renewal_on_file' AND (renewal_precise_still_active IS NULL OR renewal_precise_still_active = 0)
  ("Renewal status uncertain" = expired_but_renewal_on_file rows that don't have a high-confidence precise renewal date confirming they're still active -- distinct from "Uncontracted", which has no renewal on file at all.)
- "active contract breakdown" / "active vs uncontracted" / "portfolio risk summary" / "show active contract breakdown" / "list our active contracts" → the user wants to see the actual 17 active base contracts themselves, NOT a count/chart -- a base agreement counts as Active if computed_status = 'active' OR (computed_status = 'expired_but_renewal_on_file' AND renewal_precise_still_active = 1) -- the latter is a real, specific implied-renewal-expiration date computed from the highest-confidence renewal letter on file (see renewal_precise_expiration/renewal_precise_still_active in the schema), NOT a guess. Return the full list with key attributes, using COALESCE(renewal_precise_expiration, parsed_expiration_date) as the expiration date so the 6 renewed contracts show their real current (renewed) expiration rather than their original, already-passed one. Use this exact query, no LIMIT (the point is to enumerate all 17, not aggregate/sample them):
  SELECT counterparty_name_value AS vendor,
         parsed_effective_date AS effective_date,
         COALESCE(renewal_precise_expiration, parsed_expiration_date) AS expiration_date,
         contract_value_value AS contract_value,
         fee_structure_value AS fee_structure
  FROM contracts_master
  WHERE doc_role = 'base_agreement' AND extraction_tier IN ('1','1.5') AND (computed_status = 'active' OR (computed_status = 'expired_but_renewal_on_file' AND renewal_precise_still_active = 1))
  ORDER BY vendor
  If the user asks specifically for the Active-vs-Uncontracted COUNT (not the list), use the 2-row aggregate instead: SELECT 'Active' AS status, COUNT(*) AS count FROM contracts_master WHERE doc_role = 'base_agreement' AND extraction_tier IN ('1','1.5') AND (computed_status = 'active' OR (computed_status = 'expired_but_renewal_on_file' AND renewal_precise_still_active = 1)) UNION ALL SELECT 'Uncontracted', COUNT(*) FROM contracts_master WHERE doc_role = 'base_agreement' AND extraction_tier IN ('1','1.5') AND computed_status = 'expired'. Do NOT include a 3rd row for "immediate action" contracts in that aggregate form -- renewal_action_needed is a cross-cutting flag on an already-Active contract (e.g. Stellar Services), not a separate mutually-exclusive bucket.
- "assignment consent among our active contracts" / "of our active/portfolio contracts how many need consent" → scope to the same Active definition as above (computed_status='active' OR (computed_status='expired_but_renewal_on_file' AND renewal_precise_still_active=1)), doc_role='base_agreement', then count assignment_consent_required_value = 'Y' vs the total. Report any non-'Y' values (e.g. 'not_stated', or 'not_stated - see referenced ...') explicitly as deferred/unresolved -- do NOT silently exclude them from the denominator or round up to imply 100% coverage.
- "who are our vendors" / "which vendors" / "vendor breakdown" / "top vendors by number of contracts" → GROUP BY counterparty_name_value ORDER BY count DESC LIMIT 15, excluding rows where counterparty_name_value IS NULL or = 'not_stated' ('not_stated' is a missing-data placeholder, not a real vendor, and would otherwise show up as a fake "vendor" in the ranking). Always include the LIMIT 15 -- there are ~90 distinct vendors in the corpus, and "top vendors" means a leaderboard, not an exhaustive list; without the limit the chart becomes too tall to read.
- "what types of contracts do we have" / "contract breakdown" / "contract mix" → GROUP BY doc_role
- "how are we paying vendors" / "payment types" / "fee types" / "how contracts are structured" / "how are our base contracts structured by fee type" → GROUP BY fee_structure_value, scoped to doc_role = 'base_agreement' (base contracts only -- modifications/amendments/SOWs mostly don't restate their own fee structure and would skew this toward 'not_stated', and mixing them in double-counts a single commercial relationship across its base agreement and its amendments), exclude rows where fee_structure_value IS NULL or starts with 'not_stated'. No JOIN needed -- filter on doc_role directly, per the note above.
- "which contracts are expiring soon and need action" / "which contracts are coming up for renewal" / "renewal risk" / "what needs attention" → the renewal_action_needed flag alone MISSES contracts whose true current deadline is a renewal-implied date (renewal_precise_expiration) rather than their original, already-passed parsed_expiration_date -- e.g. several base agreements originally expired in 2024 but were renewed and are genuinely due again in the next few months; renewal_action_needed's own math only ever looks at the original date and never fires for these. Scope to the corrected Active list (computed_status='active' OR (computed_status='expired_but_renewal_on_file' AND renewal_precise_still_active=1)), base_agreement only, with an option/auto-renew mechanism, expiring in the next ~120 days by COALESCE(renewal_precise_expiration, parsed_expiration_date). Use this exact query:
  SELECT counterparty_name_value AS vendor,
         COALESCE(renewal_precise_expiration, parsed_expiration_date) AS expiration_date,
         renewal_mechanism_value AS renewal_mechanism,
         contract_value_value AS contract_value
  FROM contracts_master
  WHERE doc_role = 'base_agreement' AND extraction_tier IN ('1','1.5')
    AND (computed_status = 'active' OR (computed_status = 'expired_but_renewal_on_file' AND renewal_precise_still_active = 1))
    AND renewal_mechanism_value IN ('option_to_renew', 'auto_renew_unless_cancelled')
    AND COALESCE(renewal_precise_expiration, parsed_expiration_date) IS NOT NULL
    AND date(COALESCE(renewal_precise_expiration, parsed_expiration_date)) BETWEEN date('now') AND date('now', '+120 days')
  ORDER BY expiration_date
- "do our contracts allow assignment" / "assignment risk" / "M&A exposure" → GROUP BY assignment_consent_required_value
- "how do we procure" / "competitive vs sole source" / "procurement breakdown" → GROUP BY sole_source_vs_competitive_bid_value, exclude not_stated variants
"""

ROUTER_SYSTEM_PROMPT = f"""You are the query router for a Lake County contract due-diligence tool. Given a user's
natural-language question, decide whether it's best answered by:
(a) a SQL query against the structured contracts_master table (query_via_sql), or
(b) a semantic search against a vector store of actual contract clause text (query_via_vector).

{INTENT_NORMALIZATION}

{CONTRACTS_MASTER_SCHEMA}

Call exactly one of the two tools. If a question could use either but needs actual clause
wording (not just a Y/N or enum value), prefer query_via_vector. If it's a list/count/aggregate
over structured fields, prefer query_via_sql.

If the input is NOT a coherent question about the contracts -- gibberish, off-topic, too vague
to map to either tool -- do not call a tool. Just reply in plain text that you can't process it."""

# Scans the whole query string, including inside string literals -- a legitimate
# REPLACE() call or a filename/vendor name containing one of these words as a
# substring would be rejected. Checked against the actual corpus: no filename or
# vendor name contains any of these nine keywords, so there's no live false
# positive today. A real fix needs a SQL tokenizer to distinguish literals from
# keywords, which is disproportionate for a validated non-issue -- left as-is.
SQL_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|attach|create|replace|pragma)\b", re.IGNORECASE)


class RouterClassificationError(Exception):
    """Raised when the router model doesn't call either tool -- distinct from
    other failures so answer_question can give the specific fallback message."""


def _as_answer(text: str, stream: bool):
    """A fallback message is always a plain string internally; st.write_stream()
    requires a generator/iterable-of-chunks, not a bare str (iterating a str
    yields characters, and st.write_stream explicitly rejects str for that
    reason). Wrap consistently here so every fallback path -- not just the
    ones someone remembered to wrap by hand -- matches what the caller asked for."""
    return iter([text]) if stream else text


@contextmanager
def _timed(timings: dict, label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[label] = time.perf_counter() - t0


def _validate_readonly_sql(sql: str) -> None:
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:
        raise ValueError("Multiple statements are not allowed.")
    if not stripped.lower().startswith("select"):
        raise ValueError("Only SELECT statements are allowed.")
    if SQL_FORBIDDEN.search(stripped):
        raise ValueError("Query contains a forbidden keyword.")


def is_chart_candidate(columns: list[str], rows: list[tuple]) -> bool:
    """True if the result shape is a simple (label, number) pair per row --
    e.g. contracts-by-vendor, contracts-by-status counts -- worth a bar chart."""
    if len(columns) != 2 or not rows:
        return False
    try:
        for row in rows:
            float(row[1])
    except (TypeError, ValueError):
        return False
    return True


def should_visualize(columns: list[str], rows: list[tuple]) -> bool:
    return len(rows) > 3 or is_chart_candidate(columns, rows)


def route_question(client: anthropic.Anthropic, question: str) -> dict:
    """Returns {"path": "sql"|"vector", "sql": ..., "search_query": ..., "reasoning": ...}.
    Raises RouterClassificationError if the model doesn't call either tool."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=ROUTER_SYSTEM_PROMPT,
        tools=ROUTER_TOOLS,
        tool_choice={"type": "auto"},
        messages=[{"role": "user", "content": question}],
    )
    for block in response.content:
        if block.type == "tool_use":
            if block.name == "query_via_sql":
                return {"path": "sql", "sql": block.input["sql"], "reasoning": block.input["reasoning"]}
            elif block.name == "query_via_vector":
                return {
                    "path": "vector", "search_query": block.input["search_query"],
                    "vendor_or_contract_hint": block.input.get("vendor_or_contract_hint"),
                    "reasoning": block.input["reasoning"],
                }
    raise RouterClassificationError("Router did not call a tool.")


def run_sql_path(sql: str) -> tuple[list[str], list[tuple]]:
    _validate_readonly_sql(sql)
    with contextlib.closing(sqlite3.connect(DB_FILE)) as conn:
        cur = conn.cursor()
        cur.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        return columns, rows


def _regenerate_sql(client: anthropic.Anthropic, question: str, bad_sql: str, error_msg: str) -> str | None:
    """Ask the model once to fix a SQL query that failed to execute. Returns
    the corrected SQL, or None if the model can't produce one."""
    prompt = (
        f"Original question: {question}\n\n"
        f"This SQL query failed to execute:\n{bad_sql}\n\n"
        f"Error: {error_msg}\n\n"
        f"{CONTRACTS_MASTER_SCHEMA}\n\n"
        "Write a corrected single read-only SELECT statement against contracts_master that "
        "answers the original question and fixes the error above."
    )
    response = client.messages.create(
        model=MODEL, max_tokens=1024,
        tools=[ROUTER_TOOLS[0]], tool_choice={"type": "tool", "name": "query_via_sql"},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "query_via_sql":
            return block.input["sql"]
    return None


def run_sql_with_retry(client: anthropic.Anthropic, question: str, sql: str, timings: dict) -> tuple[list[str], list[tuple], str] | None:
    """Executes sql; on failure, asks the model to fix it ONCE and retries.
    Returns (columns, rows, final_sql) on success, or None if it still fails
    after the single retry -- callers should show FALLBACK_SQL_ERROR in that case."""
    attempted_sql = sql
    for attempt in range(MAX_SQL_RETRIES + 1):
        try:
            with _timed(timings, f"sql_execute_attempt{attempt + 1}"):
                columns, rows = run_sql_path(attempted_sql)
            return columns, rows, attempted_sql
        except Exception as e:
            if attempt >= MAX_SQL_RETRIES:
                return None
            with _timed(timings, "sql_regenerate"):
                try:
                    fixed_sql = _regenerate_sql(client, question, attempted_sql, str(e))
                except Exception:
                    fixed_sql = None
            if not fixed_sql:
                return None
            attempted_sql = fixed_sql
    return None


def _filenames_for_vendor_hint(hint: str) -> list[str]:
    """Looks up contracts_master for filenames whose counterparty_name_value
    mentions the hint (e.g. 'Tyler Technologies') -- used to pre-filter the
    vector search to that party's own documents by metadata, rather than
    relying on embedding similarity alone. See run_vector_path()'s docstring
    for why: generic clause boilerplate (e.g. "termination") can outweigh a
    company-name signal in a large corpus, burying the actually-relevant doc
    below the similarity threshold even though it's genuinely there."""
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute(
        "SELECT DISTINCT filename FROM contracts_master WHERE counterparty_name_value LIKE ?",
        (f"%{hint}%",),
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_chroma_collection() -> "chromadb.Collection":
    """Fresh Chroma client + collection handle. Callers that run many queries
    (e.g. Streamlit) should build this once via a cached factory and pass it
    into run_vector_path -- mirrors the anthropic_client/voyage_client pattern
    in answer_question(). A plain CLI call omits it and gets a fresh one."""
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return chroma_client.get_collection(COLLECTION_NAME)


def run_vector_path(voyage_client: voyageai.Client, search_query: str, timings: dict,
                     vendor_or_contract_hint: str | None = None, n_results: int = 5,
                     collection: "chromadb.Collection | None" = None) -> list[dict]:
    """Returns relevant chunks for search_query.

    If vendor_or_contract_hint names a party found in contracts_master, the
    ChromaDB query is pre-filtered by metadata to that party's own documents
    first (see _filenames_for_vendor_hint) -- chunks from a metadata-confirmed
    match are returned regardless of embedding distance, since the party match
    already establishes relevance (this is the fix for a real gap found while
    testing: "Tyler Technologies termination terms" scored WORSE by embedding
    distance, 1.1655, than a prior session's genuine gibberish query, 1.085 --
    a flat distance threshold cannot separate "on-topic but boilerplate-heavy"
    from "irrelevant" here, only a structured party match can).

    Otherwise, falls back to plain top-k search with distance <=
    VECTOR_DISTANCE_THRESHOLD -- chunks that don't clear that bar are dropped
    rather than fed to the synthesis LLM, so a nonsense/off-topic question
    doesn't get padded with irrelevant text.
    """
    with _timed(timings, "embedding_call"):
        embedding = voyage_client.embed([search_query], model="voyage-3.5", input_type="query").embeddings[0]

    collection = collection or get_chroma_collection()

    if vendor_or_contract_hint:
        matched_filenames = _filenames_for_vendor_hint(vendor_or_contract_hint)
        if matched_filenames:
            with _timed(timings, "chroma_query_filtered"):
                results = collection.query(
                    query_embeddings=[embedding], n_results=n_results,
                    where={"filename": {"$in": matched_filenames}},
                )
            chunks = [
                {"text": doc, "metadata": meta, "distance": dist}
                for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
            ]
            if chunks:
                return chunks
            # Matched a real vendor but got no chunks back (shouldn't normally
            # happen since every Tier 1/1.5 doc has chunks) -- fall through to
            # the unfiltered search below rather than give up.

    with _timed(timings, "chroma_query"):
        results = collection.query(query_embeddings=[embedding], n_results=n_results)
    chunks = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        if dist <= VECTOR_DISTANCE_THRESHOLD:
            chunks.append({"text": doc, "metadata": meta, "distance": dist})
    return chunks


def synthesize_sql_answer(client: anthropic.Anthropic, question: str, columns: list[str], rows: list[tuple],
                           visualize: bool = False, stream: bool = False):
    """Returns a str (stream=False) or a generator of text chunks (stream=True).
    When visualize=True, the result is about to be rendered as a table/chart in
    the UI, so the model is asked for a one-line headline instead of a full
    per-row citation dump -- the visual carries the detail."""
    if not rows:
        msg = "No matching rows found in contracts_master for this question."
        return iter([msg]) if stream else msg

    today = date.today().isoformat()
    table_str = " | ".join(columns) + "\n" + "\n".join(" | ".join(str(v) for v in row) for row in rows)

    total_count_hint = ""
    if "total_count" in columns:
        total = rows[0][columns.index("total_count")]
        if total is not None and total != len(rows):
            total_count_hint = (
                f"\n\nNote: results were capped -- {len(rows)} row(s) shown out of {total} total "
                f"matching contracts. Say so explicitly (e.g. 'showing {len(rows)} of {total} total')."
            )

    if visualize:
        prompt = (
            f"Today's date is {today}.\n\n"
            f"User question: {question}\n\n"
            f"SQL query results (columns then rows):\n{table_str}\n\n"
            "These results will be rendered directly as a table or chart below your response, so "
            "the user will see every row themselves. Write ONE short headline sentence introducing "
            "what the visual shows (e.g. 'Here are the contracts expiring in the next 90 days:'). "
            "Do not list individual rows, do not cite pages, do not add commentary -- just the "
            f"one-line headline.{total_count_hint}"
        )
    else:
        prompt = (
            f"Today's date is {today}.\n\n"
            f"User question: {question}\n\n"
            f"SQL query results (columns then rows):\n{table_str}\n\n"
            "The SQL WHERE clause has ALREADY correctly filtered these rows to match the question's "
            "criteria (e.g. if the question asked about a date range, every row returned already falls "
            "within that range) -- do not re-derive or second-guess date/range logic yourself, and do "
            "not conclude 'none match' when rows were returned. Simply report and cite the rows given. "
            "Write a concise, direct answer to the user's question using only this data. "
            "Cite the source for every fact using the format (filename, p.N) where N is the "
            "corresponding _page column value for the field you're citing -- if a _page value is "
            "present in the results, use it; do not invent a page number. If a field is not_stated, "
            f"say so explicitly rather than omitting it.{total_count_hint}"
        )

    if stream:
        def _gen():
            with client.messages.stream(model=MODEL, max_tokens=1024, messages=[{"role": "user", "content": prompt}]) as s:
                yield from s.text_stream
        return _gen()

    response = client.messages.create(model=MODEL, max_tokens=1024, messages=[{"role": "user", "content": prompt}])
    return response.content[0].text


def synthesize_vector_answer(client: anthropic.Anthropic, question: str, chunks: list[dict], stream: bool = False):
    """Returns a str (stream=False) or a generator of text chunks (stream=True)."""
    if not chunks:
        msg = FALLBACK_VECTOR_NO_RESULTS
        return iter([msg]) if stream else msg

    chunk_str = "\n\n---\n\n".join(
        f"[{c['metadata']['filename']}, section: {c['metadata']['section_title'] or c['metadata']['section_number'] or 'n/a'}, "
        f"page {c['metadata']['page_number']}, type: {c['metadata']['chunk_type']}]\n{c['text'][:1500]}"
        for c in chunks
    )
    prompt = (
        f"User question: {question}\n\n"
        f"Retrieved clause chunks:\n{chunk_str}\n\n"
        "Write a concise, direct answer to the user's question using only this retrieved text. "
        "Cite the source for every fact using the format (filename, section, p.N). If the "
        "retrieved chunks don't actually answer the question, say so explicitly rather than "
        "guessing or padding with unrelated content."
    )

    if stream:
        def _gen():
            with client.messages.stream(model=MODEL, max_tokens=1024, messages=[{"role": "user", "content": prompt}]) as s:
                yield from s.text_stream
        return _gen()

    response = client.messages.create(model=MODEL, max_tokens=1024, messages=[{"role": "user", "content": prompt}])
    return response.content[0].text


def answer_question(question: str, anthropic_client: anthropic.Anthropic | None = None,
                     voyage_client: voyageai.Client | None = None, stream: bool = False,
                     chroma_collection: "chromadb.Collection | None" = None) -> dict:
    """Full pipeline: route -> execute -> synthesize. Returns
    {"path", "reasoning", "answer", "raw": ..., "timings": {...}} for display/debugging.

    Callers (e.g. Streamlit) can pass in pre-built, cached clients (and a cached
    Chroma collection) to avoid re-initializing on every call; a plain CLI call
    omits them and gets fresh ones. Every failure point returns a clean fallback
    message instead of raising -- this function never lets an exception escape.
    """
    timings: dict[str, float] = {}
    try:
        anthropic_client = anthropic_client or anthropic.Anthropic()
        voyage_client = voyage_client or voyageai.Client()

        try:
            with _timed(timings, "routing_llm_call"):
                routing = route_question(anthropic_client, question)
        except RouterClassificationError:
            return {
                "path": "error", "reasoning": None,
                "answer": _as_answer(FALLBACK_ROUTER_UNCLASSIFIED, stream), "raw": None,
                "visualize": False, "timings": timings,
            }

        if routing["path"] == "sql":
            result = run_sql_with_retry(anthropic_client, question, routing["sql"], timings)
            if result is None:
                return {
                    "path": "sql", "reasoning": routing["reasoning"], "sql": routing["sql"],
                    "answer": _as_answer(FALLBACK_SQL_ERROR, stream), "raw": None,
                    "visualize": False, "timings": timings,
                }
            columns, rows, final_sql = result
            visualize = should_visualize(columns, rows)
            with _timed(timings, "synthesis_llm_call"):
                answer = synthesize_sql_answer(anthropic_client, question, columns, rows, visualize=visualize, stream=stream)
            return {
                "path": "sql", "reasoning": routing["reasoning"], "sql": final_sql,
                "answer": answer, "raw": {"columns": columns, "rows": rows},
                "visualize": visualize, "is_chart": is_chart_candidate(columns, rows), "timings": timings,
            }
        else:
            chunks = run_vector_path(
                voyage_client, routing["search_query"], timings,
                vendor_or_contract_hint=routing.get("vendor_or_contract_hint"),
                collection=chroma_collection,
            )
            if not chunks:
                return {
                    "path": "vector", "reasoning": routing["reasoning"], "search_query": routing["search_query"],
                    "answer": _as_answer(FALLBACK_VECTOR_NO_RESULTS, stream),
                    "raw": [], "visualize": False, "timings": timings,
                }
            with _timed(timings, "synthesis_llm_call"):
                answer = synthesize_vector_answer(anthropic_client, question, chunks, stream=stream)
            return {
                "path": "vector", "reasoning": routing["reasoning"], "search_query": routing["search_query"],
                "answer": answer, "raw": chunks, "visualize": False, "timings": timings,
            }
    except Exception:
        return {
            "path": "error", "reasoning": None,
            "answer": _as_answer(FALLBACK_UNHANDLED, stream), "raw": None,
            "visualize": False, "timings": timings,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()
    result = answer_question(args.question)
    print(f"[path: {result['path']}] {result['reasoning']}\n")
    print(result["answer"])
    print("\ntimings:", {k: f"{v:.3f}s" for k, v in result["timings"].items()})
