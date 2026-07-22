"""
Retrieval/router eval: 10 test cases (5 SQL-path, 5 vector-path) run against the
actual query_router.answer_question() pipeline -- not mocked, not hand-picked
after the fact. Each case has an explicit, checkable pass criterion grounded in
facts already verified against contracts.db in this session (see CLAUDE.md).

This is a NEW script written for the eval slide -- there was no prior
eval_retrieval.py or eval run in this repo before this one. Run:
    py eval_retrieval.py
Prints per-case detail and a final summary count. Nothing here is invented --
every printed pass/fail is computed from the live pipeline's actual output.
"""

import re
import sqlite3
import time

from query_router import answer_question, FALLBACK_ROUTER_UNCLASSIFIED

DB_FILE = "contracts.db"


def _rows_as_dicts(result):
    if not result.get("raw") or not isinstance(result["raw"], dict):
        return []
    cols, rows = result["raw"]["columns"], result["raw"]["rows"]
    return [dict(zip(cols, r)) for r in rows]


# ---- pass-criteria checkers ------------------------------------------------
# Each returns (passed: bool, detail: str). "detail" always states what was
# actually observed, so a failure explains itself without re-running anything.

def check_base_status(result):
    """NOTE -- date-sensitive: the 17/16 split depends on renewal_precise_still_active,
    which was computed against date.today() at the last build_master_table.py run, not
    at eval time (see README/CLAUDE.md Limitations on computed_status being a snapshot).
    A failure here after enough calendar drift without a rebuild means "data needs
    refreshing," not necessarily a router/SQL regression -- check contracts_master's
    last-build date before assuming the latter."""
    if result["path"] != "sql":
        return False, f"expected SQL path, got '{result['path']}'"
    rows = {str(r[0]).strip().lower(): r[1] for r in result["raw"]["rows"]} if result.get("raw") else {}
    active, uncontracted = rows.get("active"), rows.get("uncontracted")
    ok = active == 17 and uncontracted == 16
    return ok, f"rows={rows}"


def check_active_list(result):
    """NOTE -- date-sensitive, same root cause as check_base_status above: the expected
    row count of 17 depends on a build-time snapshot, not live date math."""
    if result["path"] != "sql":
        return False, f"expected SQL path, got '{result['path']}'"
    rows = _rows_as_dicts(result)
    n = len(rows)
    vendors = [str(r.get("vendor", "")) for r in rows]
    has_stellar = any("Stellar Services" in v for v in vendors)
    ok = n == 17 and has_stellar
    return ok, f"{n} rows returned (expected 17); Stellar Services present: {has_stellar}"


def check_expiring_soon(result):
    if result["path"] != "sql":
        return False, f"expected SQL path, got '{result['path']}'"
    rows = _rows_as_dicts(result)
    vendors = [str(r.get("vendor", "")) for r in rows]
    has_stellar = any("Stellar Services" in v for v in vendors)
    ok = has_stellar and len(rows) >= 1
    return ok, f"{len(rows)} row(s): {vendors}"


def check_fee_type(result):
    """NOT date-sensitive, unlike check_base_status/check_active_list above --
    fee_structure_value is a static extracted field with no date math involved; a
    failure here means the extraction data actually changed, not calendar drift."""
    if result["path"] != "sql":
        return False, f"expected SQL path, got '{result['path']}'"
    rows = {str(r[0]).strip().lower(): r[1] for r in result["raw"]["rows"]} if result.get("raw") else {}
    ok = rows.get("fixed_fee") == 31 and rows.get("rate_based") == 29 and "not_stated" not in rows
    return ok, f"rows={rows}"


def check_top_vendors(result):
    if result["path"] != "sql":
        return False, f"expected SQL path, got '{result['path']}'"
    rows = _rows_as_dicts(result)
    if not rows:
        return False, "no rows returned"
    first_vendor = str(rows[0].get("counterparty_name_value", ""))
    first_count = rows[0].get(list(rows[0].keys())[1]) if len(rows[0]) > 1 else None
    no_not_stated = all("not_stated" not in str(r.get("counterparty_name_value", "")) for r in rows)
    ok = "Applied Technologies" in first_vendor and first_count == 6 and no_not_stated
    return ok, f"top row: {rows[0]}; not_stated excluded: {no_not_stated}; {len(rows)} rows total"


def check_indemnification(result):
    if result["path"] != "vector":
        return False, f"expected vector path, got '{result['path']}'"
    answer = result["answer"] if isinstance(result["answer"], str) else "".join(result["answer"])
    has_citation = bool(re.search(r"\([^)]+\.pdf[^)]*p\.\d+\)", answer))
    mentions_indemnify = "indemnif" in answer.lower()
    has_chunks = bool(result.get("raw"))
    ok = mentions_indemnify and has_citation and has_chunks
    return ok, f"chunks retrieved: {len(result.get('raw') or [])}; has page citation: {has_citation}; mentions indemnification: {mentions_indemnify}"


def check_tyler_termination(result):
    if result["path"] != "vector":
        return False, f"expected vector path, got '{result['path']}'"
    chunks = result.get("raw") or []
    filenames = [c["metadata"]["filename"] for c in chunks]
    has_tyler_doc = any("14234" in f or "TYLER" in f.upper() for f in filenames)
    ok = has_tyler_doc
    return ok, f"retrieved filenames: {filenames}"


def check_cdw_modification_fails_safely(result):
    """Known weak case (flagged in a prior session): the vector store doesn't
    have good chunk-level coverage of the CDW change order itself. Pass
    criterion here is NOT 'gives the right answer' -- it's 'admits it doesn't
    know rather than inventing specifics'. This is the deliberate fails-safely
    check, distinct from the accuracy checks above."""
    if result["path"] != "vector":
        return False, f"expected vector path, got '{result['path']}'"
    answer = result["answer"] if isinstance(result["answer"], str) else "".join(result["answer"])
    hedge_markers = ["cannot identify", "does not contain", "do not contain", "not available",
                      "no information", "would need", "not included", "unable to determine",
                      "doesn't contain", "don't contain", "not specify", "not specified"]
    hedges = any(m in answer.lower() for m in hedge_markers)
    return hedges, f"hedge language present: {hedges}\nfull answer: {answer}"


def check_gibberish_rejected(result):
    answer = result["answer"] if isinstance(result["answer"], str) else "".join(result["answer"])
    ok = result["path"] == "error" and answer == FALLBACK_ROUTER_UNCLASSIFIED
    return ok, f"path={result['path']}, answer matches fallback: {answer == FALLBACK_ROUTER_UNCLASSIFIED}"


def check_ciorba_named_entity(result):
    """The 'named entity at 1,457-chunk scale' case: does the vendor-hint
    metadata pre-filter (added in a prior session) actually surface Ciorba
    Group's own documents, or does generic clause boilerplate outrank the
    company-name signal? Genuinely unknown until run -- this is not a
    pre-verified fact like the others."""
    if result["path"] != "vector":
        return False, f"expected vector path, got '{result['path']}'"
    chunks = result.get("raw") or []
    filenames = [c["metadata"]["filename"] for c in chunks]
    distances = [round(c["distance"], 4) for c in chunks]
    has_ciorba_doc = any("22128" in f or "CIORBA" in f.upper() or "Ciorba" in f for f in filenames)
    return has_ciorba_doc, f"retrieved filenames: {filenames}; distances: {distances}"


TEST_CASES = [
    {"id": 1, "query": "What's the status of our base contracts?", "expected_path": "sql",
     "criterion": "Active count == 17 and Uncontracted count == 16 (the corrected walkthrough headline)",
     "check": check_base_status},
    {"id": 2, "query": "Show active contract breakdown", "expected_path": "sql",
     "criterion": "returns all 17 active base contracts as a list (not a 2-row count), Stellar Services present",
     "check": check_active_list},
    {"id": 3, "query": "Which contracts are expiring soon and need action?", "expected_path": "sql",
     "criterion": "Stellar Services LLC present in results",
     "check": check_expiring_soon},
    {"id": 4, "query": "How are our base contracts structured by fee type?", "expected_path": "sql",
     "criterion": "fixed_fee == 31, rate_based == 29 (verified counts, base_agreement scope only), no not_stated bucket",
     "check": check_fee_type},
    {"id": 5, "query": "Who are our top vendors by number of contracts?", "expected_path": "sql",
     "criterion": "top vendor is Applied Technologies, Inc. with count 6; 'not_stated' excluded from ranking",
     "check": check_top_vendors},
    {"id": 6, "query": "What does the indemnification clause say?", "expected_path": "vector",
     "criterion": "routes to vector search, retrieves chunks, cites (filename, p.N), answer discusses indemnification",
     "check": check_indemnification},
    {"id": 7, "query": "What are the termination terms in the Tyler Technologies agreement?", "expected_path": "vector",
     "criterion": "vendor-hint pre-filter surfaces the actual Tyler Technologies document (14234...)",
     "check": check_tyler_termination},
    {"id": 8, "query": "What changed in the CDW modification?", "expected_path": "vector",
     "criterion": "FAILS SAFELY: known weak retrieval case -- pass means the answer admits uncertainty rather than inventing specifics",
     "check": check_cdw_modification_fails_safely},
    {"id": 9, "query": "asdfjkl random nonsense", "expected_path": "error",
     "criterion": "router declines to call a tool and returns the router-unclassified fallback, not a wrong SQL/vector guess",
     "check": check_gibberish_rejected},
    {"id": 10, "query": "What does the indemnification clause in the Ciorba Group agreement say?", "expected_path": "vector",
     "criterion": "named-entity query at full 1,457-chunk scale: vendor-hint pre-filter must surface Ciorba Group's own document, not generic indemnification boilerplate from an unrelated vendor",
     "check": check_ciorba_named_entity},
]


def main():
    conn = sqlite3.connect(DB_FILE)
    conn.close()  # just confirms the DB is reachable before burning API calls

    results = []
    for case in TEST_CASES:
        print("=" * 90)
        print(f"[{case['id']}] QUERY: {case['query']}")
        print(f"    expected path: {case['expected_path']}")
        print(f"    pass criterion: {case['criterion']}")
        t0 = time.perf_counter()
        result = answer_question(case["query"], stream=False)
        elapsed = time.perf_counter() - t0
        passed, detail = case["check"](result)
        status = "PASS" if passed else "FAIL"
        print(f"    routed to: {result['path']}  ({elapsed:.2f}s)")
        if result.get("reasoning"):
            print(f"    router reasoning: {result['reasoning']}")
        if result.get("sql"):
            print(f"    SQL: {result['sql']}")
        if result.get("search_query"):
            print(f"    search_query: {result['search_query']}")
        print(f"    -> {status}: {detail}")
        results.append({"id": case["id"], "query": case["query"], "path": result["path"],
                         "passed": passed, "detail": detail})
        print()

    # Grouped by each case's own declared expected_path, not positional slicing --
    # robust to TEST_CASES being reordered or resized later.
    sql_cases = [r for r in results if TEST_CASES[r["id"] - 1]["expected_path"] == "sql"]
    vector_cases = [r for r in results if TEST_CASES[r["id"] - 1]["expected_path"] in ("vector", "error")]
    n_pass = sum(1 for r in results if r["passed"])

    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Total: {n_pass}/{len(results)} passed")
    print(f"SQL-path cases: {sum(1 for r in sql_cases if r['passed'])}/{len(sql_cases)} passed")
    print(f"Vector-path cases: {sum(1 for r in vector_cases if r['passed'])}/{len(vector_cases)} passed")
    for r in results:
        print(f"  [{r['id']}] {'PASS' if r['passed'] else 'FAIL'}  (routed: {r['path']})  {r['query']}")


if __name__ == "__main__":
    main()
