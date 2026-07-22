"""Field-by-field comparison of pipeline output vs GT_FILE (currently
Berkshire_Ground_Truth_Labels_v14.xlsx) for all 17 hand-labeled docs, across all tiers."""
import sqlite3
import openpyxl

GT_FILE = "Berkshire_Ground_Truth_Labels_v14.xlsx"
DB_FILE = "contracts.db"

# Tier 1/1.5 fields: (ground_truth_col, extractions_table_value_col)
TIER1_FIELD_MAP = [
    ("counterparty_name", "counterparty_name_value"),
    ("org_role", "org_role_value"),
    ("effective_date", "effective_date_value"),
    ("expiration_date", "expiration_date_value"),
    ("renewal_mechanism", "renewal_mechanism_value"),
    ("contract_value", "contract_value_value"),
    ("fee_structure", "fee_structure_value"),
    ("payment_terms", "payment_terms_value"),
    # ground truth has one combined column; pipeline splits into type+detail
    ("annual_price_escalation", ("annual_price_escalation_type_value", "annual_price_escalation_detail_value")),
    ("termination_for_convenience", "termination_for_convenience_value"),
    ("notice_period", "notice_period_value"),
    ("assignment_consent_required", "assignment_consent_required_value"),
    ("indemnity_present", "indemnity_present_value"),
    ("insurance_required", "insurance_required_value"),
    ("sole_source_vs_competitive_bid", "sole_source_vs_competitive_bid_value"),
    ("signer_title", "signer_title_value"),
    ("fiscal_year_appropriation_contingent", "fiscal_year_appropriation_contingent_value"),
    ("modification_summary", "modification_summary_value"),
    ("modification_type", "modification_type"),  # flat in both, but nested under modification_summary in the tool schema
]


import re

from date_utils import normalize_date

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "is", "are",
    "was", "were", "be", "this", "that", "with", "as", "by", "from", "not", "no",
    "it", "its", "per", "upon", "than", "then", "if", "but", "into", "within",
}


_NOT_STATED_PREFIXES = ("not_stated", "not stated", "not_applicable", "not applicable",
                         "not_applicable_doc_type")


def _tier_str(v):
    """Coerce Excel int/float tier values (e.g. 1.5 stored as float, 1 stored as 1.0)
    to the string form the tier comparisons expect."""
    if v is None:
        return ""
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def norm(v):
    if v is None:
        return ""
    s = str(v).strip()
    s_low = s.lower()
    for prefix in _NOT_STATED_PREFIXES:
        if s_low.startswith(prefix):
            return ""
    if s_low in ("n/a", "na"):
        return ""
    s = re.sub(r"\s*\(p\.?\s*\d+[a-z]?(\s*,\s*p?\.?\s*\d+[a-z]?)*\)\s*", " ", s)  # strip "(p.15)" / "(p.1, mention of RFP)"
    s = s.strip(" -")
    return s.lower()


def _significant_words(s):
    words = re.findall(r"[a-z0-9%.$]+", s.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def word_overlap_match(gt_n, pipe_n, threshold=0.7):
    """Fraction of ground truth's significant words that also appear in pipeline's
    answer. Catches same-fact-different-wording and pipeline-gives-a-detailed-superset
    cases that strict substring matching misses (common on prose fields)."""
    gt_words = _significant_words(gt_n)
    if not gt_words:
        return False
    pipe_words = set(_significant_words(pipe_n))
    hits = sum(1 for w in gt_words if w in pipe_words)
    return (hits / len(gt_words)) >= threshold


DOLLAR_PATTERN = re.compile(r"\$\s*([\d,]+(?:\.\d{1,2})?)")


def dollar_amounts(s) -> set[float]:
    """Extract every $-prefixed number as a float, comma stripped. Deliberately
    run on the RAW (pre-norm()) string, not the normalized one -- norm() blanks
    any string starting with 'not_stated' to "", which would otherwise discard
    a real dollar figure embedded in a hedge like 'not_stated - see referenced
    Agreement 20022 (... capped at $328,875)'. Also sidesteps
    word_overlap_match's comma-splitting of "$373,250" into separate "$373"/
    "250" tokens, which made a short GT phrase like 'max $373,250' fail to
    match pipeline text using a synonym ('not-to-exceed') for 'max' even
    though the actual dollar figure was identical."""
    if not s:
        return set()
    return {float(m.replace(",", "")) for m in DOLLAR_PATTERN.findall(str(s))}


def fuzzy_match(gt, pipe):
    gt_n, pipe_n = norm(gt), norm(pipe)
    if gt_n == pipe_n:
        return True
    if gt_n and pipe_n and (gt_n in pipe_n or pipe_n in gt_n):
        return True
    gt_date, pipe_date = normalize_date(gt_n), normalize_date(pipe_n)
    if gt_date and pipe_date and gt_date == pipe_date:
        return True
    gt_amounts, pipe_amounts = dollar_amounts(gt), dollar_amounts(pipe)
    if gt_amounts and pipe_amounts and (gt_amounts & pipe_amounts):
        return True
    if gt_n and pipe_n and word_overlap_match(gt_n, pipe_n):
        return True
    return False


def load_ground_truth():
    wb = openpyxl.load_workbook(GT_FILE, data_only=True)
    ws = wb["Ground Truth Labels"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    records = {}
    for row in rows[1:]:
        if row[0] is None or row.count(None) >= len(row) - 1 or row[3] is None:
            continue  # skip section-divider rows
        rec = dict(zip(header, row))
        rec["extraction_tier"] = _tier_str(rec.get("extraction_tier"))
        records[rec["filename"]] = rec
    return records


def main():
    gt = load_ground_truth()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    extractions = {r["filename"]: r for r in cur.execute("SELECT * FROM extractions")}
    existence = {r["filename"]: r for r in cur.execute("SELECT * FROM existence_only")}
    rate_adj = {r["filename"]: r for r in cur.execute("SELECT * FROM rate_adjustments")}

    tier1_files = [f for f, r in gt.items() if r["extraction_tier"] in ("1", "1.5")]

    print("=" * 90)
    print(f"TIER 1/1.5 — full field-by-field comparison ({len(tier1_files)} docs)")
    print("=" * 90)

    # Two separate tests, never blended:
    #   "value" test  = GT has a real stated value -- can the pipeline extract the
    #                    correct fact? The harder, more informative test.
    #   "recognition" test = GT explicitly says not_stated/n-a -- does the pipeline
    #                    correctly recognize the field isn't there, rather than
    #                    inventing a value? A real pass/fail, not a free pass --
    #                    "GT blank" was previously skipped entirely (neither
    #                    credited nor penalized), which silently dropped both a
    #                    pipeline hallucination case and 6+ correct abstentions
    #                    from the reported number.
    field_stats = {gt_col: {"value_match": 0, "value_mismatch": 0,
                             "recog_match": 0, "recog_mismatch": 0}
                   for gt_col, _ in TIER1_FIELD_MAP}
    mismatches = []
    recog_flags = []  # GT says not_stated but pipeline gave a value -- needs a human look

    for fname in tier1_files:
        gt_row = gt[fname]
        ext_row = extractions.get(fname)
        if ext_row is None:
            print(f"\n!! MISSING from extractions table: {fname}")
            continue

        print(f"\n--- {fname} ---")
        for gt_col, pipe_col in TIER1_FIELD_MAP:
            gt_val = gt_row.get(gt_col)
            if isinstance(pipe_col, tuple):
                parts = [str(ext_row[c]) for c in pipe_col if ext_row[c] and norm(str(ext_row[c])) != ""]
                pipe_val = " | ".join(parts) if parts else None
            else:
                pipe_val = ext_row[pipe_col]

            # annual_price_escalation is scored as binary by design (per user
            # decision): "Y" (an escalation mechanism is explicitly documented)
            # vs. everything else. GT's own labeling convention already
            # collapses "silent" and "explicitly no escalation" into the same
            # not_stated/n-a cell, so the pipeline's 'none' enum value (which
            # means exactly that: no escalation mechanism found) is treated as
            # equivalent to blank here -- but ONLY for this field. Other enum
            # fields with a real 'none' value (e.g. renewal_mechanism) are left
            # alone, since "none" there is a distinct, meaningful fact GT wasn't
            # asked to treat as equivalent to not_stated.
            if gt_col == "annual_price_escalation":
                type_val = ext_row["annual_price_escalation_type_value"]
                if type_val and norm(type_val) == "none":
                    pipe_val = None

            if norm(gt_val) == "":
                # Recognition test: GT explicitly labeled this not_stated/n-a.
                if norm(pipe_val) == "":
                    field_stats[gt_col]["recog_match"] += 1
                else:
                    field_stats[gt_col]["recog_mismatch"] += 1
                    recog_flags.append((fname, gt_col, gt_val, pipe_val))
                    print(f"  RECOGNITION MISMATCH  {gt_col}: GT says not_stated, pipeline gave a value:")
                    print(f"    pipeline: {pipe_val!r}")
                continue

            # Value test: GT has a real stated value.
            if fuzzy_match(gt_val, pipe_val):
                field_stats[gt_col]["value_match"] += 1
            else:
                field_stats[gt_col]["value_mismatch"] += 1
                mismatches.append((fname, gt_col, gt_val, pipe_val))
                print(f"  MISMATCH  {gt_col}:")
                print(f"    ground truth: {gt_val!r}")
                print(f"    pipeline:     {pipe_val!r}")

    total_value_match = sum(s["value_match"] for s in field_stats.values())
    total_value_scored = sum(s["value_match"] + s["value_mismatch"] for s in field_stats.values())
    overall_value_pct = f"{total_value_match / total_value_scored * 100:.1f}%" if total_value_scored else "n/a"

    print("\n" + "=" * 90)
    print(f"overall (value test only, GT-has-a-value rows): {total_value_match}/{total_value_scored} = {overall_value_pct}")
    print("TIER 1/1.5 — accuracy by field, value test vs. recognition test reported separately")
    print("=" * 90)
    print(f"{'field':<38} {'value: match/total':>19} {'acc':>6} {'recog: match/total':>19} {'acc':>6}  base-rate flag")
    for gt_col, stats in field_stats.items():
        v_total = stats["value_match"] + stats["value_mismatch"]
        r_total = stats["recog_match"] + stats["recog_mismatch"]
        v_acc = f"{stats['value_match']/v_total*100:.0f}%" if v_total else "n/a"
        r_acc = f"{stats['recog_match']/r_total*100:.0f}%" if r_total else "n/a"
        grand_total = v_total + r_total
        # Flag: if the recognition-test denominator dominates the field's total
        # rows (>=70%), a blended/overall number for this field is mostly driven
        # by "correctly says nothing here" rather than genuine value extraction --
        # worth calling out before anyone reads a high blended % as extraction skill.
        flag = ""
        if grand_total and r_total / grand_total >= 0.7:
            flag = f"BASE-RATE DRIVEN ({r_total}/{grand_total} rows are recognition-only)"
        v_frac = f"{stats['value_match']}/{v_total}"
        r_frac = f"{stats['recog_match']}/{r_total}"
        print(f"{gt_col:<38} {v_frac:>19} {v_acc:>6} {r_frac:>19} {r_acc:>6}  {flag}")

    if recog_flags:
        print("\n" + "=" * 90)
        print(f"RECOGNITION MISMATCHES needing human review ({len(recog_flags)}): GT says not_stated, pipeline gave a value")
        print("=" * 90)
        for fname, col, gt_val, pipe_val in recog_flags:
            print(f"  {fname} / {col}: pipeline said {pipe_val!r} (GT: {gt_val!r})")

    tier24_files = [f for f, r in gt.items()
                    if (r["extraction_tier"].startswith("2") or r["extraction_tier"].startswith("4"))
                    and r["doc_role"] != "rate_adjustment"]

    print("\n" + "=" * 90)
    print(f"TIER 2/4 — existence-only docs ({len(tier24_files)}): confirm identity fields only")
    print("=" * 90)
    for fname in tier24_files:
        gt_row = gt[fname]
        ex_row = existence.get(fname)
        if ex_row is None:
            print(f"  !! MISSING from existence_only table: {fname}")
            continue
        contract_id_ok = str(ex_row["contract_id"]) == str(gt_row["contract_id"])
        doc_role_ok = fuzzy_match(gt_row["doc_role"], ex_row["doc_role"])
        print(f"  {fname}: contract_id {'OK' if contract_id_ok else 'MISMATCH'} "
              f"({ex_row['contract_id']} vs {gt_row['contract_id']}), "
              f"doc_role {'OK' if doc_role_ok else 'MISMATCH'} "
              f"({ex_row['doc_role']} vs {gt_row['doc_role']})")

    print("\n" + "=" * 90)
    print("RATE_ADJUSTMENT doc (1, descoped from Tier 3 to Tier 4 -- extraction already run pre-descoping):")
    print("note ground truth has no specialized-schema columns")
    print("=" * 90)
    tier3_files = [f for f, r in gt.items() if r["doc_role"] == "rate_adjustment"]
    for fname in tier3_files:
        ra_row = rate_adj.get(fname)
        if ra_row is None:
            print(f"  !! MISSING from rate_adjustments table: {fname}")
            continue
        cols = [d[1] for d in cur.execute("PRAGMA table_info(rate_adjustments)")]
        print(f"  {fname}: present in rate_adjustments table.")
        for c in cols:
            if c.endswith("_value"):
                print(f"    {c}: {ra_row[c]!r}")
        print("  (Ground truth spreadsheet uses the general 19-field contract schema and has no "
              "columns for the specialized rate_adjustment schema (sku_or_material_type, "
              "year_of_initial_pricing, initial_price, etc.) — this doc cannot be scored field-by-field "
              "against v10 as-is; would need dedicated ground-truth columns to verify.)")

    print("\n" + "=" * 90)
    print(f"SUMMARY: {len(mismatches)} value-test mismatches + {len(recog_flags)} recognition-test mismatches "
          f"across {len(tier1_files)} Tier 1/1.5 docs")
    print("=" * 90)


if __name__ == "__main__":
    main()
