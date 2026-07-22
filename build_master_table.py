"""
Builds `contracts_master`: one row per contract document across ALL tiers,
satisfying the assignment's "a row per contract file" requirement.

- Tier 1/1.5 rows: 19 extracted features populated from `extractions`.
- Tier 2/4 rows: extracted features NULL -- nothing was extracted, these
  rows exist for document-identity completeness only.
- Tier 3 (rate_adjustment) rows: also NULL on the 19 extracted features here.
  Their data is a pricing-event record with a different shape, not a
  contract record -- it lives in `rate_adjustments`, joinable by
  contract_id for the vendor rate-escalation analysis specifically. Forcing
  it into this table's columns would misrepresent what a rate-adjustment
  letter actually contains.

extraction_tier is derived from doc_role for Tier 1/1.5 rows (the tiering
rules are a deterministic function of doc_role -- base_agreement/sow => 1,
modification/amendment => 1.5) since `extractions` doesn't store the raw
tier value from the source CSV. Tier 2/4 pull their tier directly from
existence_only.tier; Tier 3 is hardcoded (single-purpose table).

Rebuild this after any extraction run to keep it in sync -- deliberately a
plain rebuildable script rather than a SQL view, since the derived columns
(text_quality, source_pages) are much simpler to compute in Python than as
multi-column SQL string concatenation.

Usage:
    python build_master_table.py --db contracts.db
"""

import argparse
import re
import sqlite3
from datetime import date

from extract_contracts import FIELDS
from date_utils import normalize_date, parse_duration, parse_notice_period_days, parse_escalation_pct
from renewal_confidence import best_tier_per_contract, classify_renewal_filename, precise_renewal_expiration

TIER_BY_DOC_ROLE = {
    "base_agreement": "1", "sow": "1",
    "modification": "1.5", "amendment": "1.5",
}

RENEWAL_ROLES = {"renewal_letter", "extension"}
AUTO_OR_OPTIONAL_RENEWAL = {"auto_renew_unless_cancelled", "option_to_renew"}
RENEWAL_ACTION_WINDOW_DAYS = 90


def build(conn: sqlite3.Connection):
    conn.execute("DROP TABLE IF EXISTS contracts_master")
    feature_cols_sql = ", ".join(f'"{f}_value" TEXT, "{f}_page" INTEGER' for f in FIELDS)
    conn.execute(f"""
        CREATE TABLE contracts_master (
            filename TEXT PRIMARY KEY,
            contract_id TEXT,
            doc_role TEXT,
            extraction_tier TEXT,
            text_quality TEXT,
            source_pages TEXT,
            notes TEXT,
            "modification_type" TEXT,
            {feature_cols_sql},
            parsed_effective_date TEXT,
            parsed_expiration_date TEXT,
            computed_status TEXT,
            renewal_action_needed INTEGER,
            parsed_notice_period_days INTEGER,
            parsed_escalation_pct REAL,
            renewal_confidence TEXT,
            renewal_confidence_basis TEXT,
            renewal_precise_expiration TEXT,
            renewal_precise_still_active INTEGER
        )
    """)

    feature_cols = [f"{f}_value" for f in FIELDS] + [f"{f}_page" for f in FIELDS] + ["modification_type"]
    insert_cols = ["filename", "contract_id", "doc_role", "extraction_tier", "text_quality", "source_pages", "notes"] + feature_cols
    placeholders = ", ".join("?" for _ in insert_cols)
    col_list = ", ".join(f'"{c}"' for c in insert_cols)
    insert_sql = f"INSERT OR REPLACE INTO contracts_master ({col_list}) VALUES ({placeholders})"

    n_tier1, n_tier2_4, n_tier3 = 0, 0, 0

    # Tier 1 / 1.5 -- from extractions
    for r in conn.execute("SELECT * FROM extractions").fetchall():
        cols = [d[0] for d in conn.execute("SELECT * FROM extractions LIMIT 1").description]
        row = dict(zip(cols, r))
        pages = sorted({row[f"{f}_page"] for f in FIELDS if row.get(f"{f}_page") is not None})
        source_pages = ", ".join(f"p.{p}" for p in pages) if pages else None
        notes_parts = []
        if row.get("fields_missing_page"):
            notes_parts.append(f"fields_missing_page: {row['fields_missing_page']}")
        if row.get("modification_summary_issue"):
            notes_parts.append(f"modification_summary_issue: {row['modification_summary_issue']}")
        notes = "; ".join(notes_parts) or None
        text_quality = "scanned" if row.get("used_vision") else "digital"
        tier = TIER_BY_DOC_ROLE.get(row["doc_role"], row["doc_role"])

        values = [row["filename"], row["contract_id"], row["doc_role"], tier, text_quality, source_pages, notes]
        values += [row[f"{f}_value"] for f in FIELDS] + [row[f"{f}_page"] for f in FIELDS] + [row["modification_type"]]
        conn.execute(insert_sql, values)
        n_tier1 += 1

    # Tier 2 / 4 -- from existence_only, extracted features all NULL
    null_features = [None] * (len(FIELDS) * 2 + 1)
    for r in conn.execute("SELECT filename, contract_id, doc_role, tier FROM existence_only").fetchall():
        filename, contract_id, doc_role, tier = r
        values = [filename, contract_id, doc_role, str(tier), None, None, None] + null_features
        conn.execute(insert_sql, values)
        n_tier2_4 += 1

    # Tier 3 -- from rate_adjustments, extracted features NULL here (real
    # data lives in rate_adjustments, join on contract_id)
    for r in conn.execute("SELECT filename, contract_id, used_vision FROM rate_adjustments").fetchall():
        filename, contract_id, used_vision = r
        text_quality = "scanned" if used_vision else "digital"
        notes = "Rate-adjustment record; see `rate_adjustments` table (joined on contract_id) for pricing-event fields."
        values = [filename, contract_id, "rate_adjustment", "3", text_quality, None, notes] + null_features
        conn.execute(insert_sql, values)
        n_tier3 += 1

    conn.commit()
    actual_total = conn.execute("SELECT COUNT(*) FROM contracts_master").fetchone()[0]
    naive_total = n_tier1 + n_tier2_4 + n_tier3
    print(f"contracts_master built: {n_tier1} Tier 1/1.5 rows, {n_tier2_4} Tier 2/4 rows, {n_tier3} Tier 3 rows")
    print(f"Total: {actual_total} rows"
          + (f" ({naive_total - actual_total} filename collision(s) across source tables, "
             f"e.g. a doc reclassified to existence_only that still has a leftover rate_adjustments "
             f"row -- later INSERT OR REPLACE wins, not a bug)" if actual_total != naive_total else ""))

    status_counts = _compute_date_columns(conn)
    print(f"computed_status: {status_counts}")

    _compute_renewal_confidence(conn)
    _compute_precise_renewal_status(conn)


def _compute_date_columns(conn: sqlite3.Connection) -> dict:
    """Post-processing pass over Tier 1/1.5 rows: parse effective_date/
    expiration_date prose into computable date columns (no re-extraction --
    pure text parsing over data already in contracts_master), then derive
    computed_status and renewal_action_needed.

    computed_status accounts for renewal letters on file: if the base
    agreement's own dates say 'expired' but a renewal_letter/extension
    exists for the same contract_id (Tier 2, existence-only), the status is
    'expired_but_renewal_on_file' instead of a flat 'expired' -- continuation
    is likely, but the actual renewed end date isn't known, so this is
    deliberately not folded into 'active'.

    'expiration_not_stated' (added 2026-07-21, was folded into 'unknown'
    before this) is distinct from 'unknown': it fires when the effective date
    parsed successfully (so we know the contract started and isn't in the
    future) but no expiration date could be resolved at all -- a clean,
    legitimate fact about the document (often it just doesn't state a fixed
    end date), not a parsing failure. 'unknown' is now reserved for when the
    effective date itself couldn't be resolved -- genuinely don't know
    anything about this contract's timeline, which is a real data gap."""
    today = date.today()

    renewed_contract_ids = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT contract_id FROM existence_only WHERE doc_role IN (?, ?)",
            tuple(RENEWAL_ROLES),
        ).fetchall()
        if r[0] and r[0] != "UNKNOWN"
    }

    rows = conn.execute("""
        SELECT filename, contract_id, effective_date_value, expiration_date_value,
               renewal_mechanism_value, notice_period_value, annual_price_escalation_detail_value
        FROM contracts_master
        WHERE extraction_tier IN ('1', '1.5')
    """).fetchall()

    status_counts: dict[str, int] = {}

    for (filename, contract_id, eff_raw, exp_raw, renewal_mech, notice_raw, escalation_raw) in rows:
        parsed_eff = normalize_date(eff_raw)

        parsed_exp = normalize_date(exp_raw)
        if parsed_exp is None and parsed_eff is not None:
            delta = parse_duration(exp_raw)
            if delta is not None:
                parsed_exp = (date.fromisoformat(parsed_eff) + delta).isoformat()

        # Guard against a source document's own internal date inconsistency (seen in
        # practice: an expiration_date typo'd a year earlier than the effective date,
        # which ground truth flagged manually) -- don't compute a confident status off
        # an expiration date that precedes the effective date, that's not a real term.
        if parsed_eff is not None and parsed_exp is not None and parsed_exp < parsed_eff:
            parsed_exp = None

        if parsed_eff is None:
            status = "unknown"
        else:
            eff_date_obj = date.fromisoformat(parsed_eff)
            if eff_date_obj > today:
                status = "not_yet_effective"
            elif parsed_exp is None:
                status = "expiration_not_stated"
            else:
                exp_date_obj = date.fromisoformat(parsed_exp)
                if exp_date_obj < today:
                    status = "expired_but_renewal_on_file" if contract_id in renewed_contract_ids else "expired"
                elif eff_date_obj <= today <= exp_date_obj:
                    status = "active"
                else:
                    status = "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1

        renewal_action_needed = 0
        if parsed_exp is not None and renewal_mech in AUTO_OR_OPTIONAL_RENEWAL:
            days_to_expiry = (date.fromisoformat(parsed_exp) - today).days
            if 0 <= days_to_expiry <= RENEWAL_ACTION_WINDOW_DAYS:
                renewal_action_needed = 1

        conn.execute(
            """UPDATE contracts_master SET
                   parsed_effective_date = ?, parsed_expiration_date = ?, computed_status = ?,
                   renewal_action_needed = ?, parsed_notice_period_days = ?, parsed_escalation_pct = ?
               WHERE filename = ?""",
            (parsed_eff, parsed_exp, status, renewal_action_needed,
             parse_notice_period_days(notice_raw), parse_escalation_pct(escalation_raw), filename),
        )

    conn.commit()
    return status_counts


def _compute_renewal_confidence(conn: sqlite3.Connection):
    """Filename-only heuristic (renewal_confidence.py) layered on top of
    computed_status -- does NOT overwrite it. For any contract_id that has a
    renewal_letter/extension in existence_only, classify the best-evidenced
    filename into high/moderate/low confidence that a renewal is currently
    in force, using only filename patterns (year-ranges, embedded dates) --
    no extraction from document content. Contract_ids with no renewal doc on
    file get renewal_confidence = NULL (Tier D, unaffected)."""
    today = date.today()

    renewal_docs = conn.execute(
        "SELECT contract_id, filename FROM existence_only WHERE doc_role IN (?, ?) AND contract_id != 'UNKNOWN'",
        tuple(RENEWAL_ROLES),
    ).fetchall()

    by_contract: dict[str, list[str]] = {}
    for contract_id, filename in renewal_docs:
        by_contract.setdefault(contract_id, []).append(filename)

    best = best_tier_per_contract(by_contract, today)

    doc_tier_counts: dict[str, int] = {}
    for contract_id, filenames in by_contract.items():
        for fn in filenames:
            tier, _ = classify_renewal_filename(fn, today)
            doc_tier_counts[tier] = doc_tier_counts.get(tier, 0) + 1

    for contract_id, (tier, basis, source_filename) in best.items():
        conn.execute(
            "UPDATE contracts_master SET renewal_confidence = ?, renewal_confidence_basis = ? WHERE contract_id = ?",
            (tier, f"{basis} (from {source_filename})", contract_id),
        )
    conn.commit()

    contract_tier_counts: dict[str, int] = {}
    for tier, _, _ in best.values():
        contract_tier_counts[tier] = contract_tier_counts.get(tier, 0) + 1

    upgraded = conn.execute("""
        SELECT contract_id, filename, renewal_confidence, renewal_confidence_basis
        FROM contracts_master
        WHERE computed_status = 'expired_but_renewal_on_file'
        ORDER BY contract_id
    """).fetchall()

    print(f"\nrenewal_confidence: {len(renewal_docs)} renewal_letter/extension docs "
          f"with known contract_id, across {len(by_contract)} distinct contract_ids")
    print(f"  per-document tier breakdown (all renewal/extension filenames): {doc_tier_counts}")
    print(f"  per-contract_id best tier (drives renewal_confidence column): {contract_tier_counts}")
    print(f"\n  expired_but_renewal_on_file rows ({len(upgraded)}) -- renewal_confidence detail:")
    for contract_id, filename, tier, basis in upgraded:
        print(f"    {contract_id:<8} {filename:<45} -> {tier:<8} {basis}")


def _compute_precise_renewal_status(conn: sqlite3.Connection):
    """Escalates the 'expired_but_renewal_on_file' bucket from a coarse tier
    (renewal_confidence) to a specific implied expiration date + a real
    still-active boolean, per renewal_confidence.precise_renewal_expiration().
    Does NOT touch computed_status -- same pattern as renewal_confidence
    itself, kept as separate, auditable columns.

    Renewal letters are matched to the SPECIFIC base agreement, not just its
    contract_id: a contract_id can have multiple sub-agreements (e.g.
    22125_1, 22125_2) each with their own renewal letters (e.g.
    "22125_1_Renewal_Letter__25_26.pdf" belongs to 22125_1, not 22125_2) --
    matching on contract_id alone would conflate them. Falls back to the
    full contract_id pool only when the base agreement's filename has no
    "{contract_id}_{n}" sub-agreement suffix to match against.

    Only computed for 'expired_but_renewal_on_file' rows -- that's the only
    bucket this column is meant to resolve; 'active'/'expired' rows don't
    need it, and existence-only (Tier 2/4) rows have no base expiration date
    to anchor the calculation to.
    """
    today = date.today()

    rows = conn.execute("""
        SELECT filename, contract_id, parsed_expiration_date
        FROM contracts_master
        WHERE computed_status = 'expired_but_renewal_on_file'
    """).fetchall()

    resolved = 0
    for filename, contract_id, parsed_exp_str in rows:
        m = re.match(rf"^{re.escape(contract_id)}_(\d+)", filename)
        suffix = f"{contract_id}_{m.group(1)}" if m else None

        all_letters = [r[0] for r in conn.execute(
            "SELECT filename FROM existence_only WHERE contract_id = ? AND doc_role IN (?, ?)",
            (contract_id, *RENEWAL_ROLES),
        ).fetchall()]
        matched = [f for f in all_letters if suffix in f] if suffix else all_letters

        base_exp = date.fromisoformat(parsed_exp_str)
        implied, still_active, source_filename = precise_renewal_expiration(base_exp, matched, today)

        conn.execute(
            "UPDATE contracts_master SET renewal_precise_expiration = ?, renewal_precise_still_active = ? WHERE filename = ?",
            (implied.isoformat() if implied else None, int(still_active) if still_active is not None else None, filename),
        )
        if implied:
            resolved += 1
            print(f"    precise: {contract_id:<8} {filename:<45} -> {implied} (still_active={still_active}, from {source_filename})")

    conn.commit()
    print(f"\nrenewal_precise_expiration: resolved {resolved} of {len(rows)} 'expired_but_renewal_on_file' rows "
          f"to a specific implied date (rest have no 'high'-tier renewal letter to anchor a date to)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="contracts.db")
    args = parser.parse_args()
    conn = sqlite3.connect(args.db)
    build(conn)
    conn.close()


if __name__ == "__main__":
    main()
