"""
Build the existence-only table for Tier 2 (renewal_letter, extension, lease) and
Tier 4 (vendor_disclosure, award_notice) documents.

These documents don't get field extraction -- the hand-labeling pass showed they carry
almost no real content on the 20-field schema (3-6 of 20 fields populated vs 18-20 for
base agreements/modifications). Instead, this script derives a lightweight existence
record for each from data already computed during family grouping: does this document
exist, which contract family does it belong to, and (for Tier 2 docs specifically,
where the filename/text often contains a usable date) a best-effort date if the family
grouping pass happened to catch one.

This produces zero incremental API cost and requires no LLM call at all -- it's a
straight transform of family_grouped_files.csv into a database table.

Usage:
    python build_existence_table.py --tier2 tier2_existence_only.csv --tier4 tier4_existence_only.csv --db contracts.db
"""

import argparse
import csv
import sqlite3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier2", required=True)
    parser.add_argument("--tier4", required=True)
    parser.add_argument("--db", default="contracts.db")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS existence_only (
            filename TEXT PRIMARY KEY,
            contract_id TEXT,
            doc_role TEXT,
            tier INTEGER,
            family_size INTEGER,
            file_size_kb REAL
        )
    """)

    total = 0
    for tier_num, path in [(2, args.tier2), (4, args.tier4)]:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            conn.execute(
                "INSERT OR REPLACE INTO existence_only (filename, contract_id, doc_role, tier, family_size, file_size_kb) VALUES (?, ?, ?, ?, ?, ?)",
                (r["filename"], r["contract_id"], r["doc_role"], tier_num, r.get("family_size"), r.get("file_size_kb"))
            )
        print(f"Tier {tier_num}: {len(rows)} docs added from {path}")
        total += len(rows)

    conn.commit()

    # Summary: renewal/extension/lease existence rolled up per contract family --
    # this is the "has this contract been renewed" signal for the renewal calendar,
    # derived for free from what's already in the table.
    cur = conn.execute("""
        SELECT contract_id, GROUP_CONCAT(DISTINCT doc_role) as roles_present, COUNT(*) as doc_count
        FROM existence_only
        WHERE tier = 2 AND contract_id != 'UNKNOWN'
        GROUP BY contract_id
        ORDER BY doc_count DESC
        LIMIT 10
    """)
    print("\nTop 10 contract families by Tier 2 (renewal/extension/lease) document count:")
    for row in cur.fetchall():
        print(f"  {row[0]:<10} {row[1]:<40} {row[2]} docs")

    conn.close()
    print(f"\nDone. {total} existence-only records written to {args.db}, table `existence_only`.")


if __name__ == "__main__":
    main()
