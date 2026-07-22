"""
Filename-only heuristic for renewal recency. Classifies renewal_letter/
extension filenames (from `existence_only`) into a confidence tier for
whether a contract's renewal is likely to still cover today, WITHOUT
extracting anything from the documents themselves -- pure regex over
filenames already in the database.

Spot-check finding (6 renewal letters opened directly, see contract_ids
21003, 18018, 22125, 22128, 22162): Lake County's renewal periods are
anchored to each contract's own anniversary date (observed start months:
Feb, Mar, Aug, Oct), NOT a uniform Nov-30 fiscal year end as originally
hypothesized. What IS consistent across every sample: the two calendar
years named in the filename (e.g. "25_26", "2020_2021") always match the
two calendar years the real coverage period spans. So TIER A trusts only
that weaker, verified claim -- "today's year falls within [first_year,
second_year]" -- not a specific month/day boundary within those years.

Tiers:
  high     -- filename has an explicit year-range covering today's year.
  moderate -- filename has an embedded YYYY-MM-DD letter date, recent
              (<= MODERATE_WINDOW_DAYS old), no explicit range.
  low      -- filename parses (range or date) but doesn't support "current",
              or renewal exists but the filename matches no known pattern.
              (This is the prior flat 'expired_but_renewal_on_file' behavior,
              kept as the honest fallback.)
"""
import re
from datetime import date

MODERATE_WINDOW_DAYS = 548  # ~18 months, per task's "roughly 12-18 months"

# 4-digit range: "2020_2021", "2024-2025"
_YEAR_RANGE_4 = re.compile(r"(?<!\d)(20\d{2})[_-](20\d{2})(?!\d)")
# 2-digit range: "25_26", "23-24". Excludes a 2-digit pair immediately preceded
# by a 4-digit-year+separator (e.g. the "09_10" in "2024_09_10_Renewal...") --
# without this, a YYYY_MM_DD date's month/day can coincidentally be
# consecutive numbers and get misread as a fiscal-year range (found while
# building precise_renewal_expiration(): "2024_09_10_..." -> bogus "2009-2010").
_YEAR_RANGE_2 = re.compile(r"(?<!\d{4}[_-])(?<!\d)(\d{2})[_-](\d{2})(?!\d)")
# mixed range: "2022_23" (seen in corpus -- 4-digit start, 2-digit end)
_YEAR_RANGE_MIXED = re.compile(r"(?<!\d)(20\d{2})[_-](\d{2})(?!\d)")
# embedded date: "2024_08_08", "2025-09-23"
_DATE_YMD = re.compile(r"(?<!\d)(20\d{2})[_-](\d{1,2})[_-](\d{1,2})(?!\d)")


def _expand_2digit_year(yy: str) -> int:
    return 2000 + int(yy)


def _valid_ranges(name: str) -> list[tuple[int, int, int]]:
    """Returns (start_index, year1, year2) for every consecutive-year range
    match in `name`, across all three range patterns. 'Consecutive' (year2 ==
    year1 + 1) is the filter that keeps this from matching unrelated digit
    pairs (e.g. a month/day pair in an embedded date)."""
    found = []
    for m in _YEAR_RANGE_4.finditer(name):
        y1, y2 = int(m.group(1)), int(m.group(2))
        if y2 == y1 + 1:
            found.append((m.start(), y1, y2))
    for m in _YEAR_RANGE_MIXED.finditer(name):
        y1 = int(m.group(1))
        y2 = _expand_2digit_year(m.group(2))
        if y2 == y1 + 1:
            found.append((m.start(), y1, y2))
    for m in _YEAR_RANGE_2.finditer(name):
        y1, y2 = _expand_2digit_year(m.group(1)), _expand_2digit_year(m.group(2))
        if y2 == y1 + 1:
            found.append((m.start(), y1, y2))
    return found


def classify_renewal_filename(filename: str, today: date | None = None) -> tuple[str, str]:
    """Returns (tier, basis) where tier is 'high' | 'moderate' | 'low'."""
    if today is None:
        today = date.today()
    name = filename.rsplit(".", 1)[0]

    ranges = _valid_ranges(name)
    if ranges:
        # Real letters place the authoritative range at the end of the
        # filename (after "Renewal_Letter"/"Extension"); a leading date
        # prefix, if any, is the send date, not the coverage period. Take
        # the rightmost match.
        _, y1, y2 = max(ranges, key=lambda t: t[0])
        covers = y1 <= today.year <= y2
        basis = f"year-range {y1}-{y2} parsed from filename"
        if covers:
            return "high", f"{basis}; covers today ({today.isoformat()})"
        return "low", f"{basis}; does not cover today ({today.isoformat()}) -- stale, not assumed current"

    for m in _DATE_YMD.finditer(name):
        y, mo, d = m.groups()
        try:
            letter_date = date(int(y), int(mo), int(d))
        except ValueError:
            continue
        days_old = (today - letter_date).days
        if 0 <= days_old <= MODERATE_WINDOW_DAYS:
            return "moderate", (
                f"letter dated {letter_date.isoformat()} ({days_old} days old, "
                f"within {MODERATE_WINDOW_DAYS}-day recency window), no explicit coverage range"
            )
        return "low", f"letter dated {letter_date.isoformat()} -- outside recency window, not assumed current"

    return "low", "filename matches no known year-range or date pattern"


TIER_RANK = {"high": 2, "moderate": 1, "low": 0}


def precise_renewal_expiration(base_exp: date, renewal_filenames: list[str],
                                today: date | None = None) -> tuple[date | None, bool | None, str | None]:
    """Computes a specific implied renewal-expiration date, not just a
    high/moderate/low tier. Rule: among renewal_filenames classified 'high'
    (an explicit year-range covering today's year), take the one with the
    LATEST year_end -- a newer renewal supersedes an older one on file, e.g.
    a '26_27' letter supersedes an earlier '25_26' letter for the same
    sub-agreement -- and assume the renewed term ends on the same month/day
    as the base agreement's own expiration, just in year_end. This is a
    real, deliberate escalation past classify_renewal_filename()'s coarser
    "today's year falls in [y1,y2]" check: two 'high'-tier contracts can
    still differ on whether they're ACTUALLY still active today once the
    specific month/day is applied (e.g. a 'high' tier from a year-range that
    covers today's year, but whose implied month/day anniversary already
    passed earlier this year).

    Returns (implied_expiration, still_active, source_filename), or
    (None, None, None) if no 'high'-tier renewal filename is found -- callers
    should NOT compute a date for moderate/low/no-renewal cases; the coarser
    tier is all that's supportable there.
    """
    if today is None:
        today = date.today()

    high_candidates = []
    for fn in renewal_filenames:
        tier, _ = classify_renewal_filename(fn, today)
        if tier != "high":
            continue
        ranges = _valid_ranges(fn.rsplit(".", 1)[0])
        if not ranges:
            continue
        _, y1, y2 = max(ranges, key=lambda t: t[0])
        high_candidates.append((y2, fn))

    if not high_candidates:
        return None, None, None

    year_end, source_filename = max(high_candidates, key=lambda t: t[0])
    try:
        implied = date(year_end, base_exp.month, base_exp.day)
    except ValueError:
        # e.g. base expiration is Feb 29 and year_end isn't a leap year --
        # fall back one day rather than silently dropping the computation.
        implied = date(year_end, base_exp.month, base_exp.day - 1)
    return implied, implied > today, source_filename


def best_tier_per_contract(renewal_filenames_by_contract: dict[str, list[str]], today: date | None = None) -> dict[str, tuple[str, str, str]]:
    """renewal_filenames_by_contract: {contract_id: [filenames]}.
    Returns {contract_id: (best_tier, basis, source_filename)} -- the
    highest-confidence classification among that contract's renewal/
    extension documents."""
    result = {}
    for contract_id, filenames in renewal_filenames_by_contract.items():
        best = None
        for fn in filenames:
            tier, basis = classify_renewal_filename(fn, today)
            if best is None or TIER_RANK[tier] > TIER_RANK[best[0]]:
                best = (tier, basis, fn)
        if best is not None:
            result[contract_id] = best
    return result
