"""
Shared date-parsing utilities for prose contract fields (effective_date,
expiration_date, notice_period, annual_price_escalation_detail). Single
source of truth so compare_ground_truth.py's fuzzy-date matching and
build_master_table.py's computed date columns don't drift out of sync.
"""
import re

from dateutil.relativedelta import relativedelta

MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04", "may": "05",
    "june": "06", "july": "07", "august": "08", "september": "09",
    "october": "10", "november": "11", "december": "12",
}

WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

DURATION_PATTERN = re.compile(r"\b(\d+)\s*[- ]?\s*(year|month|week|day)s?\b", re.IGNORECASE)
# Legal prose commonly spells out the number and repeats it in parens, e.g.
# "one (1) year period" or just "one year" with no digit at all -- match the
# word form directly rather than relying on the parenthetical digit being
# adjacent to the unit (it usually isn't: "two (2) year" has ") " between
# the digit and "year", so DURATION_PATTERN's \d+ doesn't reach it).
WORD_DURATION_PATTERN = re.compile(
    r"\b(" + "|".join(WORD_NUMBERS) + r")\b(?:\s*\(\d+\))?\s*[- ]?\s*(year|month|week|day)s?\b",
    re.IGNORECASE,
)
NOTICE_DAYS_PATTERN = re.compile(r"(\d+)\s*day", re.IGNORECASE)
PERCENT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _expand_2digit_year(yy: str) -> int:
    """2-digit years in this corpus are all modern contracts (observed range
    ~2015-2027) -- always expand to 20XX, never 19XX."""
    return 2000 + int(yy)


def normalize_date(s) -> str | None:
    """Best-effort: pull an absolute date out of prose, render as YYYY-MM-DD.
    Handles YYYY-MM-DD, "Month DD, YYYY", "DD Month YYYY", M/D/YYYY or M/D/YY
    (slash), and M-D-YYYY or M-D-YY (dash)."""
    if not s:
        return None
    s = str(s).lower()
    # \d{1,2} not \d{2} -- ground truth cells write unpadded dates like
    # "2018-6-15" (single-digit month), which the old strict \d{2} pattern
    # silently failed to match at all (fell through to None), even though
    # it's the same YYYY-MM-DD format, just unpadded. Re-render zero-padded
    # so it compares equal to a padded date from either side.
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})\b", s)
    if m:
        year, month, day = m.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    m = re.search(r"(" + "|".join(MONTHS) + r")\s+(\d{1,2}),?\s+(\d{4})", s)
    if m:
        mon, day, year = m.groups()
        return f"{year}-{MONTHS[mon]}-{int(day):02d}"
    m = re.search(r"\b(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(\d{4})", s)
    if m:
        day, mon, year = m.groups()
        return f"{year}-{MONTHS[mon]}-{int(day):02d}"
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", s)
    if m:
        month, day, year = m.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2})\b", s)
    if m:
        month, day, yy = m.groups()
        return f"{_expand_2digit_year(yy)}-{int(month):02d}-{int(day):02d}"
    return None


def parse_duration(s) -> relativedelta | None:
    """Best-effort: find the first "N year(s)/month(s)/week(s)/day(s)" in
    text (digit or spelled-out word form -- legal prose favors "one year"
    over "1 year") and return it as a relativedelta, or None if nothing found."""
    if not s:
        return None
    s = str(s)
    m = DURATION_PATTERN.search(s)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        return relativedelta(**{f"{unit}s": n})
    m = WORD_DURATION_PATTERN.search(s)
    if m:
        n, unit = WORD_NUMBERS[m.group(1).lower()], m.group(2).lower()
        return relativedelta(**{f"{unit}s": n})
    return None


def parse_notice_period_days(s) -> int | None:
    """Extract an integer day count from notice_period text (almost always "N days")."""
    if not s:
        return None
    m = NOTICE_DAYS_PATTERN.search(str(s))
    return int(m.group(1)) if m else None


def parse_escalation_pct(s) -> float | None:
    """Extract the first percentage figure from escalation-detail text."""
    if not s:
        return None
    m = PERCENT_PATTERN.search(str(s))
    return float(m.group(1)) if m else None
