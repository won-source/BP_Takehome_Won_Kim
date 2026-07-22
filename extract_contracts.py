"""
Contract extraction pipeline -- Tier 1 (base_agreement, sow) + Tier 1.5
(modification, amendment), the two document types that carry the full
19-field schema (see tiered_extraction_approach.md for why the corpus is
split this way; Tiers 2/4, including the descoped rate_adjustment role,
are handled by build_existence_table.py).

Reads tier1_combined_full_extraction.csv, extracts text from each PDF, calls
Claude Haiku with the extract_contract_fields tool, and writes results into
a SQLite database with one row per document plus per-field page citations.
Scanned documents (where pypdf finds ~no text layer) are rasterized to page
images and sent to Claude directly as vision input, in the same request as
the tool call -- no separate OCR step.

IMPORTANT: test on 3-5 documents first (see --limit flag) before running the
full sample. Check the extraction quality and JSON parsing before spending
the full API budget.

Usage:
    pip install anthropic pypdf pymupdf

    export ANTHROPIC_API_KEY=...

    python extract_contracts.py --folder /path/to/contracts --sample-csv tier1_combined_full_extraction.csv --limit 5
    python extract_contracts.py --folder /path/to/contracts --sample-csv tier1_combined_full_extraction.csv   # full run

Output:
    contracts.db (SQLite) -- table `extractions`, one row per document
"""

import argparse
import base64
import csv
import json
import os
import sqlite3
import sys
import time
from pathlib import Path


def _load_dotenv():
    """Minimal .env loader (no python-dotenv dependency): sets any KEY=value
    pairs found in .env alongside this script, without overriding vars
    already present in the real environment."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

try:
    from pypdf import PdfReader
except ImportError:
    print("Run: pip install pypdf")
    sys.exit(1)

try:
    import fitz  # PyMuPDF -- pure Python wheel, no system binary dependency
except ImportError:
    print("Run: pip install pymupdf")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("Run: pip install anthropic")
    sys.exit(1)


MODEL = "claude-haiku-4-5"  # supports image input + tool use in the same request
MAX_VISION_PAGES = 35  # cap per document; docs beyond this range are the ~50+ page
# scanned outliers that were moved to Tier 2 existence-only (see
# tiered_extraction_approach.md), so this mainly guards against any doc that
# wasn't caught by that triage rather than routinely truncating normal-length
# scanned contracts (e.g. a 28-page doc used to be truncated at 20, no longer is)
MAX_IMAGE_DIMENSION = 1568  # px on the long edge -- Anthropic's cost/quality sweet spot
INPUT_COST_PER_MTOK = 1.0  # Haiku 4.5 standard pricing, https://platform.claude.com/docs/en/about-claude/pricing
OUTPUT_COST_PER_MTOK = 5.0
SCHEMA_PATH = Path(__file__).parent / "extraction_schema.json"
PROMPT_PATH = Path(__file__).parent / "extraction_system_prompt.txt"

# flat list of the 19 substantive fields the tool schema returns.
# initial_term, liability_cap, and governing_law were removed: liability_cap
# and governing_law showed zero variance across all labeled ground truth
# (every doc is one county under one state's law), and initial_term was
# redundant with expiration_date. modification_summary is the odd one out:
# on base_agreement/sow docs its value is the sentinel
# "not_applicable_doc_type" (treated like not_stated for citation purposes),
# and it carries an extra modification_type sub-field the other 18 fields
# don't have -- handled specially in main().
FIELDS = [
    "counterparty_name", "org_role", "effective_date", "expiration_date",
    "renewal_mechanism", "contract_value", "fee_structure",
    "payment_terms", "annual_price_escalation_type", "annual_price_escalation_detail",
    "termination_for_convenience", "notice_period", "assignment_consent_required",
    "indemnity_present", "insurance_required",
    "sole_source_vs_competitive_bid", "signer_title", "fiscal_year_appropriation_contingent",
    "modification_summary",
]

# Sentinel values, in addition to "not_stated"/"not_stated - ...", that
# legitimately have no source_page.
NULL_PAGE_OK_VALUES = {"not_applicable_doc_type"}


def _build_text_content(text: str, note: str = "") -> str:
    content = f"Extract the fields from this contract document:\n\n{text}"
    return f"{content}\n\n{note}" if note else content


def _build_vision_content(page_images: list[bytes], truncated: bool, note: str = "") -> list[dict]:
    """Builds multimodal message content: an intro, then each page as a
    "[PAGE N]" text marker immediately followed by its image -- preserving
    the same page-citation convention the system prompt already teaches for
    text input, so no prompt changes were needed for the vision path."""
    intro = (
        "Extract the fields from this contract document. It is a scanned "
        "document; each page is provided below as an image, in order, "
        "starting at page 1."
    )
    if truncated:
        intro += (
            f" NOTE: this document has more than {len(page_images)} pages; only the "
            f"first {len(page_images)} are included below. If a field's answer might "
            f"be on a later, omitted page, mark it not_stated rather than guessing."
        )
    content = [{"type": "text", "text": intro}]
    for i, img_bytes in enumerate(page_images):
        content.append({"type": "text", "text": f"[PAGE {i + 1}]"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(img_bytes).decode(),
            },
        })
    if note:
        content.append({"type": "text", "text": note})
    return content


def _rasterize_pages(pdf_path: Path, max_pages: int, max_dimension: int) -> tuple[list[bytes], bool]:
    """Returns (page_images_png, truncated), downsized so the long edge is
    at most max_dimension px -- large enough to stay legible, small enough
    to keep vision token cost in the same ballpark as the text path."""
    doc = fitz.open(str(pdf_path))
    try:
        num_pages = doc.page_count
        page_images = []
        for i in range(min(num_pages, max_pages)):
            page = doc[i]
            rect = page.rect
            scale = max_dimension / max(rect.width, rect.height)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            page_images.append(pix.tobytes("png"))
        return page_images, num_pages > max_pages
    finally:
        doc.close()


def load_document_content(pdf_path: Path):
    """Returns (content_fn, used_vision, page_truncated).

    content_fn(note="") builds the Claude message content for this
    document -- either the extracted text (digital-text documents), or, for
    scanned documents where pypdf finds ~no text layer, the rasterized page
    images sent directly as vision input. `note` lets call_extraction append
    a correction note on retry without re-deriving the underlying content.
    """
    reader = PdfReader(str(pdf_path))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    # Check the real body text alone -- "[PAGE N]" markers on an otherwise-blank
    # multi-page scan can add up to >100 chars by themselves and falsely look
    # like a digital-text document.
    body_only = "".join(pages_text)

    if len(body_only.strip()) > 100:
        full_text = "\n\n".join(f"[PAGE {i + 1}]\n{t}" for i, t in enumerate(pages_text))
        truncated_text = full_text[:60000]
        content_fn = lambda note="": _build_text_content(truncated_text, note)
        return content_fn, False, False

    page_images, truncated = _rasterize_pages(pdf_path, MAX_VISION_PAGES, MAX_IMAGE_DIMENSION)
    content_fn = lambda note="": _build_vision_content(page_images, truncated, note)
    return content_fn, True, truncated


def _field_value_str(field_data) -> str:
    """Extracts the value regardless of whether the model returned the
    correct {value, source_page} object or (occasionally) collapsed a
    field to a bare scalar."""
    if isinstance(field_data, dict):
        return str(field_data.get("value", ""))
    return "" if field_data is None else str(field_data)


def _fields_missing_page(result: dict, fields: list[str] = FIELDS) -> list[str]:
    """Fields with a real (non-not_stated) value but no source_page -- these
    can't be audited against the source document and indicate the model
    either skipped the citation step or (rarer) returned the field as a
    bare scalar instead of the required {value, source_page} object.
    `fields` defaults to the main 21-field schema but is overridable so the
    same retry logic works for the narrower rate-adjustment schema too."""
    missing = []
    for f in fields:
        field_data = result.get(f)
        value = _field_value_str(field_data)
        if not value or value.startswith("not_stated") or value in NULL_PAGE_OK_VALUES:
            continue
        source_page = field_data.get("source_page") if isinstance(field_data, dict) else None
        if source_page is None:
            missing.append(f)
    return missing


MODIFICATION_TYPE_ENUM = {
    "rate_increase", "rate_schedule_expansion", "term_extension", "scope_change",
    "vendor_or_party_change", "assignment", "termination", "other", "not_applicable",
}


def _modification_summary_issue(result: dict) -> str | None:
    """modification_summary has a shape distinct from every other field (a
    prose `value` plus a separate `modification_type` category), and that
    extra structure has its own failure mode _fields_missing_page can't see:
    observed in practice, a retry pass that correctly fixed another field's
    citation swapped modification_summary's value for a bare category label
    and dropped modification_type entirely. Returns a description of the
    problem, or None if the field looks fine."""
    field_data = result.get("modification_summary")
    if not isinstance(field_data, dict):
        # Bare-scalar collapse: normally caught by _fields_missing_page, but that
        # function exempts "not_applicable_doc_type" via NULL_PAGE_OK_VALUES (it's
        # a legitimate null-page sentinel), which also means it never notices the
        # value arrived as a bare string instead of the required {value,
        # modification_type, source_page} object -- so modification_type silently
        # never gets set. Catch that collapse here instead.
        if str(field_data) == "not_applicable_doc_type":
            return "modification_summary collapsed to a bare string 'not_applicable_doc_type' instead of the required object shape; modification_type is missing"
        return None
    value = str(field_data.get("value", ""))
    mod_type = field_data.get("modification_type")
    if value == "not_applicable_doc_type":
        if mod_type != "not_applicable":
            return f"value is 'not_applicable_doc_type' but modification_type is {mod_type!r}, expected 'not_applicable'"
        return None
    if not value or value.startswith("not_stated"):
        return None
    if mod_type not in MODIFICATION_TYPE_ENUM:
        return f"modification_type is {mod_type!r}, not a valid category"
    if value in MODIFICATION_TYPE_ENUM:
        return f"value {value!r} looks like a category label, not a prose summary"
    return None


def _call_tool_once(client, schema: dict, system_prompt: str, user_content: str) -> tuple[dict, dict]:
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=system_prompt,
        tools=[{
            "name": schema["name"],
            "description": schema["description"],
            "input_schema": schema["input_schema"],
        }],
        tool_choice={"type": "tool", "name": schema["name"]},
        messages=[{"role": "user", "content": user_content}],
    )
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    for block in response.content:
        if block.type == "tool_use":
            return block.input, usage
    raise ValueError("No tool_use block in response")


def call_extraction(client, schema: dict, system_prompt: str, content_fn, fields: list[str] = FIELDS) -> tuple[dict, bool, list[dict]]:
    """Returns (result, needed_retry, usage_per_call). content_fn(note="")
    builds the message content (text or vision) for this document. Runs one
    automatic retry, with the specific gaps called out via `note`, if the
    first pass left real values uncited -- a null source_page on a real
    value is unauditable and worse than a slower response. Works identically
    for the text and vision paths since content_fn hides the difference.
    `fields` is overridable so this same retry logic serves both the main
    21-field schema and the narrower rate-adjustment schema.

    The retry result is merged into the original field-by-field, touching
    only the fields that actually needed fixing. Observed in practice: a
    wholesale replacement let a retry pass that correctly fixed one field's
    citation silently corrupt an unrelated field (modification_summary) that
    was already fine -- the model's attention was on the reported gaps, not
    on faithfully reproducing everything else."""
    result, usage = _call_tool_once(client, schema, system_prompt, content_fn())

    missing = _fields_missing_page(result, fields)
    mod_issue = _modification_summary_issue(result) if "modification_summary" in fields else None
    needs_fix = list(missing) + (["modification_summary"] if mod_issue and "modification_summary" not in missing else [])
    if not needs_fix:
        return result, False, [usage]

    gap_lines = [f"- {f}: value={_field_value_str(result.get(f))!r}, source_page=null" for f in missing]
    if mod_issue and "modification_summary" not in missing:
        gap_lines.append(f"- modification_summary: {mod_issue}")
    gaps = "\n".join(gap_lines)
    note = (
        f"CORRECTION NEEDED FROM A PRIOR PASS on these specific fields:\n{gaps}\n\n"
        f"For missing source_page: re-examine the document above (page markers are "
        f"\"[PAGE N]\") and find the page number. If, on closer inspection, a value "
        f"truly cannot be verified, change it to \"not_stated\" instead -- but every "
        f"field must end up with either a real source_page or a not_stated value, never "
        f"a real value with a null page. Every field must be the object shape "
        f"{{\"value\": ..., \"source_page\": ...}} -- never a bare string.\n"
        f"For modification_summary specifically: `value` must be a one-to-two sentence "
        f"prose description of what changed, never a bare category label -- "
        f"`modification_type` is the separate field for that category.\n\n"
        f"Only these listed fields were wrong -- return your best answer for every field "
        f"in the schema as usual, but take particular care not to change anything about "
        f"the fields NOT listed above, which were already correct."
    )
    retried_result, retry_usage = _call_tool_once(client, schema, system_prompt, content_fn(note))

    # Merge: only the fields flagged as needing a fix take the retry's answer;
    # everything else keeps the original pass's (already-correct) value.
    merged_result = dict(result)
    for f in needs_fix:
        if f in retried_result:
            merged_result[f] = retried_result[f]

    return merged_result, True, [usage, retry_usage]


def init_db(db_path: str):
    conn = sqlite3.connect(db_path)
    cols_sql = ", ".join(f'"{f}_value" TEXT, "{f}_page" INTEGER' for f in FIELDS)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS extractions (
            filename TEXT PRIMARY KEY,
            contract_id TEXT,
            doc_role TEXT,
            family_size INTEGER,
            used_vision INTEGER,
            page_truncated INTEGER,
            page_citation_retry INTEGER,
            fields_missing_page TEXT,
            modification_summary_issue TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            extraction_status TEXT,
            error_message TEXT,
            modification_type TEXT,
            {cols_sql}
        )
    """)
    conn.commit()
    return conn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--sample-csv", required=True)
    parser.add_argument("--db", default="contracts.db")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N docs, for testing")
    args = parser.parse_args()

    schema = json.loads(SCHEMA_PATH.read_text())
    system_prompt = PROMPT_PATH.read_text()
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    conn = init_db(args.db)

    with open(args.sample_csv) as f:
        sample = list(csv.DictReader(f))
    if args.limit:
        sample = sample[:args.limit]

    print(f"Processing {len(sample)} documents...")
    total_input_tokens = 0
    total_output_tokens = 0

    for i, row in enumerate(sample):
        fname = row["filename"]
        pdf_path = Path(args.folder) / fname
        print(f"[{i+1}/{len(sample)}] {fname}")

        if not pdf_path.exists():
            conn.execute(
                "INSERT OR REPLACE INTO extractions (filename, contract_id, doc_role, family_size, extraction_status, error_message) VALUES (?, ?, ?, ?, ?, ?)",
                (fname, row.get("contract_id"), row.get("doc_role"), row.get("family_size"), "error", "file not found on disk")
            )
            conn.commit()
            continue

        try:
            content_fn, used_vision, page_truncated = load_document_content(pdf_path)
            if used_vision:
                print(f"  (scanned document -- using vision extraction{', truncated' if page_truncated else ''})")

            result, retried, usage_per_call = call_extraction(client, schema, system_prompt, content_fn)
            still_missing = _fields_missing_page(result)
            if still_missing:
                print(f"  WARNING: {len(still_missing)} field(s) still missing source_page after retry: {still_missing}")
            mod_issue = _modification_summary_issue(result)
            if mod_issue:
                print(f"  WARNING: modification_summary still malformed after retry: {mod_issue}")

            doc_input_tokens = sum(u["input_tokens"] for u in usage_per_call)
            doc_output_tokens = sum(u["output_tokens"] for u in usage_per_call)
            doc_cost = doc_input_tokens / 1e6 * INPUT_COST_PER_MTOK + doc_output_tokens / 1e6 * OUTPUT_COST_PER_MTOK
            total_input_tokens += doc_input_tokens
            total_output_tokens += doc_output_tokens
            print(f"  tokens: {doc_input_tokens} in / {doc_output_tokens} out (${doc_cost:.4f})")

            values = {}
            for f in FIELDS:
                field_data = result.get(f, {})
                values[f"{f}_value"] = field_data.get("value", "not_stated") if isinstance(field_data, dict) else str(field_data)
                values[f"{f}_page"] = field_data.get("source_page") if isinstance(field_data, dict) else None
            mod_field = result.get("modification_summary", {})
            modification_type = mod_field.get("modification_type") if isinstance(mod_field, dict) else None

            col_names = ["filename", "contract_id", "doc_role", "family_size", "used_vision", "page_truncated",
                         "page_citation_retry", "fields_missing_page", "modification_summary_issue",
                         "input_tokens", "output_tokens", "extraction_status", "error_message", "modification_type"]
            col_names += [f"{f}_value" for f in FIELDS] + [f"{f}_page" for f in FIELDS]
            placeholders = ", ".join("?" for _ in col_names)
            col_list = ", ".join(f'"{c}"' for c in col_names)
            row_values = [fname, row.get("contract_id"), row.get("doc_role"), row.get("family_size"),
                          int(used_vision), int(page_truncated), int(retried), ",".join(still_missing) or None,
                          mod_issue, doc_input_tokens, doc_output_tokens, "success", None, modification_type]
            row_values += [values[f"{f}_value"] for f in FIELDS] + [values[f"{f}_page"] for f in FIELDS]

            conn.execute(f"INSERT OR REPLACE INTO extractions ({col_list}) VALUES ({placeholders})", row_values)
            conn.commit()

        except Exception as e:
            print(f"  ERROR: {e}")
            conn.execute(
                "INSERT OR REPLACE INTO extractions (filename, contract_id, doc_role, family_size, extraction_status, error_message) VALUES (?, ?, ?, ?, ?, ?)",
                (fname, row.get("contract_id"), row.get("doc_role"), row.get("family_size"), "error", str(e))
            )
            conn.commit()

        time.sleep(0.5)  # light rate-limit courtesy

    conn.close()
    total_cost = total_input_tokens / 1e6 * INPUT_COST_PER_MTOK + total_output_tokens / 1e6 * OUTPUT_COST_PER_MTOK
    print(f"\nDone. Results in {args.db}")
    print(f"Total: {total_input_tokens} input / {total_output_tokens} output tokens, ${total_cost:.4f}")
    if sample:
        per_doc = total_cost / len(sample)
        print(f"Avg ${per_doc:.4f}/doc -- extrapolated to 100 docs: ${per_doc * 100:.2f}")


if __name__ == "__main__":
    main()
