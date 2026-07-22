"""
Builds the clause-retrieval vector store for the 12 Tier 1/1.5 pilot
documents (base_agreement, sow, modification, amendment) -- the documents
with real negotiated clause content worth retrieving. Tier 2/4
existence-only documents are not chunked/embedded; there's nothing
substantive in them.

Chunking strategy:
- Digital-text docs: chunk by "SECTION N. TITLE" headers when >= 3 are
  cleanly detected (these Lake County contracts consistently use numbered
  section headers). Falls back to page-based chunking otherwise -- some
  short modification/amendment letters don't use this convention at all.
- Scanned docs (used_vision=1 in `extractions`): there is no persisted text
  to chunk, since the extraction pipeline sends page images straight to
  Claude for structured-field extraction and never produces a transcript.
  A one-time vision transcription pass (same rasterization as the
  extraction pipeline, a plain-text request instead of the tool-use
  schema) produces page-marked text, which then goes through the same
  section-detection/fallback logic as digital docs.
- The 6 modification/amendment docs also get one extra chunk each, holding
  their already-extracted modification_summary, tagged
  chunk_type="modification_summary" (vs "raw_clause" for the rest) -- the
  pre-digested "what changed" answer as a directly retrievable option,
  distinct from the raw clause text.

Embeddings: Voyage (voyage-3.5, 1024-dim), per the original architecture
plan. Storage: ChromaDB, persisted to ./chroma_db, collection
"contract_clauses". Every chunk carries filename/contract_id/doc_role/
section_number/section_title/page_number/chunk_type metadata so a
retrieved chunk can be cited back to its source, the same provenance
discipline as the structured extraction fields.

Usage:
    python build_vector_store.py --folder /path/to/contracts --db contracts.db
"""

import argparse
import re
import sqlite3
import time
from pathlib import Path

import anthropic
import chromadb
import httpx
import voyageai

from extract_contracts import (
    MAX_IMAGE_DIMENSION,
    MAX_VISION_PAGES,
    _rasterize_pages,
    _build_vision_content,
)

EMBED_MODEL = "voyage-3.5"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "contract_clauses"

SECTION_PATTERN = re.compile(r"^\s*SECTION\s+(\d+)\.?\s*([A-Z][A-Za-z0-9 /&,'-]*)", re.MULTILINE)
PAGE_MARKER_PATTERN = re.compile(r"\[PAGE (\d+)\]")

MODIFICATION_DOC_ROLES = {"modification", "amendment"}


RETRYABLE_EXCEPTIONS = (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError, anthropic.APIConnectionError)


def _stream_with_retry(client: anthropic.Anthropic, content: list[dict], max_retries: int = 3) -> tuple[list[str], str]:
    """Transient network errors (seen in practice: a connection forcibly closed
    mid-stream on a 123-doc run, killing the whole batch since this script didn't
    persist anything until the very end) shouldn't take down the entire build.
    Retries with backoff; only on network-level errors, not content/API errors."""
    for attempt in range(1, max_retries + 1):
        try:
            text_parts = []
            with client.messages.stream(
                model="claude-haiku-4-5", max_tokens=64000, messages=[{"role": "user", "content": content}]
            ) as stream:
                for chunk in stream.text_stream:
                    text_parts.append(chunk)
                final = stream.get_final_message()
            return text_parts, final.stop_reason
        except RETRYABLE_EXCEPTIONS as e:
            if attempt == max_retries:
                raise
            wait = 2 ** attempt
            print(f"    transient network error ({e!r}), retry {attempt}/{max_retries} in {wait}s...")
            time.sleep(wait)


def transcribe_scanned_doc(client: anthropic.Anthropic, pdf_path: Path) -> str:
    """One-time plain-text vision transcription for a scanned doc, using the
    same rasterization as the extraction pipeline but asking for verbatim
    page text instead of structured fields."""
    images, truncated = _rasterize_pages(pdf_path, MAX_VISION_PAGES, MAX_IMAGE_DIMENSION)
    content = _build_vision_content(images, truncated)
    content[0] = {
        "type": "text",
        "text": (
            'Transcribe the full text of each page below verbatim, preserving paragraph '
            'and section structure as plain text. Output format: a "[PAGE N]" marker '
            "immediately followed by that page's transcribed text, for every page, in "
            "order. Do not summarize, skip, or add commentary -- verbatim transcription only."
        ),
    }
    # A fixed max_tokens silently truncated verbatim transcriptions of longer scanned
    # docs (confirmed: stop_reason == "max_tokens" on a 28-page document at both 4000
    # and briefly 16000, losing pages with no error). Checked the real corpus: 5+ of
    # the 9 scanned Tier 1/1.5 docs over 20 pages need 16-22k+ tokens at the observed
    # ~550-600 tokens/page rate, which a non-streaming call can't request at all above
    # ~24000 (the API requires streaming past that). Streaming removes the ceiling
    # entirely instead of raising it again and hitting the same wall on the next
    # long document.
    text_parts, stop_reason = _stream_with_retry(client, content)
    if stop_reason == "max_tokens":
        print(f"  WARNING: transcription of {pdf_path.name} hit max_tokens even with streaming -- likely truncated, some pages may be missing")
    return "".join(text_parts)


def get_document_text(client: anthropic.Anthropic, pdf_path: Path, used_vision: bool) -> str:
    """Returns page-marked text ("[PAGE N]\\n...") for a document, either
    from pypdf (digital) or a one-time vision transcription (scanned)."""
    if used_vision:
        return transcribe_scanned_doc(client, pdf_path)
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(f"[PAGE {i + 1}]\n{t}" for i, t in enumerate(pages_text))


def _page_for_offset(page_positions: list[tuple[int, int]], offset: int) -> int:
    """page_positions: [(char_offset_of_marker, page_number), ...] sorted by offset.
    Returns the page number whose marker precedes `offset`."""
    page = page_positions[0][1] if page_positions else 1
    for marker_offset, page_num in page_positions:
        if marker_offset <= offset:
            page = page_num
        else:
            break
    return page


def chunk_document(text: str) -> list[dict]:
    """Returns a list of {text, section_number, section_title, page_number}.
    Prefers SECTION-header chunking; falls back to page-based chunking when
    fewer than 3 clean section headers are found (short modification/
    amendment docs frequently don't use this convention at all)."""
    page_positions = [(m.start(), int(m.group(1))) for m in PAGE_MARKER_PATTERN.finditer(text)]
    section_matches = list(SECTION_PATTERN.finditer(text))

    if len(section_matches) >= 3:
        chunks = []
        for i, m in enumerate(section_matches):
            start = m.start()
            end = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(text)
            body = text[start:end].strip()
            if not body:
                continue
            chunks.append({
                "text": body,
                "section_number": m.group(1),
                "section_title": m.group(2).strip(),
                "page_number": _page_for_offset(page_positions, start),
            })
        return chunks

    # Fallback: one chunk per page
    chunks = []
    for i, (marker_offset, page_num) in enumerate(page_positions):
        end = page_positions[i + 1][0] if i + 1 < len(page_positions) else len(text)
        body = text[marker_offset:end].strip()
        body = PAGE_MARKER_PATTERN.sub("", body, count=1).strip()
        if not body:
            continue
        chunks.append({
            "text": body,
            "section_number": None,
            "section_title": None,
            "page_number": page_num,
        })
    if not chunks and text.strip():
        # No page markers at all (shouldn't happen given get_document_text,
        # but guard anyway) -- whole document as a single chunk.
        chunks.append({"text": text.strip(), "section_number": None, "section_title": None, "page_number": None})
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--db", default="contracts.db")
    parser.add_argument("--resume", action="store_true",
                         help="Skip filenames already in the collection instead of rebuilding from scratch.")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    docs = conn.execute("""
        SELECT filename, contract_id, doc_role, used_vision,
               modification_summary_value, modification_summary_page
        FROM extractions
    """).fetchall()

    anthropic_client = anthropic.Anthropic()
    voyage_client = voyageai.Client()

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Embed and write per document, not one batch at the end -- a 123-doc run hit a
    # transient network error (connection forcibly closed mid-stream) on doc 109/123
    # and lost all 108 already-processed documents' work, since nothing was persisted
    # until a single collection.add() call after the full loop. Per-doc writes mean a
    # crash only costs the one in-flight document. --resume skips filenames already
    # in the collection so a restart after a crash doesn't re-process (and re-pay for)
    # completed work.
    existing_collection_names = [c.name for c in chroma_client.list_collections()]
    if args.resume and COLLECTION_NAME in existing_collection_names:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        already_done = {m["filename"] for m in collection.get()["metadatas"]}
        print(f"Resuming: {len(already_done)} filenames already in the collection, skipping those.")
    else:
        if COLLECTION_NAME in existing_collection_names:
            chroma_client.delete_collection(COLLECTION_NAME)
        collection = chroma_client.create_collection(COLLECTION_NAME)
        already_done = set()

    chunk_counter = len(collection.get()["ids"])
    section_chunked_docs, page_chunked_docs = [], []
    total_chunks_this_run = 0

    for doc in docs:
        fname = doc["filename"]
        if fname in already_done:
            continue
        pdf_path = Path(args.folder) / fname
        print(f"Processing {fname} ({'vision transcript' if doc['used_vision'] else 'digital text'})...")

        text = get_document_text(anthropic_client, pdf_path, bool(doc["used_vision"]))
        chunks = chunk_document(text)
        used_sections = any(c["section_number"] is not None for c in chunks)
        (section_chunked_docs if used_sections else page_chunked_docs).append(fname)
        print(f"  {len(chunks)} chunks ({'section-based' if used_sections else 'page-based fallback'})")

        doc_chunk_texts, doc_chunk_ids, doc_chunk_metas = [], [], []
        for c in chunks:
            chunk_counter += 1
            doc_chunk_texts.append(c["text"])
            doc_chunk_ids.append(f"chunk_{chunk_counter}")
            doc_chunk_metas.append({
                "filename": fname,
                "contract_id": str(doc["contract_id"]),
                "doc_role": doc["doc_role"],
                "section_number": c["section_number"] or "",
                "section_title": c["section_title"] or "",
                "page_number": c["page_number"] or 0,
                "chunk_type": "raw_clause",
            })

        if doc["doc_role"] in MODIFICATION_DOC_ROLES and doc["modification_summary_value"]:
            if not str(doc["modification_summary_value"]).startswith("not_applicable"):
                chunk_counter += 1
                doc_chunk_texts.append(doc["modification_summary_value"])
                doc_chunk_ids.append(f"chunk_{chunk_counter}")
                doc_chunk_metas.append({
                    "filename": fname,
                    "contract_id": str(doc["contract_id"]),
                    "doc_role": doc["doc_role"],
                    "section_number": "",
                    "section_title": "Modification Summary",
                    "page_number": doc["modification_summary_page"] or 0,
                    "chunk_type": "modification_summary",
                })

        if not doc_chunk_texts:
            continue
        embed_result = voyage_client.embed(doc_chunk_texts, model=EMBED_MODEL, input_type="document")
        collection.add(
            ids=doc_chunk_ids,
            embeddings=embed_result.embeddings,
            documents=doc_chunk_texts,
            metadatas=doc_chunk_metas,
        )
        total_chunks_this_run += len(doc_chunk_texts)

    total_chunks = len(collection.get()["ids"])
    print(f"\nDone. {total_chunks_this_run} chunks added this run, {total_chunks} total stored in "
          f"{CHROMA_DIR} (collection '{COLLECTION_NAME}').")
    print(f"Section-chunked docs this run ({len(section_chunked_docs)}): {section_chunked_docs}")
    print(f"Page-chunked fallback docs this run ({len(page_chunked_docs)}): {page_chunked_docs}")


if __name__ == "__main__":
    main()
