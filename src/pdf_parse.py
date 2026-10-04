"""PDF helpers for the RiskLabs prototype.

Uses PyMuPDF to extract text from PDFs and to build a small synthetic sample
report used by the demo / dashboard.

extract_text_from_pdf(path) keeps its original signature and returns the
cleaned full text (backward compatible). extract_report(path) returns a richer
dict: per-page text, extracted tables, and a warnings list describing anything
suspicious (empty pages, encrypted file, no extractable text, truncation).

Robustness notes:
  - encrypted / malformed PDFs raise informative ValueErrors instead of
    tracebacks from deep inside PyMuPDF;
  - empty pages are skipped (recorded in warnings), not treated as errors;
  - tables are extracted with page.find_tables() where PyMuPDF detects them;
  - headers/footers are heuristically removed: a line that appears verbatim on
    more than half the pages is treated as repeated chrome, not content;
  - whitespace is normalized (duplicate spaces, stray line breaks);
  - max_pages / max_chars guards keep pathological files from hanging the app.
"""

import re
from pathlib import Path

import fitz  # PyMuPDF

MAX_PAGES = 200
MAX_CHARS = 500_000

# Synthetic risky-company report excerpt for the demo. Clearly marked as
# fictional: a college prototype must never be mistaken for real data.
SAMPLE_REPORT_TEXT = """SYNTHETIC DATA - FOR EDUCATIONAL USE ONLY.
The company below is entirely fictional and was invented for a college
prototype. Any resemblance to a real company is coincidental.

Management Discussion & Analysis - SynCo Industries Ltd. (fictional), FY 2025-26

The company faces growing liquidity concerns as operating cash flows have
weakened for three consecutive quarters and working capital pressure has
intensified. Revenue declined 18% year-on-year amid falling margins in the
core manufacturing segment and softening demand from key customers.

Total borrowings rose sharply during the year as the company drew down its
working-capital facilities, increasing refinancing risk ahead of large
repayments falling due next year. The auditors have flagged delayed payments
to suppliers, and the company breached a covenant on one term loan, which
management is currently renegotiating with lenders.

Cost-cutting measures are underway, but with mounting losses and declining
revenue, the board continues to monitor the group's going-concern position
closely. No dividend has been declared for the year."""


def clean_text(text: str) -> str:
    """Normalize extracted text: unicode cleanup, whitespace collapsing."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")  # non-breaking space
    text = text.replace("\u200b", "")    # zero-width space
    text = re.sub(r"[ \t]+", " ", text)          # duplicate spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)       # 3+ newlines -> paragraph break
    text = re.sub(r" *\n *", "\n", text)         # spaces around newlines
    # De-hyphenate line-break hyphenation: "liqui-\ndity" -> "liquidity"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text.strip()


def _strip_repeated_lines(pages: list[str]) -> list[str]:
    """Remove header/footer lines: any non-empty line appearing verbatim on
    more than half of the (non-empty) pages is treated as repeated chrome."""
    nonempty = [p for p in pages if p.strip()]
    if len(nonempty) < 3:
        return pages
    from collections import Counter
    counts = Counter()
    page_lines = []
    for p in nonempty:
        lines = {ln.strip() for ln in p.split("\n") if ln.strip() and len(ln.strip()) < 120}
        page_lines.append(lines)
        counts.update(lines)
    repeated = {ln for ln, c in counts.items() if c > len(nonempty) / 2}
    if not repeated:
        return pages
    return ["\n".join(ln for ln in p.split("\n") if ln.strip() not in repeated)
            for p in pages]


def _tables_to_text(page) -> str:
    """Best-effort table extraction via PyMuPDF's table finder.

    Returns tables as pipe-separated rows, or "" if none detected / on error.
    Table detection on messy PDFs can fail; it must never crash extraction.
    """
    try:
        tables = page.find_tables()
    except Exception:
        return ""
    chunks = []
    for table in tables:
        try:
            rows = table.extract()
        except Exception:
            continue
        for row in rows:
            cells = [" ".join(str(c or "").split()) for c in row]
            if any(cells):
                chunks.append(" | ".join(cells))
    return "\n".join(chunks)


def extract_report(path: str, max_pages: int = MAX_PAGES,
                   max_chars: int = MAX_CHARS) -> dict:
    """Extract a PDF report robustly.

    Returns {"text": str, "pages": [str], "tables": [str], "warnings": [str],
             "n_pages": int}. Never raises on content problems — problems are
    reported in "warnings". Raises ValueError only for unreadable files
    (missing, encrypted, not a PDF, zero pages).
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"PDF not found: {path}")
    if p.stat().st_size == 0:
        raise ValueError(f"PDF is empty (0 bytes): {path}")

    try:
        doc = fitz.open(str(p))
    except Exception as exc:
        raise ValueError(f"Could not open as PDF ({type(exc).__name__}): {path}") from exc

    warnings = []
    try:
        if doc.is_encrypted:
            raise ValueError(
                f"PDF is encrypted and cannot be read without a password: {path}")
        n_pages = doc.page_count
        if n_pages == 0:
            raise ValueError(f"PDF has no pages: {path}")
        if n_pages > max_pages:
            warnings.append(
                f"PDF has {n_pages} pages; only the first {max_pages} were read.")
            n_pages = max_pages

        pages, tables = [], []
        empty_pages = 0
        for i in range(n_pages):
            page = doc[i]
            try:
                raw = page.get_text() or ""
            except Exception:
                raw = ""
                warnings.append(f"page {i + 1}: text extraction failed; skipped.")
            tbl = _tables_to_text(page)
            if tbl:
                tables.append(f"[table on page {i + 1}]\n{tbl}")
                raw = raw + "\n" + tbl
            cleaned = clean_text(raw)
            if not cleaned:
                empty_pages += 1
            pages.append(cleaned)

        if empty_pages == n_pages:
            warnings.append(
                "No extractable text found on any page — the PDF may be "
                "scanned images (needs OCR) or contain only graphics.")

        pages = _strip_repeated_lines(pages)
        text = clean_text("\n\n".join(p for p in pages if p))
        if len(text) > max_chars:
            warnings.append(
                f"Text truncated to {max_chars} characters "
                f"(extracted {len(text)}).")
            text = text[:max_chars]
        return {"text": text, "pages": pages, "tables": tables,
                "warnings": warnings, "n_pages": n_pages}
    finally:
        doc.close()


def extract_text_from_pdf(path: str) -> str:
    """Extract and return all text from the PDF at *path*, in page order.

    Backward-compatible wrapper around extract_report(): returns the cleaned
    text. Raises ValueError with a clear message for unreadable files.
    """
    return extract_report(path)["text"]


def create_sample_pdf(path: str, text: str) -> None:
    """Build a small PDF at *path* containing *text* (flows onto new pages if needed)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()
    margin = 72
    page = doc.new_page()
    full = fitz.Rect(margin, margin, page.rect.width - margin, page.rect.height - margin)
    cursor = fitz.Rect(full.x0, full.y0, full.x1, full.y1)

    def new_page() -> None:
        nonlocal page, cursor
        page = doc.new_page()
        cursor = fitz.Rect(full.x0, full.y0, full.x1, full.y1)

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if cursor.y0 >= full.y1 - 40:
            new_page()
        leftover = page.insert_textbox(cursor, paragraph, fontsize=11, fontname="helv")
        if leftover < 0:
            # Paragraph did not fit: retry on a fresh page.
            new_page()
            leftover = page.insert_textbox(cursor, paragraph, fontsize=11, fontname="helv")
        used = (cursor.y1 - cursor.y0) - max(leftover, 0.0)
        cursor.y0 += used + 16

    doc.save(str(out))
    doc.close()


if __name__ == "__main__":
    # Demo: write data/sample_report.pdf, then extract it back.
    root = Path(__file__).resolve().parent.parent
    sample_path = root / "data" / "sample_report.pdf"
    create_sample_pdf(str(sample_path), SAMPLE_REPORT_TEXT)
    report = extract_report(str(sample_path))
    print(f"wrote: {sample_path}")
    print(f"pages: {report['n_pages']} | tables: {len(report['tables'])} "
          f"| warnings: {report['warnings']}")
    print(f"char count: {len(report['text'])}")
    print("first 400 chars:")
    print(report["text"][:400])
