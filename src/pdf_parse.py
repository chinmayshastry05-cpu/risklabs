"""PDF helpers for the RiskLabs prototype.

Uses PyMuPDF (fitz) to extract text from PDFs and to build a small
synthetic sample report used by the demo / dashboard.
"""

from pathlib import Path

import fitz  # PyMuPDF

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


def extract_text_from_pdf(path: str) -> str:
    """Extract and return all text from the PDF at *path*, in page order."""
    doc = fitz.open(path)
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


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
    extracted = extract_text_from_pdf(str(sample_path))
    print(f"wrote: {sample_path}")
    print(f"char count: {len(extracted)}")
    print("first 400 chars:")
    print(extracted[:400])
