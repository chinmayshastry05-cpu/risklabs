"""Reproducible fetch of real annual reports for the RiskLabs real-data cohort.

Downloads official annual-report PDFs from company investor pages, verifies
integrity (server Content-Length when offered + PyMuPDF open + page count),
and writes a provenance manifest. PDFs themselves are NEVER committed to the
repo (see .gitignore) — only this manifest and the extracted data are. The
manifest links the original URLs instead of redistributing the reports:
public access is not an open redistribution licence.

Run:  python scripts/fetch_reports.py
Re-running is safe: complete files are kept, partial files are resumed.

Cohort (all FY2021 or later, official sources only):
  - Bajaj Auto Ltd        FY2023-24  (healthy)   investors.bajajauto.com
  - Reliance Industries   FY2023-24  (healthy)   ril.com
  - Vodafone Idea Ltd     FY2023-24  (stressed)  myvi.in (official Vi site)
Vodafone Idea FY2022-23 figures come from the prior-year comparative columns
of the FY2023-24 report (traced at extraction time). Note: the file named
"Annual_Report_2022-23.pdf" on myvi.in actually serves the FY2019-20 report
(verified by inspection 2026-10-20) — it is deliberately NOT used.
"""

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "real" / "pdfs"
MANIFEST_JSON = ROOT / "data" / "real" / "provenance_manifest.json"
MANIFEST_CSV = ROOT / "data" / "real" / "provenance_manifest.csv"

SOURCES = [
    {
        "company": "Bajaj Auto Limited",
        "short": "bajaj_auto",
        "fy": "2023-24",
        "basis": "consolidated (standalone also in report)",
        "source_page": "https://investors.bajajauto.com/ar24/",
        "pdf_url": "https://investors.bajajauto.com/ar24/wp-content/uploads/2024/06/BAL_Annual_Report_2024.pdf",
        "filename": "bajaj_auto_ar_fy2023-24.pdf",
    },
    {
        "company": "Reliance Industries Limited",
        "short": "reliance",
        "fy": "2023-24",
        "basis": "consolidated (standalone also in report)",
        "source_page": "https://www.ril.com/ar2023-24/index.html",
        "pdf_url": "https://www.ril.com/ar2023-24/pdf/RIL_IAR_2024.pdf",
        "filename": "reliance_ar_fy2023-24.pdf",
    },
    {
        "company": "Vodafone Idea Limited",
        "short": "vodafone_idea",
        "fy": "2023-24",
        "basis": "consolidated (standalone also in report)",
        "source_page": "https://www.myvi.in/investors/annual-reports",
        "pdf_url": "https://www.myvi.in/documents/106724/171100/VIL-Annual-Report-2023-24.pdf",
        "filename": "vodafone_idea_ar_fy2023-24.pdf",
    },
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(src: dict) -> dict:
    import fitz  # PyMuPDF, already a project dependency

    dest = PDF_DIR / src["filename"]
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    # Download to a private .part file, never onto the final name: resuming
    # onto a file another process may still be writing can concatenate two
    # copies into a hybrid that opens fine but has doubled pages (observed
    # with the Vodafone Idea report, 2026-10-20). Verify a proper %%EOF
    # trailer, then atomically rename into place.
    part = PDF_DIR / (src["filename"] + ".part")
    part.unlink(missing_ok=True)
    subprocess.run(
        ["curl", "-sL", "--retry", "5", "--retry-all-errors",
         "-o", str(part), src["pdf_url"]],
        check=True,
    )
    tail = part.read_bytes()[-4096:]
    if b"%%EOF" not in tail:
        raise RuntimeError(f"{src['filename']}: no %%EOF trailer — "
                           "download looks truncated, refusing to keep it")
    part.replace(dest)
    doc = fitz.open(dest)
    record = {
        **src,
        "local_file": f"data/real/pdfs/{src['filename']}",
        "bytes": dest.stat().st_size,
        "sha256": sha256_of(dest),
        "pages": doc.page_count,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "usage_note": (
            "Publicly accessible company annual report. Linked, not bundled: "
            "this PDF is gitignored and is not redistributed by this repo."
        ),
    }
    return record


def main() -> int:
    records = []
    for src in SOURCES:
        rec = fetch(src)
        print(f"{rec['company']} FY{rec['fy']}: {rec['bytes']} bytes, "
              f"{rec['pages']} pages, sha256 {rec['sha256'][:16]}...")
        records.append(rec)
    MANIFEST_JSON.write_text(json.dumps(records, indent=2))
    with MANIFEST_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Manifest: {MANIFEST_JSON} + {MANIFEST_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
