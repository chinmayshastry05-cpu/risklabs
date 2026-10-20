"""Build the real-data CSVs and text excerpts from extracted_financials.json.

The JSON is the auditable transcription record: every value in it names the
PDF page, printed page, statement section and printed line label it came
from. This script validates each record (balance-sheet identity
total_assets == total_equity + total_liabilities within rounding), then
writes:
  data/real/risklabs_real.csv   one row per company-year (INR crore)
  data/real/field_trace.csv     one row per field (full provenance)
  data/real/excerpts/*.txt      MD&A / auditor / notes text excerpts
  data/real/excerpts_index.json excerpt provenance

Run:  python scripts/extract_real.py
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fitz  # noqa: E402

from src.pdf_parse import clean_text  # noqa: E402

REAL = ROOT / "data" / "real"
RAW_FIELDS = ["revenue", "net_profit", "total_assets", "total_liabilities",
              "total_debt", "equity", "current_assets", "current_liabilities",
              "operating_cash_flow"]
MAX_EXCERPT_CHARS = 14000


def main() -> int:
    data = json.loads((REAL / "extracted_financials.json").read_text())
    manifest = {r["filename"]: r for r in
                json.loads((REAL / "provenance_manifest.json").read_text())}

    rows, trace = [], []
    for rec in data["records"]:
        f = rec["fields"]
        identity = f["equity"]["value"] + f["total_liabilities"]["value"]
        gap = abs(identity - f["total_assets"]["value"])
        status = "OK" if gap <= 1.0 else f"FAIL gap={gap}"
        print(f"{rec['record_id']}: balance identity {status} "
              f"(equity {f['equity']['value']} + liabilities "
              f"{f['total_liabilities']['value']} vs assets "
              f"{f['total_assets']['value']})")
        if gap > 1.0:
            return 1
        src = manifest[rec["pdf"]]
        row = {"record_id": rec["record_id"], "company": rec["company"],
               "fy": rec["fy"], "basis": rec["basis"], "units": rec["units"],
               "statements_date": rec["statements_date"],
               "source_page": src["source_page"],
               "source_pdf_url": src["pdf_url"]}
        for name in RAW_FIELDS:
            row[name] = f[name]["value"]
            trace.append({
                "record_id": rec["record_id"], "field": name,
                "value_inr_crore": f[name]["value"],
                "printed_value": f[name].get("printed_value_mn", ""),
                "pdf_page": f[name]["pdf_page"],
                "printed_page": f[name].get("printed_page", ""),
                "section": f[name]["section"],
                "printed_label": f[name]["label"],
                "source_pdf_url": src["pdf_url"],
            })
        row["other_major_obligations_inr_crore"] = \
            rec["other_major_obligations"]["value"]
        row["other_major_obligations_note"] = \
            rec["other_major_obligations"]["label"]
        rows.append(row)

    with (REAL / "risklabs_real.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (REAL / "field_trace.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(trace[0].keys()))
        writer.writeheader()
        writer.writerows(trace)
    print(f"Wrote {len(rows)} records, {len(trace)} traced fields")

    # Text excerpts for the classical-NLP and LLM pipelines.
    excerpt_dir = REAL / "excerpts"
    excerpt_dir.mkdir(exist_ok=True)
    index = []
    docs = {}
    for rec in data["records"]:
        for ex in rec.get("excerpts", []):
            if rec["pdf"] not in docs:
                docs[rec["pdf"]] = fitz.open(REAL / "pdfs" / rec["pdf"])
            doc = docs[rec["pdf"]]
            lo, hi = ex["pdf_pages"]
            text = "\n".join(doc[p - 1].get_text()
                             for p in range(lo, hi + 1))
            text = clean_text(text)[:MAX_EXCERPT_CHARS]
            (excerpt_dir / ex["file"]).write_text(text)
            index.append({
                "record_id": rec["record_id"], "file": ex["file"],
                "section": ex["section"], "pdf_pages": ex["pdf_pages"],
                "chars": len(text),
                "basis": ex.get("basis", "not labelled"),
                "basis_note": ex.get("basis_note", ""),
                "source_pdf_url": manifest[rec["pdf"]]["pdf_url"],
            })
            print(f"excerpt {ex['file']}: {len(text)} chars "
                  f"(PDF pp. {lo}-{hi})")
    (REAL / "excerpts_index.json").write_text(json.dumps(index, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
