"""Run the NIM LLM extraction over the real annual-report excerpts.

For every excerpt in data/real/excerpts_index.json: chunk the text, call
src.nim_llm.NimClient per chunk, keep only signals whose evidence quotes
verify against the source chunk, and aggregate per company-year into:

  data/real/llm_features.csv  one row per record: per-category max severity,
                              signal counts, going-concern flag, model(s),
                              origin (live/cache), dropped-quote count
  data/real/llm_signals.json  every verified signal with its quote (this is
                              the evidence the dashboard shows)
  data/real/llm_run_log.json  run provenance: timestamps, per-call origin —
                              a "live" claim is only made for calls this run
                              actually made; re-runs read the disk cache and
                              are labelled "cache".

Run:  python scripts/run_llm_extraction.py
"""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.nim_llm import NimClient, NimError, chunk_text  # noqa: E402

REAL = ROOT / "data" / "real"
CATEGORIES = ["liquidity", "leverage", "profitability", "credit",
              "legal", "going_concern", "operational", "positive"]


def main() -> int:
    index = json.loads((REAL / "excerpts_index.json").read_text())
    client = NimClient(cache_dir=REAL / "llm_cache")
    run_log = {"started": datetime.now(timezone.utc).isoformat(),
               "configured_model": client.config.model, "calls": []}
    all_signals, per_record = [], {}

    for entry in index:
        text = (REAL / "excerpts" / entry["file"]).read_text()
        record = per_record.setdefault(entry["record_id"], {
            "record_id": entry["record_id"], "signals": 0,
            "going_concern_flag": False, "dropped_quotes": 0,
            "origins": set(), "models": set(),
            **{f"sev_{c}": 0 for c in CATEGORIES},
        })
        for n, chunk in enumerate(chunk_text(text)):
            try:
                result = client.extract_signals(
                    chunk, report_id=f"{entry['file']}#chunk{n}")
            except NimError as exc:
                # A failed chunk (e.g. free-tier rate limit) must not kill
                # the run: record it and keep the partial, honest result.
                run_log["calls"].append({
                    "excerpt": entry["file"], "chunk": n,
                    "origin": "FAILED", "error": str(exc)[:300]})
                print(f"{entry['file']} chunk {n}: FAILED — {exc}")
                continue
            run_log["calls"].append({
                "excerpt": entry["file"], "chunk": n,
                "origin": result.origin, "model": result.model,
                "signals": len(result.signals),
                "dropped_quotes": result.dropped_quotes})
            print(f"{entry['file']} chunk {n}: origin={result.origin} "
                  f"model={result.model} signals={len(result.signals)} "
                  f"dropped={result.dropped_quotes}")
            record["origins"].add(result.origin)
            record["models"].add(result.model)
            record["dropped_quotes"] += result.dropped_quotes
            record["going_concern_flag"] |= result.going_concern_flag
            for sig in result.signals:
                record["signals"] += 1
                key = f"sev_{sig['category']}"
                record[key] = max(record[key], sig["severity"])
                all_signals.append({
                    "record_id": entry["record_id"],
                    "excerpt_file": entry["file"],
                    "section": entry["section"],
                    "source_pdf_url": entry["source_pdf_url"],
                    "model": result.model, "origin": result.origin, **sig})

    rows = []
    for record in per_record.values():
        record["origins"] = "+".join(sorted(record.pop("origins")))
        record["models"] = ",".join(sorted(record.pop("models")))
        rows.append(record)
    with (REAL / "llm_features.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (REAL / "llm_signals.json").write_text(json.dumps(all_signals, indent=2))
    run_log["finished"] = datetime.now(timezone.utc).isoformat()
    (REAL / "llm_run_log.json").write_text(json.dumps(run_log, indent=2))
    print(f"Wrote {len(rows)} feature rows, {len(all_signals)} verified "
          f"signals, {len(run_log['calls'])} chunk calls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
