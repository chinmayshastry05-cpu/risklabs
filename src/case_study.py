"""Real-data case studies: historical, evidence-backed risk assessment.

Loads the real cohort (data/real/risklabs_real.csv — figures transcribed
from published annual reports, see data/real/field_trace.csv for the page
and line each value came from), computes the same ratios as the synthetic
pipeline, and lines them up with the two text views:

  * classical NLP baseline — the existing lexicon features (src.nlp) run on
    the report excerpts, and
  * LLM signals — Nemotron extractions (src.nim_llm) whose evidence quotes
    were verified verbatim against the source text.

This module deliberately produces NO risk label and NO prediction for real
companies. With a small cohort and no verified future-outcome labels, a
trained prediction claim would be unfounded; this is a historical
risk-assessment case-study mode, and predictive validation on real data is
pending. The synthetic pipeline (src.train) remains the place where the
Model A vs Model B predictive comparison lives.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features import compute_ratios
from src.nlp import extract_lexicon_features

PENDING_VALIDATION_NOTE = (
    "Historical assessment of published financial statements — not a "
    "prediction. Predictive validation on real outcome labels is pending: "
    "the cohort is small and no verified future-outcome label set exists "
    "for it, so no accuracy claim is made for real companies."
)


def load_real_cohort(base: Path) -> dict:
    """Load the real cohort + LLM outputs. Returns dict of frames/lists."""
    real_dir = Path(base) / "data" / "real"
    financials = pd.read_csv(real_dir / "risklabs_real.csv")
    financials = compute_ratios(financials)
    llm_features = pd.read_csv(real_dir / "llm_features.csv") \
        if (real_dir / "llm_features.csv").exists() else pd.DataFrame()
    signals = json.loads((real_dir / "llm_signals.json").read_text()) \
        if (real_dir / "llm_signals.json").exists() else []
    run_log = json.loads((real_dir / "llm_run_log.json").read_text()) \
        if (real_dir / "llm_run_log.json").exists() else {}
    excerpts = {}
    for entry in json.loads((real_dir / "excerpts_index.json").read_text()):
        excerpts.setdefault(entry["record_id"], []).append(entry)
    return {"financials": financials, "llm_features": llm_features,
            "signals": signals, "run_log": run_log, "excerpts": excerpts}


def record_limitations(row: pd.Series) -> list[str]:
    """Honest, record-specific caveats. Shown verbatim in the dashboard."""
    notes = []
    if row["equity"] <= 0:
        notes.append(
            "Equity is negative (net worth eroded): debt-to-equity and ROE "
            "are not meaningful for this record and are shown for "
            "traceability only.")
    if row["total_debt"] == 0:
        notes.append(
            "No borrowings were reported for this year: ocf_to_debt is not "
            "applicable (the pipeline's divide-by-zero guard value is not a "
            "real ratio).")
    other = row.get("other_major_obligations_inr_crore", 0) or 0
    if other > row["total_debt"]:
        notes.append(
            f"total_debt counts borrowings only and excludes "
            f"₹{other:,.0f} crore of other major obligations "
            f"({row.get('other_major_obligations_note', '')}) — leverage "
            "ratios understate total obligations for this company.")
    if str(row["record_id"]).endswith("fy2022-23"):
        notes.append(
            "Figures are the prior-year comparative columns of the "
            "FY2023-24 annual report (as restated there), not the original "
            "FY2022-23 report.")
    notes.append(PENDING_VALIDATION_NOTE)
    return notes


def lexicon_baseline(base: Path, cohort: dict, record_id: str):
    """Classical lexicon features over the record's own excerpts.

    Returns None for comparative-year records: their source report's text
    belongs to the FY2023-24 record, and attributing it to FY2022-23 would
    be future-text leakage."""
    entries = cohort["excerpts"].get(record_id, [])
    if not entries:
        return None
    text = "\n".join(
        (Path(base) / "data" / "real" / "excerpts" / e["file"]).read_text()
        for e in entries)
    return extract_lexicon_features([text]).iloc[0].to_dict()


def case_study(base: Path, cohort: dict, record_id: str) -> dict:
    """Everything the dashboard tab needs for one company-year."""
    fin = cohort["financials"]
    row = fin[fin["record_id"] == record_id].iloc[0]
    llm_row = None
    if not cohort["llm_features"].empty:
        match = cohort["llm_features"][
            cohort["llm_features"]["record_id"] == record_id]
        if not match.empty:
            llm_row = match.iloc[0].to_dict()
    signals = [s for s in cohort["signals"]
               if s["record_id"] == record_id]
    return {
        "record": row.to_dict(),
        "limitations": record_limitations(row),
        "lexicon": lexicon_baseline(base, cohort, record_id),
        "llm": llm_row,
        "signals": signals,
        "excerpt_entries": cohort["excerpts"].get(record_id, []),
    }
