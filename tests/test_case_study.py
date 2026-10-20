"""Tests for the real-data case-study module (no network, no LLM calls)."""

from pathlib import Path

from src.case_study import (case_study, load_real_cohort,
                            record_limitations)

BASE = Path(__file__).resolve().parent.parent


def _row(cohort, record_id):
    fin = cohort["financials"]
    return fin[fin["record_id"] == record_id].iloc[0]


def test_cohort_loads_six_traced_records():
    cohort = load_real_cohort(BASE)
    assert len(cohort["financials"]) == 6
    assert set(cohort["financials"]["fy"]) == {"2023-24", "2022-23"}


def test_limitations_flag_negative_equity_and_deferred_obligations():
    cohort = load_real_cohort(BASE)
    notes = " ".join(record_limitations(_row(cohort, "vodafone_idea_fy2023-24")))
    assert "negative" in notes
    assert "understate total obligations" in notes
    assert "not a prediction" in notes


def test_limitations_flag_zero_debt_and_comparatives():
    cohort = load_real_cohort(BASE)
    notes = " ".join(record_limitations(_row(cohort, "bajaj_auto_fy2022-23")))
    assert "ocf_to_debt is not applicable" in notes
    assert "comparative columns" in notes


def test_lexicon_baseline_skips_comparative_years():
    cohort = load_real_cohort(BASE)
    study24 = case_study(BASE, cohort, "bajaj_auto_fy2023-24")
    study23 = case_study(BASE, cohort, "bajaj_auto_fy2022-23")
    assert study24["lexicon"] is not None
    assert study23["lexicon"] is None  # would be future-text leakage
