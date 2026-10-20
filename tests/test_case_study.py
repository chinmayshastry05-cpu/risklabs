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


def test_excerpt_basis_attached_and_cautioned():
    cohort = load_real_cohort(BASE)
    bajaj = case_study(BASE, cohort, "bajaj_auto_fy2023-24")
    by_excerpt = {}
    for s in bajaj["signals"]:
        by_excerpt.setdefault(s["excerpt_file"], set()).add(s["excerpt_basis"])
    # MD&A ratio quotes are standalone; the auditor report is consolidated.
    assert by_excerpt["bajaj_auto_fy2023-24__mda.txt"] == {"standalone"}
    assert by_excerpt["bajaj_auto_fy2023-24__auditor.txt"] == {"consolidated"}
    assert any("Basis caution" in n and "standalone" in n
               for n in bajaj["limitations"])
    ril = case_study(BASE, cohort, "reliance_fy2023-24")
    ril_bases = {s["excerpt_file"]: s["excerpt_basis"] for s in ril["signals"]}
    assert ril_bases["reliance_fy2023-24__discussion.txt"] == "mixed"
    assert ril_bases["reliance_fy2023-24__auditor.txt"] == "consolidated"
    vil = case_study(BASE, cohort, "vodafone_idea_fy2023-24")
    vil_bases = {s["excerpt_file"]: s["excerpt_basis"] for s in vil["signals"]}
    assert vil_bases["vodafone_idea_fy2023-24__going_concern_note.txt"] == \
        "consolidated"
