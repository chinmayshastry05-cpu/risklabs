"""RiskLabs smoke tests — run from the project root with:

    python -m pytest tests/ -q

These are fast pipeline checks (no full retraining). They verify that feature
engineering, NLP, PDF extraction, evaluation, calibration helpers, temporal
utilities, trend scoring, explanations, and the saved artifacts are all
internally consistent.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import evaluate_classifier, expected_calibration_error, ordered_proba  # noqa: E402
from src.explain import (display_name, factor_sentence, feature_family,  # noqa: E402
                         reasons_to_text, risk_score_from_proba,
                         shap_waterfall_figure)
from src.features import compute_ratios, compute_ratios_dict  # noqa: E402
from src.nlp import (RiskTextVectorizer, extract_lexicon_features,  # noqa: E402
                     select_text_features, text_feature_report)
from src.pdf_parse import clean_text, create_sample_pdf, extract_report  # noqa: E402
from src.temporal import temporal_split, validate_temporal_columns, walk_forward_splits  # noqa: E402
from src.trends import trend_direction, yearly_risk_profile  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"


# ---------- features ----------

def test_compute_ratios_basic():
    df = pd.DataFrame([{
        "revenue": 100.0, "net_profit": 10.0, "total_assets": 200.0,
        "total_liabilities": 80.0, "total_debt": 60.0, "equity": 120.0,
        "current_assets": 50.0, "current_liabilities": 25.0,
        "operating_cash_flow": 12.0,
    }])
    r = compute_ratios(df).iloc[0]
    assert r["current_ratio"] == pytest.approx(2.0)
    assert r["debt_to_equity"] == pytest.approx(0.5)
    assert r["net_profit_margin"] == pytest.approx(0.1)
    d = compute_ratios_dict(df.iloc[0].to_dict())
    assert d["current_ratio"] == pytest.approx(2.0)


def test_compute_ratios_zero_guard():
    df = pd.DataFrame([{
        "revenue": 100.0, "net_profit": 10.0, "total_assets": 200.0,
        "total_liabilities": 80.0, "total_debt": 60.0, "equity": 0.0,
        "current_assets": 50.0, "current_liabilities": 0.0,
        "operating_cash_flow": 12.0,
    }])
    r = compute_ratios(df).iloc[0]
    assert np.isfinite(r["current_ratio"]) and np.isfinite(r["debt_to_equity"])


# ---------- nlp ----------

def test_lexicon_detects_risk_phrase():
    df = extract_lexicon_features(["We face a cash crunch and a covenant breach."])
    assert df["lex_liquidity"].iloc[0] > 0
    assert df["lex_leverage"].iloc[0] > 0
    df2 = extract_lexicon_features(["Stable demand and a robust balance sheet."])
    assert df2["lex_positive"].iloc[0] < 0  # healthy language has negative weight


def test_tfidf_fit_transform_shapes():
    vec = RiskTextVectorizer(max_features=50).fit(["cash crunch risk", "stable demand growth"])
    X = vec.transform(["cash crunch", "stable demand"])
    assert X.shape == (2, len(vec.get_feature_names()))


def test_text_feature_report_keys():
    vec = RiskTextVectorizer(max_features=50).fit(["cash crunch risk"] * 5 + ["stable demand"] * 5)
    rep = text_feature_report(["cash crunch"] * 5 + ["stable demand"] * 5, vec)
    assert rep["n_docs"] == 10 and rep["n_terms"] <= 50
    assert 0.0 <= rep["sparsity"] <= 1.0


def test_select_text_features_reduces_dim():
    texts = [f"risk word number {i} term alpha" for i in range(20)] + \
            [f"safe word number {i} term beta" for i in range(20)]
    vec = RiskTextVectorizer(max_features=100).fit(texts)
    X = vec.transform(texts)
    assert X.shape[1] > 10  # enough terms for a meaningful selection
    Xs, sel = select_text_features(X, np.array([1] * 20 + [0] * 20), k=10)
    assert Xs.shape == (40, 10)


# ---------- pdf ----------

def test_clean_text_collapses_whitespace():
    assert clean_text("a   b\n\n\nc") == "a b\n\nc"
    assert clean_text("liqui-\ndity") == "liquidity"


def test_extract_report_sample_pdf(tmp_path):
    from src.pdf_parse import SAMPLE_REPORT_TEXT
    p = tmp_path / "sample.pdf"
    create_sample_pdf(str(p), SAMPLE_REPORT_TEXT)
    rep = extract_report(str(p))
    assert rep["n_pages"] >= 1
    assert "liquidity" in rep["text"].lower()
    assert isinstance(rep["warnings"], list)


def test_extract_report_missing_file():
    with pytest.raises(ValueError):
        extract_report("/nonexistent/path/report.pdf")


# ---------- evaluate ----------

def _toy():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    y_proba = np.array([
        [0.7, 0.2, 0.1], [0.4, 0.5, 0.1], [0.2, 0.6, 0.2],
        [0.1, 0.8, 0.1], [0.1, 0.2, 0.7], [0.5, 0.3, 0.2],
    ])
    return y_true, y_pred, y_proba


def test_evaluate_classifier_keys():
    y_true, y_pred, y_proba = _toy()
    m = evaluate_classifier(y_true, y_pred, y_proba)
    for key in ("accuracy", "confusion_matrix", "per_class", "f1_macro",
                "f1_weighted", "roc_auc_ovr", "high_risk_precision",
                "high_risk_recall", "ece", "classification_report"):
        assert key in m
    assert np.asarray(m["confusion_matrix"]).shape == (3, 3)
    assert m["per_class"]["High"]["support"] == 2
    assert 0.0 <= m["ece"] <= 1.0


def test_expected_calibration_error_perfect():
    y = np.array([0, 1, 2])
    p = np.eye(3)
    assert expected_calibration_error(y, p) == pytest.approx(0.0)


class _IntClassModel:
    classes_ = np.array([0, 1, 2])
    def predict_proba(self, x):
        return np.array([[0.1, 0.2, 0.7]] * len(x))


class _StrClassModel:
    classes_ = np.array(["High", "Low", "Medium"])  # alphabetical storage
    def predict_proba(self, x):
        return np.array([[0.7, 0.1, 0.2]] * len(x))  # P(High)=0.7


def test_ordered_proba_int_and_str_classes():
    x = np.zeros((1, 3))
    p_int = ordered_proba(_IntClassModel(), x)
    assert list(np.round(p_int, 4)) == [0.1, 0.2, 0.7]
    p_str = ordered_proba(_StrClassModel(), x)
    assert list(np.round(p_str, 4)) == [0.1, 0.2, 0.7]  # reordered to Low/Med/High


# ---------- temporal ----------

def test_temporal_rejects_cross_section():
    df = pd.DataFrame({"risk_label": ["Low", "High"]})
    with pytest.raises(ValueError, match="report_year"):
        validate_temporal_columns(df)


def _longitudinal_df():
    rows = []
    for year in range(2015, 2023):
        for i in range(10):
            rows.append({"report_year": year, "risk_label": "Low", "v": i})
    return pd.DataFrame(rows)


def test_temporal_split_windows():
    df = _longitudinal_df()
    tr, va, te = temporal_split(df, train_years=(2015, 2018),
                                val_years=(2019, 2020), test_years=(2021, 2022))
    assert tr["report_year"].max() < va["report_year"].min() <= va["report_year"].max() < te["report_year"].min()
    assert len(tr) + len(va) + len(te) == len(df)


def test_walk_forward_folds_chronological():
    folds = walk_forward_splits(_longitudinal_df(), train_window=4, test_window=1)
    assert len(folds) >= 2
    for f in folds:
        assert f["train_years"][1] < f["test_years"][0]


# ---------- trends ----------

def _yearly_df():
    return pd.DataFrame([
        {"year": 2021, "revenue": 100.0, "net_profit": 12.0, "total_assets": 200.0,
         "total_liabilities": 60.0, "total_debt": 40.0, "equity": 140.0,
         "current_assets": 60.0, "current_liabilities": 30.0, "operating_cash_flow": 15.0},
        {"year": 2022, "revenue": 100.0, "net_profit": 2.0, "total_assets": 200.0,
         "total_liabilities": 100.0, "total_debt": 80.0, "equity": 100.0,
         "current_assets": 40.0, "current_liabilities": 50.0, "operating_cash_flow": 2.0},
        {"year": 2023, "revenue": 100.0, "net_profit": -8.0, "total_assets": 200.0,
         "total_liabilities": 150.0, "total_debt": 130.0, "equity": 50.0,
         "current_assets": 30.0, "current_liabilities": 60.0, "operating_cash_flow": -10.0},
    ])


class _RiskyModel:
    """Mock: deteriorating financials -> rising High probability."""
    classes_ = np.array([0, 1, 2])
    def predict_proba(self, x):
        # x rows correspond to 2021 (safe) .. 2023 (risky); use debt_to_equity col
        out = []
        for row in x:
            dte = row[1]
            p_high = float(np.clip(dte / 3.0, 0.05, 0.95))
            out.append([1 - p_high - 0.05, 0.05, p_high])
        return np.array(out)


def test_yearly_risk_profile_and_trend():
    profile = yearly_risk_profile(_RiskyModel(), _yearly_df())
    assert list(profile["year"]) == [2021, 2022, 2023]
    assert set(profile["risk_class"]) <= {"Low", "Medium", "High"}
    trend = trend_direction(profile)
    assert trend["direction"] == "increasing"
    assert trend["n_years"] == 3


def test_trend_needs_two_years():
    with pytest.raises(ValueError):
        trend_direction(_yearly_df().head(0).assign(risk_score=[]))


# ---------- explain ----------

def test_risk_score_from_proba():
    assert risk_score_from_proba([1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert risk_score_from_proba([0.0, 0.0, 1.0]) == pytest.approx(100.0)
    assert risk_score_from_proba([0.0, 1.0, 0.0]) == pytest.approx(50.0)


def test_feature_family_and_display():
    assert feature_family("debt_to_equity") == "financial"
    assert feature_family("lex_leverage") == "lexicon"
    assert feature_family("some_tfidf_term") == "tfidf"
    assert display_name("debt_to_equity") == "Debt-to-equity ratio"
    assert display_name("bankruptcy") == 'word/phrase "bankruptcy"'


def test_factor_sentence_traceability():
    up = {"display": "Debt-to-equity ratio", "family": "financial",
          "value": 2.4, "contribution": 0.3, "direction": "increases risk"}
    s = factor_sentence(up)
    assert s is not None and "2.40" in s and "increased" in s
    # TF-IDF terms must never get narrated sentences
    tfidf = {"display": 'word/phrase "growth"', "family": "tfidf",
             "value": 0.5, "contribution": 0.3, "direction": "increases risk"}
    assert factor_sentence(tfidf) is None
    # Contradictory value/direction -> no sentence invented
    odd = {"display": "Debt-to-equity ratio", "family": "financial",
           "value": 0.1, "contribution": 0.3, "direction": "increases risk"}
    assert factor_sentence(odd) is None


def test_reasons_to_text_mentions_level():
    reasons = [{"display": "Debt-to-equity ratio", "family": "financial",
                "value": 2.4, "contribution": 0.3, "direction": "increases risk"}]
    txt = reasons_to_text(reasons, "High")
    assert "High" in txt and "debt-to-equity" in txt.lower()


def test_shap_waterfall_figure():
    reasons = [
        {"display": "Debt-to-equity ratio", "family": "financial", "value": 2.4,
         "contribution": 0.3, "direction": "increases risk"},
        {"display": "Current ratio", "family": "financial", "value": 2.5,
         "contribution": -0.2, "direction": "reduces risk"},
    ]
    fig = shap_waterfall_figure(reasons)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------- artifacts ----------

def _n_features(model):
    """Feature count a saved artifact expects, unwrapping pipelines/calibrators."""
    m = model
    if hasattr(m, "calibrated_classifiers_"):  # CalibratedClassifierCV
        m = m.calibrated_classifiers_[0].estimator
    if hasattr(m, "steps"):  # Pipeline
        m = m.steps[-1][1]
    return int(m.n_features_in_)


def test_artifacts_feature_counts_match_feature_sets():
    import joblib
    n_tfidf = len(json.loads((MODELS_DIR / "feature_names.json").read_text())["tfidf"])
    n_b = 7 + 8 + n_tfidf
    for name, expected in [("model_a.pkl", 7), ("model_b.pkl", n_b),
                           ("model_a_lr.pkl", 7), ("model_b_lr.pkl", n_b),
                           ("model_a_xgb.pkl", 7), ("model_b_xgb.pkl", n_b),
                           ("model_a_calibrated.pkl", 7),
                           ("model_b_calibrated.pkl", n_b)]:
        m = joblib.load(MODELS_DIR / name)
        assert _n_features(m) == expected, f"{name} expects wrong feature count"


def test_artifacts_present_and_consistent():
    required = ["model_a.pkl", "model_b.pkl", "tfidf_vectorizer.pkl",
                "feature_names.json", "median_profile.json", "metrics.json"]
    for name in required:
        assert (MODELS_DIR / name).exists(), f"missing artifact: {name}"
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text())
    assert "synthetic" in metrics["note"].lower()
    assert metrics["n_train"] == 960 and metrics["n_test"] == 240
    # Upgraded evaluation stores the full per-model metric sets.
    assert "models" in metrics
    for mid in ("rf_a", "lr_a", "xgb_a", "rf_b", "lr_b", "xgb_b"):
        assert mid in metrics["models"], f"missing model metrics: {mid}"
        m = metrics["models"][mid]
        assert np.asarray(m["confusion_matrix"]).shape == (3, 3)
        assert "High" in m["per_class"]
    # Legacy flat keys kept for backward compatibility.
    assert set(metrics["model_a"]) >= {"accuracy", "f1_macro", "roc_auc_ovr"}
