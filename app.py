"""RiskLabs Streamlit dashboard - explainable financial-risk prototype.

Run from the project root with:  streamlit run app.py

All inputs here are synthetic / user-provided demo data. This is a college
prototype for educational use only - not financial advice.

Tabs:
  1. Analyze company — risk classification, model probabilities (raw and
     calibrated), heuristic risk indicators, SHAP explanations.
  2. Model comparison — classifier × feature-set table, confusion matrices,
     classification reports, calibration results, and the temporal-validation
     plan.
  3. Risk trend — multi-year risk trajectory infrastructure. Requires the user
     to supply real per-year financials; the shipped dataset is a single
     cross-section, so no trend is pre-computed or fabricated.
"""

import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Make `src.*` importable no matter where Streamlit is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from src.evaluate import ordered_proba
    from src.explain import (PROBABILITY_TERMS, explain_prediction,
                             reasons_to_text, risk_score_from_proba,
                             shap_waterfall_figure)
    from src.features import RATIO_FEATURES, compute_ratios_dict
    from src.nlp import LEXICON_FEATURES, extract_lexicon_features
    from src.pdf_parse import extract_report
    from src.temporal import describe_temporal_plan
    from src.trends import (profile_to_yearly_table, trend_direction,
                            yearly_risk_profile)

    _IMPORT_ERROR = None
except Exception as exc:  # sibling modules are built by parallel workers
    _IMPORT_ERROR = exc

LABELS = ["Low", "Medium", "High"]

# (key, input label, sensible default in Rs. crore)
FINANCIAL_FIELDS = [
    ("revenue", "Revenue (Rs. crore)", 100.0),
    ("net_profit", "Net profit (Rs. crore)", 5.0),
    ("total_assets", "Total assets (Rs. crore)", 120.0),
    ("total_liabilities", "Total liabilities (Rs. crore)", 60.0),
    ("total_debt", "Total debt (Rs. crore)", 45.0),
    ("equity", "Equity (Rs. crore)", 60.0),
    ("current_assets", "Current assets (Rs. crore)", 50.0),
    ("current_liabilities", "Current liabilities (Rs. crore)", 40.0),
    ("operating_cash_flow", "Operating cash flow (Rs. crore)", 8.0),
]

MODEL_PRETTY = {
    "rf_a": "RF · ratios (Model A)",
    "lr_a": "LogReg · ratios",
    "xgb_a": "XGBoost · ratios",
    "rf_b": "RF · ratios+text (Model B)",
    "lr_b": "LogReg · ratios+text",
    "xgb_b": "XGBoost · ratios+text",
}


@st.cache_resource
def load_artifacts():
    """Load trained models/vectorizer/JSONs; paths are relative to app.py."""
    import json

    base = Path(__file__).resolve().parent / "models"
    required = [
        "model_a.pkl",
        "model_b.pkl",
        "tfidf_vectorizer.pkl",
        "feature_names.json",
        "median_profile.json",
        "metrics.json",
    ]
    missing = [name for name in required if not (base / name).exists()]
    if missing:
        raise FileNotFoundError(
            "missing model file(s): " + ", ".join(missing)
        )
    artifacts = {
        "model_a": joblib.load(base / "model_a.pkl"),
        "model_b": joblib.load(base / "model_b.pkl"),
        "vectorizer": joblib.load(base / "tfidf_vectorizer.pkl"),
        "feature_names": json.loads((base / "feature_names.json").read_text()),
        "median_profile": json.loads((base / "median_profile.json").read_text()),
        "metrics": json.loads((base / "metrics.json").read_text()),
    }
    # Newer artifacts (present after retraining with the upgraded train.py).
    for name in ["model_a_lr.pkl", "model_b_lr.pkl",
                 "model_a_xgb.pkl", "model_b_xgb.pkl",
                 "model_a_calibrated.pkl", "model_b_calibrated.pkl"]:
        key = name[:-4]
        artifacts[key] = joblib.load(base / name) if (base / name).exists() else None
    return artifacts


def build_feature_rows(financials, text, artifacts):
    """Build the exact feature rows Model A and Model B were trained on."""
    ratios = compute_ratios_dict(financials)
    lex_df = extract_lexicon_features([text])
    feature_names = artifacts["feature_names"]

    x_a = np.array([[ratios[f] for f in RATIO_FEATURES]], dtype=float)
    x_b = np.hstack(
        [
            x_a,
            lex_df[LEXICON_FEATURES].to_numpy(dtype=float),
            artifacts["vectorizer"].transform([text]).toarray(),
        ]
    )
    names_b = (
        feature_names["financial"]
        + feature_names["lexicon"]
        + feature_names["tfidf"]
    )
    return x_a, x_b, names_b, ratios, lex_df.iloc[0].to_dict()


def clamp100(value):
    return float(max(0.0, min(100.0, value)))


def heuristic_category_scores(ratios, lex):
    """Transparent 0-100 heuristic sub-scores (higher = riskier).

    NOT model output - simple documented transforms of ratios + lexicon hits:
      liquidity:     100/(1+current_ratio), plus 15 per liquidity-phrase hit
      leverage:      saturating 100*de/(1+de) on debt_to_equity, plus 15 per leverage hit
      profitability: logistic 100/(1+exp(20*margin)) on net_profit_margin, plus 15 per hit
      operational:   20*legal hits + 25*going_concern hits + 10*risk sentences
    """
    current_ratio = ratios.get("current_ratio", 1.0) or 0.0
    debt_to_equity = ratios.get("debt_to_equity", 0.0) or 0.0
    margin = ratios.get("net_profit_margin", 0.0) or 0.0

    liquidity = 100.0 / (1.0 + max(current_ratio, 0.0))
    liquidity = clamp100(liquidity + 15.0 * lex.get("lex_liquidity", 0.0))

    de = max(debt_to_equity, 0.0)
    leverage = 100.0 * de / (1.0 + de)
    leverage = clamp100(leverage + 15.0 * lex.get("lex_leverage", 0.0))

    profitability = 100.0 / (1.0 + np.exp(20.0 * margin))
    profitability = clamp100(profitability + 15.0 * lex.get("lex_profitability", 0.0))

    operational = clamp100(
        20.0 * lex.get("lex_legal", 0.0)
        + 25.0 * lex.get("lex_going_concern", 0.0)
        + 10.0 * lex.get("risk_sentence_count", 0.0)
    )
    return {
        "liquidity": liquidity,
        "leverage": leverage,
        "profitability": profitability,
        "operational": operational,
    }


def term_caption(key: str) -> None:
    """Render the canonical label + one-line meaning for a probability term."""
    label, meaning = PROBABILITY_TERMS[key]
    st.caption(f"**{label}:** {meaning}")


def manual_inputs():
    st.subheader("Financial inputs (Rs. crore)")
    financials = {}
    cols = st.columns(3)
    for i, (key, label, default) in enumerate(FINANCIAL_FIELDS):
        with cols[i % 3]:
            financials[key] = st.number_input(label, value=float(default), key=f"fin_{key}")
    text = st.text_area(
        "Report excerpt (optional)",
        height=150,
        key="manual_text",
        placeholder="Paste a few sentences from the annual report / MD&A ...",
    )
    return financials, text or ""


def pdf_inputs(median_profile):
    uploaded = st.file_uploader("pdf", type=["pdf"], help="Upload a PDF annual-report excerpt")
    if uploaded is None:
        st.info("Upload a PDF annual-report excerpt to begin.")
        st.stop()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name
    try:
        try:
            report = extract_report(tmp_path)
        except Exception as exc:
            st.error(f"Could not extract text from the uploaded PDF: {exc}")
            st.stop()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    extracted = report["text"]
    for w in report["warnings"]:
        st.warning(w)
    if report["tables"]:
        st.info(f"Detected {len(report['tables'])} table(s); their cell text was "
                "included in the extraction.")
    if not extracted.strip():
        st.warning("No text could be extracted from this PDF - analysis will use financial numbers only.")
    text = st.text_area("Extracted report text (editable)", value=extracted, height=200, key="pdf_text")

    st.subheader("Financial inputs (Rs. crore) - defaults from dataset median")
    financials = {}
    cols = st.columns(3)
    for i, (key, label, default) in enumerate(FINANCIAL_FIELDS):
        with cols[i % 3]:
            financials[key] = st.number_input(
                label, value=float(median_profile.get(key, default)), key=f"pdf_{key}"
            )
    return financials, text or ""


def show_proba_bars(p_a, p_b, p_a_cal, p_b_cal):
    """Side-by-side raw vs calibrated model probabilities."""
    st.subheader(PROBABILITY_TERMS["model_probability"][0])
    term_caption("model_probability")
    proba_df = pd.DataFrame(
        {"Model A (ratios)": p_a, "Model B (ratios+text)": p_b}, index=LABELS
    )
    st.bar_chart(proba_df)

    if p_a_cal is not None and p_b_cal is not None:
        st.subheader(PROBABILITY_TERMS["calibrated_probability"][0])
        term_caption("calibrated_probability")
        cal_df = pd.DataFrame(
            {"Model A (ratios)": p_a_cal, "Model B (ratios+text)": p_b_cal},
            index=LABELS,
        )
        st.bar_chart(cal_df)


def show_results(financials, text, artifacts):
    x_a, x_b, names_b, ratios, lex = build_feature_rows(financials, text, artifacts)

    p_a = ordered_proba(artifacts["model_a"], x_a)
    p_b = ordered_proba(artifacts["model_b"], x_b)
    cal_a = artifacts.get("model_a_calibrated")
    cal_b = artifacts.get("model_b_calibrated")
    p_a_cal = ordered_proba(cal_a, x_a) if cal_a is not None else None
    p_b_cal = ordered_proba(cal_b, x_b) if cal_b is not None else None
    # Classification uses raw model probabilities (argmax); calibration only
    # rescales the displayed probabilities.
    level_a = LABELS[int(np.argmax(p_a))]
    level_b = LABELS[int(np.argmax(p_b))]
    score_a = risk_score_from_proba(p_a)
    score_b = risk_score_from_proba(p_b)

    st.subheader(PROBABILITY_TERMS["risk_classification"][0])
    term_caption("risk_classification")
    color = {"Low": "green", "Medium": "orange", "High": "red"}[level_b]
    st.markdown(f"## :{color}[{level_b} risk]")

    st.subheader(PROBABILITY_TERMS["risk_score"][0])
    term_caption("risk_score")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Model A - financial ratios only", f"{score_a:.1f} / 100",
                  help="100 × (0·P(Low) + 0.5·P(Medium) + 1·P(High)) from Model A")
    with col_b:
        st.metric("Model B - ratios + report-text signals", f"{score_b:.1f} / 100",
                  help="100 × (0·P(Low) + 0.5·P(Medium) + 1·P(High)) from Model B")

    show_proba_bars(p_a, p_b, p_a_cal, p_b_cal)

    st.subheader(PROBABILITY_TERMS["heuristic_indicators"][0])
    term_caption("heuristic_indicators")
    for name, value in heuristic_category_scores(ratios, lex).items():
        st.write(f"{name.capitalize()}: {value:.0f} / 100")
        st.progress(int(round(value)))

    st.subheader("Why this prediction? (Model B, SHAP)")
    st.caption("Top features by |SHAP contribution| for the predicted class. "
               "Red = increases risk, green = reduces risk. "
               "TF-IDF word features are shown but never narrated — a single "
               "word's contribution is not interpretable in isolation.")
    try:
        reasons = explain_prediction(artifacts["model_b"], names_b, x_b, top_n=8)
    except Exception as exc:
        st.warning(f"Could not compute SHAP explanations: {exc}")
        reasons = []
    if reasons:
        st.dataframe(
            pd.DataFrame(reasons)[["display", "family", "value", "contribution", "direction"]],
            use_container_width=True,
        )
        st.pyplot(shap_waterfall_figure(reasons,
                                        title="SHAP contributions — Model B prediction"))
        st.write(reasons_to_text(reasons, level_b))


def analyze_tab(artifacts):
    mode = st.sidebar.radio("Input mode", ["Enter financial numbers", "Upload PDF report"])
    if mode == "Enter financial numbers":
        financials, text = manual_inputs()
    else:
        financials, text = pdf_inputs(artifacts["median_profile"])

    if st.button("Analyze Risk", type="primary"):
        show_results(financials, text, artifacts)


def confusion_matrix_figure(cm, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=10)
    ax.set_title("Confusion matrix (rows = true, cols = predicted)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def comparison_tab(artifacts):
    st.header("Model comparison")
    st.caption("All classifiers trained on the same stratified 80/20 split "
               "(random_state=42) and evaluated with the same protocol, so "
               "differences isolate the classifier and the feature set. "
               "Metrics are on SYNTHETIC data — they validate the pipeline, "
               "not real-world performance.")
    metrics = artifacts["metrics"]
    models = metrics.get("models")
    if not models:
        st.warning("metrics.json predates the upgraded evaluation — "
                   "re-run `python src/train.py` to regenerate it.")
        st.stop()

    order = ["rf_a", "lr_a", "xgb_a", "rf_b", "lr_b", "xgb_b"]
    order = [m for m in order if m in models]
    rows = ["accuracy", "f1_macro", "f1_weighted", "high_risk_recall",
            "high_risk_precision", "roc_auc_ovr", "ece"]
    table = pd.DataFrame(
        {MODEL_PRETTY.get(m, m): [models[m][r] for r in rows] for m in order},
        index=rows,
    ).round(4)
    st.subheader("Classifier × feature-set comparison (held-out test set)")
    st.dataframe(table, use_container_width=True)

    st.info(
        "On this synthetic dataset, **Model B (ratios + text) does not beat "
        "Model A (ratios only)** — e.g. RF accuracy 0.9583 → 0.9458. This is "
        "reported as-is. The label rule gives text only ~10/100 points, and "
        "the 500 TF-IDF terms spread ~39% of the RandomForest's importance "
        "mass over mostly noisy terms (see VIVA.md for the full investigation). "
        "With real report text the question stays open."
    )

    st.subheader("Per-model detail")
    for m in order:
        with st.expander(MODEL_PRETTY.get(m, m)):
            d = models[m]
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(confusion_matrix_figure(
                    d["confusion_matrix"], d["confusion_matrix_labels"]))
            with c2:
                per = pd.DataFrame(d["per_class"]).T.round(4)
                st.dataframe(per, use_container_width=True)
            st.text(d["classification_report"])

    cal = metrics.get("calibration")
    if cal:
        st.subheader("Probability calibration (RandomForest, sigmoid, 5-fold CV)")
        st.caption("Calibrated probabilities are exposed separately in the "
                   "Analyze tab; SHAP explanations still use the base model.")
        cal_df = pd.DataFrame(cal).T.round(4)
        st.dataframe(cal_df, use_container_width=True)

    st.subheader("Temporal (out-of-time) validation — framework, not yet executable")
    st.caption(describe_temporal_plan())


def trend_tab(artifacts):
    st.header("Risk trend")
    st.warning(
        "The shipped dataset is a **single cross-section** (one row per company, "
        "no year column), so no historical trend can be computed from it — and "
        "none is fabricated here. This tab is the infrastructure: supply real "
        "per-year financials below and RiskLabs will score each year with "
        "Model A and classify the trajectory."
    )
    st.caption("Add one row per year for a single company (financials in Rs. crore).")
    if "trend_rows" not in st.session_state:
        st.session_state.trend_rows = []

    with st.form("trend_add_year"):
        cols = st.columns(4)
        year = cols[0].number_input("Year", value=2024, step=1, format="%d")
        fin = {}
        for i, (key, label, default) in enumerate(FINANCIAL_FIELDS):
            with cols[(i + 1) % 4]:
                fin[key] = st.number_input(label, value=float(default),
                                           key=f"trend_{key}")
        if st.form_submit_button("Add year"):
            row = {"year": int(year)}
            row.update(fin)
            st.session_state.trend_rows = [
                r for r in st.session_state.trend_rows if r["year"] != int(year)
            ] + [row]
            st.rerun()

    rows = sorted(st.session_state.trend_rows, key=lambda r: r["year"])
    if len(rows) < 2:
        st.info("Add at least 2 different years to compute a trend.")
        return
    if st.button("Clear all years"):
        st.session_state.trend_rows = []
        st.rerun()

    try:
        profile = yearly_risk_profile(artifacts["model_a"], pd.DataFrame(rows))
    except Exception as exc:
        st.error(f"Could not score the yearly observations: {exc}")
        return
    st.subheader("Year → Risk Score → Risk Class")
    st.dataframe(profile_to_yearly_table(profile), use_container_width=True,
                 hide_index=True)
    st.line_chart(profile.set_index("year")["risk_score"])
    trend = trend_direction(profile)
    st.metric("Trajectory", trend["direction"].capitalize(),
              delta=f"{trend['delta']:+.1f} points over {trend['n_years']} years",
              help=f"OLS slope {trend['slope_per_year']:+.2f} score-points/year; "
                   f"|slope| ≤ 2.0 counts as stable.")


def main():
    st.set_page_config(page_title="RiskLabs - Explainable Financial Risk", layout="wide")
    st.title("RiskLabs - Explainable Financial-Risk Prototype")
    st.caption("Synthetic-data demo for a college project. Educational use only - not financial advice.")

    if _IMPORT_ERROR is not None:
        st.error(
            "Could not import the project's src/ modules "
            f"({type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}). "
            "They may still be under construction - please try again later."
        )
        st.stop()

    try:
        artifacts = load_artifacts()
    except Exception as exc:
        st.error(
            "Model files are missing or could not be loaded "
            f"({type(exc).__name__}: {exc}). Training may still be running - "
            "run `python src/train.py` from the project root, then reload this page."
        )
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Analyze company", "Model comparison", "Risk trend"])
    with tab1:
        analyze_tab(artifacts)
    with tab2:
        comparison_tab(artifacts)
    with tab3:
        trend_tab(artifacts)

    st.divider()
    st.caption(
        "**Disclaimer:** RiskLabs is a synthetic-data prototype built for a college project. "
        "For educational use only - not financial advice, and not a prediction of real-world outcomes."
    )


if __name__ == "__main__":
    main()
