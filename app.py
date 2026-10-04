"""RiskLabs Streamlit dashboard - explainable financial-risk prototype.

Run from the project root with:  streamlit run app.py

All inputs here are synthetic / user-provided demo data. This is a college
prototype for educational use only - not financial advice.
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
    from src.explain import explain_prediction, reasons_to_text, risk_score_from_proba
    from src.features import RATIO_FEATURES, compute_ratios_dict
    from src.nlp import LEXICON_FEATURES, extract_lexicon_features
    from src.pdf_parse import extract_text_from_pdf

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
    return {
        "model_a": joblib.load(base / "model_a.pkl"),
        "model_b": joblib.load(base / "model_b.pkl"),
        "vectorizer": joblib.load(base / "tfidf_vectorizer.pkl"),
        "feature_names": json.loads((base / "feature_names.json").read_text()),
        "median_profile": json.loads((base / "median_profile.json").read_text()),
        "metrics": json.loads((base / "metrics.json").read_text()),
    }


def ordered_proba(model, x_row):
    """Return predict_proba reordered to [Low, Medium, High] (models store
    classes alphabetically as High/Low/Medium)."""
    proba = model.predict_proba(x_row)[0]
    idx = [list(model.classes_).index(label) for label in LABELS]
    return proba[idx]


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
            extracted = extract_text_from_pdf(tmp_path)
        except Exception as exc:
            st.error(f"Could not extract text from the uploaded PDF: {exc}")
            st.stop()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
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


def show_results(financials, text, artifacts):
    x_a, x_b, names_b, ratios, lex = build_feature_rows(financials, text, artifacts)

    p_a = ordered_proba(artifacts["model_a"], x_a)
    p_b = ordered_proba(artifacts["model_b"], x_b)
    level_a = LABELS[int(np.argmax(p_a))]
    level_b = LABELS[int(np.argmax(p_b))]
    score_a = risk_score_from_proba(p_a)
    score_b = risk_score_from_proba(p_b)

    color = {"Low": "green", "Medium": "orange", "High": "red"}[level_b]
    st.markdown(f"## :{color}[{level_b} risk] - score {score_b:.1f} / 100")

    st.subheader("Model A vs Model B")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Model A - financial ratios only**")
        st.metric("Predicted level", level_a)
        st.metric("Risk score", f"{score_a:.1f} / 100")
    with col_b:
        st.markdown("**Model B - ratios + report-text signals**")
        st.metric("Predicted level", level_b)
        st.metric("Risk score", f"{score_b:.1f} / 100")

    st.subheader("Risk-category sub-scores (heuristic)")
    st.caption(
        "Heuristic 0-100 sub-scores derived transparently from ratios + lexicon hits - "
        "not model output. Higher = riskier."
    )
    for name, value in heuristic_category_scores(ratios, lex).items():
        st.write(f"{name.capitalize()}: {value:.0f} / 100")
        st.progress(int(round(value)))

    st.subheader("Why this prediction? (Model B, SHAP)")
    try:
        reasons = explain_prediction(artifacts["model_b"], names_b, x_b, top_n=6)
    except Exception as exc:
        st.warning(f"Could not compute SHAP explanations: {exc}")
        reasons = []
    if reasons:
        st.dataframe(
            pd.DataFrame(reasons)[["display", "value", "contribution", "direction"]],
            use_container_width=True,
        )
        chart_df = pd.DataFrame(
            {"contribution": [r["contribution"] for r in reasons]},
            index=[r["display"] for r in reasons],
        )
        st.bar_chart(chart_df)
        st.write(reasons_to_text(reasons, level_b))


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

    mode = st.sidebar.radio("Input mode", ["Enter financial numbers", "Upload PDF report"])

    if mode == "Enter financial numbers":
        financials, text = manual_inputs()
    else:
        financials, text = pdf_inputs(artifacts["median_profile"])

    if st.button("Analyze Risk", type="primary"):
        show_results(financials, text, artifacts)

    st.divider()
    st.caption(
        "**Disclaimer:** RiskLabs is a synthetic-data prototype built for a college project. "
        "For educational use only - not financial advice, and not a prediction of real-world outcomes."
    )


if __name__ == "__main__":
    main()
