"""SHAP-based explanations for RiskLabs models.

explain_prediction() explains a single row for the *predicted*
(highest-probability) class, using shap.TreeExplainer on a fitted
RandomForestClassifier. Input rows are dense numpy arrays
(train.py densifies the TF-IDF block with .toarray()).

Each reason now carries a "family" tag — "financial", "lexicon" or
"tfidf" — so the UI can separate learned ratio effects from text signals.

factor_sentence() produces ONE traceable sentence per top reason, e.g.
"High leverage (debt-to-equity = 2.40) increased predicted risk." Every
sentence is conditioned on the actual feature value AND the sign of its SHAP
contribution; when no template matches, no sentence is invented.

shap_waterfall_figure() renders a matplotlib diverging bar chart of the
reasons (positive = increases risk, negative = reduces risk), annotated with
the feature family.
"""

import numpy as np
import shap

#: Human-readable names for engineered features.
#: TF-IDF terms are not listed individually; display_name() renders them as
#: `word/phrase "…`.
FEATURE_DISPLAY = {
    "current_ratio": "Current ratio",
    "debt_to_equity": "Debt-to-equity ratio",
    "debt_to_assets": "Debt-to-assets ratio",
    "net_profit_margin": "Net profit margin",
    "roa": "Return on assets (ROA)",
    "roe": "Return on equity (ROE)",
    "ocf_to_debt": "Operating cash flow to debt",
    "lex_total": "Overall risk language in report",
    "lex_liquidity": "Liquidity risk language in report",
    "lex_leverage": "Leverage risk language in report",
    "lex_profitability": "Profitability risk language in report",
    "lex_legal": "Legal/regulatory risk language in report",
    "lex_going_concern": "Going-concern risk language in report",
    "lex_positive": "Positive/strength language in report",
    "risk_sentence_count": "Sentences with risk language",
}

CLASS_NAMES = ["Low", "Medium", "High"]

FINANCIAL_FEATURES = {
    "current_ratio", "debt_to_equity", "debt_to_assets",
    "net_profit_margin", "roa", "roe", "ocf_to_debt",
}
LEXICON_FEATURES_SET = {
    "lex_total", "lex_liquidity", "lex_leverage", "lex_profitability",
    "lex_legal", "lex_going_concern", "lex_positive", "risk_sentence_count",
}

#: Canonical UI labels that distinguish what each number IS. Used by app.py
#: so the dashboard never presents a heuristic as a calibrated probability.
PROBABILITY_TERMS = {
    "model_probability": (
        "Model Probability",
        "The classifier's estimated probability for each risk class. "
        "A model's raw probability — use the calibrated variant for display.",
    ),
    "calibrated_probability": (
        "Calibrated Model Probability",
        "Model probability adjusted (sigmoid, 5-fold CV) so that a stated "
        "probability better matches observed frequencies on the test set.",
    ),
    "risk_classification": (
        "Risk Classification",
        "The single risk class with the highest model probability "
        "(Low / Medium / High). A decision, not a probability.",
    ),
    "risk_score": (
        "Risk Score (0–100)",
        "100 × (0·P(Low) + 0.5·P(Medium) + 1·P(High)). A summary of the "
        "model's probability distribution — NOT a calibrated real-world "
        "default probability.",
    ),
    "heuristic_indicators": (
        "Heuristic Risk Indicators",
        "Transparent 0–100 sub-scores (liquidity, leverage, profitability, "
        "operational) computed from documented ratio transforms and lexicon "
        "counts. NOT model output and NOT probabilities.",
    ),
}


def display_name(feature: str) -> str:
    """Human name for a feature; TF-IDF terms render as `word/phrase "…"`."""
    if feature in FEATURE_DISPLAY:
        return FEATURE_DISPLAY[feature]
    return f'word/phrase "{feature}"'


def feature_family(feature: str) -> str:
    """Which feature family a reason belongs to: financial / lexicon / tfidf."""
    if feature in FINANCIAL_FEATURES:
        return "financial"
    if feature in LEXICON_FEATURES_SET:
        return "lexicon"
    return "tfidf"


def _format_value(value: float) -> str:
    v = float(value)
    if v == 0.0:
        return "0"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:,.2f}"
    return f"{v:.3f}"


def explain_prediction(model, feature_names, x_row_2d: np.ndarray, top_n: int = 6):
    """SHAP reasons for the model's predicted class on a single row.

    Returns a list of dicts with keys:
    {"feature", "display", "family", "value", "contribution", "direction"},
    where "direction" is "increases risk" when the SHAP contribution is > 0
    and "reduces risk" otherwise. Sorted by |contribution|, top_n entries.
    """
    x = np.asarray(x_row_2d, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    proba = model.predict_proba(x)[0]
    predicted_class = int(np.argmax(proba))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)

    # shap >= 0.40 returns (n_samples, n_features, n_classes); older versions
    # return a list of per-class arrays.
    if isinstance(shap_values, list):
        contributions = np.asarray(shap_values[predicted_class])[0]
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            contributions = arr[0, :, predicted_class]
        elif arr.ndim == 2:
            contributions = arr[0]  # single-class model
        else:
            contributions = arr.ravel()

    order = np.argsort(np.abs(contributions))[::-1][:top_n]
    reasons = []
    for idx in order:
        i = int(idx)
        contribution = float(contributions[i])
        feature = str(feature_names[i])
        reasons.append({
            "feature": feature,
            "display": display_name(feature),
            "family": feature_family(feature),
            "value": float(x[0, i]),
            "contribution": contribution,
            "direction": "increases risk" if contribution > 0 else "reduces risk",
        })
    return reasons


def _ratio_sentence(display: str, value: float, direction: str) -> str | None:
    """Traceable one-liners for engineered ratio/lexicon features.

    Returns None when the (value, direction) combination has no defensible
    reading — in which case the caller must NOT invent a sentence.
    """
    up = direction == "increases risk"
    v = _format_value(value)
    table = {
        "Debt-to-equity ratio": (
            (lambda val: val > 1.5, f"High leverage (debt-to-equity = {v})"),
            (lambda val: val < 0.5, f"Low leverage (debt-to-equity = {v})"),
        ),
        "Current ratio": (
            (lambda val: val < 1.0, f"Weak liquidity (current ratio = {v})"),
            (lambda val: val > 2.0, f"Comfortable liquidity (current ratio = {v})"),
        ),
        "Net profit margin": (
            (lambda val: val < 0.0, f"Loss-making operations (net margin = {v})"),
            (lambda val: val > 0.10, f"Healthy profitability (net margin = {v})"),
        ),
        "Return on assets (ROA)": (
            (lambda val: val < 0.0, f"Negative returns on assets (ROA = {v})"),
            (lambda val: val > 0.08, f"Strong returns on assets (ROA = {v})"),
        ),
        "Operating cash flow to debt": (
            (lambda val: val < 0.15, f"Weak cash-flow coverage of debt (OCF/debt = {v})"),
            (lambda val: val > 0.50, f"Strong cash-flow coverage of debt (OCF/debt = {v})"),
        ),
        "Debt-to-assets ratio": (
            (lambda val: val > 0.60, f"Heavy debt load relative to assets (debt/assets = {v})"),
            (lambda val: val < 0.25, f"Light debt load relative to assets (debt/assets = {v})"),
        ),
        "Return on equity (ROE)": (
            (lambda val: val < 0.0, f"Negative returns on equity (ROE = {v})"),
            (lambda val: val > 0.12, f"Strong returns on equity (ROE = {v})"),
        ),
        "Leverage risk language in report": (
            (lambda val: val > 0, f"Leverage-related risk language in the report (score {v})"),
            (lambda val: val == 0, None),
        ),
        "Liquidity risk language in report": (
            (lambda val: val > 0, f"Liquidity-related risk language in the report (score {v})"),
            (lambda val: val == 0, None),
        ),
        "Profitability risk language in report": (
            (lambda val: val > 0, f"Profitability-related risk language in the report (score {v})"),
            (lambda val: val == 0, None),
        ),
        "Legal/regulatory risk language in report": (
            (lambda val: val > 0, f"Legal/regulatory risk language in the report (score {v})"),
            (lambda val: val == 0, None),
        ),
        "Going-concern risk language in report": (
            (lambda val: val > 0, f"Going-concern risk language in the report (score {v})"),
            (lambda val: val == 0, None),
        ),
        # lex_positive is NEGATIVE when healthy language is present (negative
        # lexicon weights), and 0 when no positive phrases were found.
        "Positive/strength language in report": (
            (lambda val: val == 0, "Absence of positive/strength language in the report"),
            (lambda val: val < 0, f"Positive/strength language in the report (score {v})"),
        ),
        "Sentences with risk language": (
            (lambda val: val >= 2, f"Risk language in {int(value)} report sentences"),
            (lambda val: val == 0, "Absence of risk language in the report"),
        ),
        "Overall risk language in report": (
            (lambda val: val > 3, f"Strong overall risk language in the report (score {v})"),
            (lambda val: val <= 0, "Little or no risk language in the report"),
        ),
    }
    entry = table.get(display)
    if entry is None:
        return None
    cond_up, phrase_up = entry[0]
    cond_down, phrase_down = entry[1]
    if up and cond_up(value) and phrase_up:
        return f"{phrase_up} increased predicted risk."
    if not up and cond_down(value) and phrase_down:
        return f"{phrase_down} reduced predicted risk."
    return None


def factor_sentence(reason: dict) -> str | None:
    """One traceable sentence for a single SHAP reason, or None.

    TF-IDF terms never get sentences: a raw word's SHAP contribution is not
    interpretable in isolation, so inventing prose for it would be dishonest.
    """
    if reason.get("family") == "tfidf":
        return None
    return _ratio_sentence(reason["display"], reason["value"], reason["direction"])


def reasons_to_text(reasons, risk_level: str) -> str:
    """Plain-English paragraph from a reasons list.

    Every factual claim traces to a SHAP reason (feature value + contribution
    sign). TF-IDF terms are excluded from prose (see factor_sentence).
    """
    if not reasons:
        return (f"The model rates this company as {risk_level} risk, but no "
                "feature-level reasons were available for this prediction.")

    sentences = [f"The model rates this company as {risk_level} risk."]
    claimed = 0
    for r in sorted(reasons, key=lambda d: -abs(d["contribution"])):
        s = factor_sentence(r)
        if s and claimed < 3:
            sentences.append(s)
            claimed += 1

    fin_up = sum(1 for r in reasons
                 if r["family"] == "financial" and r["direction"] == "increases risk")
    text_up = sum(1 for r in reasons
                  if r["family"] in ("lexicon", "tfidf") and r["direction"] == "increases risk")
    if fin_up or text_up:
        parts = []
        if fin_up:
            parts.append(f"{fin_up} financial-ratio factor(s)")
        if text_up:
            parts.append(f"{text_up} report-text factor(s)")
        sentences.append(
            "Of the top risk-increasing factors, " + " and ".join(parts) +
            " pushed the prediction upward."
        )
    else:
        sentences.append("No individual factor strongly pushed the prediction upward.")

    sentences.append(
        "This is an explainable prototype trained on synthetic data for "
        "educational purposes — not financial advice."
    )
    return " ".join(sentences)


def shap_waterfall_figure(reasons, title: str = "SHAP contributions by feature"):
    """Matplotlib diverging bar chart of SHAP reasons.

    Positive bars (right, red) increase predicted risk; negative bars (left,
    green) reduce it. Bars are annotated with the feature family.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(reasons, key=lambda r: r["contribution"])
    labels = [f"{r['display']} [{r['family']}]" for r in ordered]
    values = [r["contribution"] for r in ordered]
    colors = ["#c0392b" if v > 0 else "#27ae60" for v in values]

    fig, ax = plt.subplots(figsize=(9, max(3.2, 0.55 * len(ordered) + 1.2)))
    y = np.arange(len(ordered))
    ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP contribution to predicted-class probability "
                  "(+ increases risk, − reduces risk)")
    ax.set_title(title, fontsize=11, pad=10)
    ax.text(0.99, 0.02, "red = increases risk · green = reduces risk",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            style="italic")
    fig.tight_layout()
    return fig


def risk_score_from_proba(proba) -> float:
    """0–100 risk score: 100 * (0*pLow + 0.5*pMedium + 1*pHigh).

    proba must be ordered [P(Low), P(Medium), P(High)], matching the
    {"Low": 0, "Medium": 1, "High": 2} label map used in training.

    NOTE: this is a weighted summary of the model's probability distribution,
    NOT a calibrated real-world default probability.
    """
    p = np.asarray(proba, dtype=float).ravel()
    if p.shape[0] != 3:
        raise ValueError(f"expected 3 class probabilities, got shape {p.shape}")
    return float(100.0 * (0.0 * p[0] + 0.5 * p[1] + 1.0 * p[2]))
