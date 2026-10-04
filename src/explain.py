"""SHAP-based explanations for RiskLabs models.

explain_prediction() explains a single row for the *predicted*
(highest-probability) class, using shap.TreeExplainer on a fitted
RandomForestClassifier. Input rows are dense numpy arrays
(train.py densifies the TF-IDF block with .toarray()).
"""

import numpy as np
import shap

#: Human-readable names for engineered features.
#: TF-IDF terms are not listed individually; display_name() renders them as
#: `word/phrase "\u2026"`.
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


def display_name(feature: str) -> str:
    """Human name for a feature; TF-IDF terms render as `word/phrase "\u2026"`."""
    if feature in FEATURE_DISPLAY:
        return FEATURE_DISPLAY[feature]
    return f'word/phrase "{feature}"'


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
    {"feature", "display", "value", "contribution", "direction"},
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
            "value": float(x[0, i]),
            "contribution": contribution,
            "direction": "increases risk" if contribution > 0 else "reduces risk",
        })
    return reasons


def reasons_to_text(reasons, risk_level: str) -> str:
    """Plain-English 3\u20135 sentence paragraph from a reasons list."""
    if not reasons:
        return (f"The model rates this company as {risk_level} risk, but no "
                "feature-level reasons were available for this prediction.")

    risk_up = [r for r in reasons if r["direction"] == "increases risk"]
    risk_down = [r for r in reasons if r["direction"] == "reduces risk"]

    s1 = f"The model rates this company as {risk_level} risk."
    if risk_up:
        top = risk_up[:3]
        detail = "; ".join(
            f'{r["display"]} ({_format_value(r["value"])})' for r in top)
        s2 = (f"The strongest risk-increasing factors were {detail}. "
              "These push the prediction toward a higher risk category.")
    else:
        s2 = "No individual factor strongly pushed the prediction upward."

    if risk_down:
        top = risk_down[:2]
        detail = "; ".join(
            f'{r["display"]} ({_format_value(r["value"])})' for r in top)
        s3 = f"Factors pointing the other way include {detail}, which reduce risk."
    else:
        s3 = "Few mitigating factors were detected, so risk signals dominate."

    s4 = ("This is an explainable prototype trained on synthetic data for "
          "educational purposes \u2014 not financial advice.")
    return " ".join([s1, s2, s3, s4])


def risk_score_from_proba(proba) -> float:
    """0\u2013100 risk score: 100 * (0*pLow + 0.5*pMedium + 1*pHigh).

    proba must be ordered [P(Low), P(Medium), P(High)], matching the
    {"Low": 0, "Medium": 1, "High": 2} label map used in training.
    """
    p = np.asarray(proba, dtype=float).ravel()
    if p.shape[0] != 3:
        raise ValueError(f"expected 3 class probabilities, got shape {p.shape}")
    return float(100.0 * (0.0 * p[0] + 0.5 * p[1] + 1.0 * p[2]))
