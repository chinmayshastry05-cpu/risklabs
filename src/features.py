"""Financial ratio feature engineering for RiskLabs.

Defines the canonical set of ratio features (RATIO_FEATURES) computed from the
nine raw financial columns, plus helpers to compute them for a whole DataFrame
or a single company dict. All divisions are guarded against divide-by-zero with
epsilon = 1e-6 (a denominator whose absolute value is below epsilon is replaced
by +/-epsilon, preserving its sign).
"""

import numpy as np
import pandas as pd

EPS = 1e-6

RATIO_FEATURES = [
    "current_ratio",
    "debt_to_equity",
    "debt_to_assets",
    "net_profit_margin",
    "roa",
    "roe",
    "ocf_to_debt",
]


def _safe_div(num, den):
    """Element-wise num / den with zero-guard on the denominator."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    guarded = np.where(np.abs(den) < EPS, np.where(den < 0, -EPS, EPS), den)
    return num / guarded


def _safe_div_scalar(a, b):
    """Scalar num / den with zero-guard on the denominator."""
    a = float(a)
    b = float(b)
    if abs(b) < EPS:
        b = -EPS if b < 0 else EPS
    return a / b


def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the RATIO_FEATURES columns added.

    Expected raw columns: revenue, net_profit, total_assets, total_liabilities,
    total_debt, equity, current_assets, current_liabilities, operating_cash_flow.
    """
    out = df.copy()
    out["current_ratio"] = _safe_div(out["current_assets"], out["current_liabilities"])
    out["debt_to_equity"] = _safe_div(out["total_debt"], out["equity"])
    out["debt_to_assets"] = _safe_div(out["total_debt"], out["total_assets"])
    out["net_profit_margin"] = _safe_div(out["net_profit"], out["revenue"])
    out["roa"] = _safe_div(out["net_profit"], out["total_assets"])
    out["roe"] = _safe_div(out["net_profit"], out["equity"])
    out["ocf_to_debt"] = _safe_div(out["operating_cash_flow"], out["total_debt"])
    return out


def compute_ratios_dict(d: dict) -> dict:
    """Compute the RATIO_FEATURES for a single company given as a dict.

    Returns a dict {feature_name: value}. Used by app.py for one-off inputs.
    """
    return {
        "current_ratio": _safe_div_scalar(d["current_assets"], d["current_liabilities"]),
        "debt_to_equity": _safe_div_scalar(d["total_debt"], d["equity"]),
        "debt_to_assets": _safe_div_scalar(d["total_debt"], d["total_assets"]),
        "net_profit_margin": _safe_div_scalar(d["net_profit"], d["revenue"]),
        "roa": _safe_div_scalar(d["net_profit"], d["total_assets"]),
        "roe": _safe_div_scalar(d["net_profit"], d["equity"]),
        "ocf_to_debt": _safe_div_scalar(d["operating_cash_flow"], d["total_debt"]),
    }
