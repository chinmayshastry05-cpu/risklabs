"""Multi-year risk-trend infrastructure for RiskLabs.

Given per-year observations for ONE company (each with the 9 raw financial
columns, and optionally a report_text), this module:

  - builds the Model A feature row for each year,
  - scores each year with a fitted classifier (raw model probabilities),
  - computes the 0–100 risk score per year,
  - classifies the trajectory as increasing / stable / decreasing via a
    least-squares slope on the yearly scores with a stability band.

STATUS: the shipped synthetic dataset is a single cross-section (one row per
company, no year column), so this module has NO DATA to run on today. It is
infrastructure: the day real longitudinal company data exists, the dashboard's
"Risk trend" tab can call yearly_risk_profile() directly. Nothing here
fabricates historical data — a caller must supply real per-year observations.

trend_direction() uses an ordinary least-squares slope on yearly scores:
  slope > +STABLE_BAND  → "increasing"
  slope < -STABLE_BAND  → "decreasing"
  otherwise             → "stable"
STABLE_BAND = 2.0 score-points per year: a deliberate, documented tolerance so
that small year-to-year wobble is not presented as a trend.
"""

import numpy as np
import pandas as pd

from src.evaluate import ordered_proba
from src.explain import risk_score_from_proba
from src.features import RATIO_FEATURES, compute_ratios

RAW_FINANCIAL_COLS = [
    "revenue", "net_profit", "total_assets", "total_liabilities",
    "total_debt", "equity", "current_assets", "current_liabilities",
    "operating_cash_flow",
]
LABELS = ["Low", "Medium", "High"]
STABLE_BAND = 2.0  # score points per year


def yearly_risk_profile(model, yearly_df: pd.DataFrame, year_col: str = "year") -> pd.DataFrame:
    """Score every year of one company's history with a fitted classifier.

    yearly_df must contain *year_col* and the 9 RAW_FINANCIAL_COLS. Returns a
    DataFrame with columns: year, risk_score, risk_class, p_low, p_medium, p_high.
    Raises ValueError on missing columns, duplicate years, or < 2 years.
    """
    missing = {year_col} | set(RAW_FINANCIAL_COLS)
    absent = sorted(missing - set(yearly_df.columns))
    if absent:
        raise ValueError(f"yearly observations are missing columns: {absent}")
    if yearly_df[year_col].duplicated().any():
        raise ValueError("yearly observations contain duplicate years.")
    if len(yearly_df) < 2:
        raise ValueError("a risk trend needs at least 2 yearly observations.")

    ratios = compute_ratios(yearly_df)[RATIO_FEATURES].to_numpy(dtype=float)
    rows = []
    for i, year in enumerate(yearly_df[year_col]):
        x = ratios[i].reshape(1, -1)
        proba = ordered_proba(model, x)
        score = risk_score_from_proba(proba)
        rows.append({
            "year": int(year),
            "risk_score": round(float(score), 1),
            "risk_class": LABELS[int(np.argmax(proba))],
            "p_low": round(float(proba[0]), 4),
            "p_medium": round(float(proba[1]), 4),
            "p_high": round(float(proba[2]), 4),
        })
    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    return out


def trend_direction(profile_df: pd.DataFrame, stable_band: float = STABLE_BAND) -> dict:
    """Classify a yearly risk profile as increasing / stable / decreasing.

    Returns {"direction": str, "slope_per_year": float, "n_years": int,
             "first_score": float, "last_score": float, "delta": float}.
    The slope is OLS on (year_index, risk_score). A trend is only reported
    when |slope| exceeds *stable_band* score-points per year.
    """
    if len(profile_df) < 2:
        raise ValueError("need at least 2 yearly observations for a trend.")
    x = np.arange(len(profile_df), dtype=float)
    y = profile_df["risk_score"].to_numpy(dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    if slope > stable_band:
        direction = "increasing"
    elif slope < -stable_band:
        direction = "decreasing"
    else:
        direction = "stable"
    return {
        "direction": direction,
        "slope_per_year": round(slope, 2),
        "n_years": int(len(profile_df)),
        "first_score": float(y[0]),
        "last_score": float(y[-1]),
        "delta": round(float(y[-1] - y[0]), 1),
    }


def profile_to_yearly_table(profile_df: pd.DataFrame) -> pd.DataFrame:
    """Year → Risk Score → Risk Class view for display in the dashboard."""
    return profile_df[["year", "risk_score", "risk_class"]].copy()
