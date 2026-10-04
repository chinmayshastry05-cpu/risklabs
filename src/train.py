"""RiskLabs training script.

Trains and compares two RandomForest risk classifiers:

  Model A — 7 financial ratio features only.
  Model B — 7 ratio features + 8 lexicon text features + 500 TF-IDF features
            (TF-IDF input densified with .toarray(); scale is small).

Usage: run from the project root as
    python src/train.py        (or: python3 src/train.py)

DATA NOTE: this trains on SYNTHETIC data (data/risklabs_synthetic.csv).
Metrics printed and saved here validate the pipeline only — they do not
describe performance on any real company.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

# Make `src.*` importable when run as `python src/train.py` from the project root
# (the script's own directory, src/, is what Python puts on sys.path by default).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import RATIO_FEATURES, compute_ratios  # noqa: E402
from src.nlp import LEXICON_FEATURES, RiskTextVectorizer, extract_lexicon_features  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "risklabs_synthetic.csv"
MODELS_DIR = PROJECT_ROOT / "models"

LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}
LABEL_NAMES = ["Low", "Medium", "High"]
RAW_FINANCIAL_COLS = [
    "revenue", "net_profit", "total_assets", "total_liabilities",
    "total_debt", "equity", "current_assets", "current_liabilities",
    "operating_cash_flow",
]
SYNTHETIC_NOTE = "Metrics on SYNTHETIC data \u2014 for pipeline validation only."


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        print(f"ERROR: dataset not found at {DATA_PATH}")
        print("Run data/generate_dataset.py first to create the synthetic dataset.")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)
    missing = {"report_text", "risk_label"} | set(RAW_FINANCIAL_COLS)
    absent = sorted(missing - set(df.columns))
    if absent:
        print(f"ERROR: dataset at {DATA_PATH} is missing columns: {absent}")
        sys.exit(1)
    return df


def build_numeric_features(df: pd.DataFrame):
    """Ratios + lexicon features as numpy arrays (stateless transforms)."""
    ratios = compute_ratios(df)[RATIO_FEATURES].to_numpy(dtype=float)
    lexicon = extract_lexicon_features(
        df["report_text"].fillna("").tolist()
    )[LEXICON_FEATURES].to_numpy(dtype=float)
    return ratios, lexicon


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc_ovr": float(roc_auc_score(y_true, y_proba, multi_class="ovr")),
    }


def print_comparison_table(metrics_a: dict, metrics_b: dict,
                           n_train: int, n_test: int) -> None:
    rows = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr"]
    header = f"{'metric':<16}{'Model A (ratios)':>18}{'Model B (ratios + text)':>25}"
    print("\nRiskLabs \u2014 Model A vs Model B  (held-out test set)")
    print(f"train={n_train}  test={n_test}  stratified 80/20, random_state=42")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row:<16}{metrics_a[row]:>18.4f}{metrics_b[row]:>25.4f}")
    print()
    print("NOTE: trained and evaluated on SYNTHETIC data \u2014 pipeline validation only.")


def main() -> None:
    df = load_dataset()
    y = df["risk_label"].map(LABEL_MAP)
    if y.isna().any():
        print(f"ERROR: unexpected risk_label values: {sorted(df['risk_label'].unique())}")
        sys.exit(1)
    y = y.to_numpy(dtype=int)

    # Stratified 80/20 split FIRST so the TF-IDF vocabulary is fit on train texts only.
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=y)
    y_train = train_df["risk_label"].map(LABEL_MAP).to_numpy(dtype=int)
    y_test = test_df["risk_label"].map(LABEL_MAP).to_numpy(dtype=int)

    Xa_train, Xl_train = build_numeric_features(train_df)
    Xa_test, Xl_test = build_numeric_features(test_df)

    vectorizer = RiskTextVectorizer()
    vectorizer.fit(train_df["report_text"].fillna("").tolist())
    Xt_train = vectorizer.transform(train_df["report_text"].fillna("").tolist()).toarray()
    Xt_test = vectorizer.transform(test_df["report_text"].fillna("").tolist()).toarray()

    Xb_train = np.hstack([Xa_train, Xl_train, Xt_train])  # dense: scale is small
    Xb_test = np.hstack([Xa_test, Xl_test, Xt_test])

    rf_params = dict(n_estimators=200, random_state=42, n_jobs=-1)
    model_a = RandomForestClassifier(**rf_params).fit(Xa_train, y_train)
    model_b = RandomForestClassifier(**rf_params).fit(Xb_train, y_train)

    metrics_a = evaluate(y_test, model_a.predict(Xa_test), model_a.predict_proba(Xa_test))
    metrics_b = evaluate(y_test, model_b.predict(Xb_test), model_b.predict_proba(Xb_test))
    print_comparison_table(metrics_a, metrics_b, len(y_train), len(y_test))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_a, MODELS_DIR / "model_a.pkl")
    joblib.dump(model_b, MODELS_DIR / "model_b.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")

    tfidf_names = vectorizer.get_feature_names()
    feature_names = {
        "financial": list(RATIO_FEATURES),
        "lexicon": list(LEXICON_FEATURES),
        "tfidf": list(tfidf_names),
    }
    (MODELS_DIR / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2), encoding="utf-8")

    median_profile = {c: float(df[c].median()) for c in RAW_FINANCIAL_COLS}
    (MODELS_DIR / "median_profile.json").write_text(
        json.dumps(median_profile, indent=2), encoding="utf-8")

    metrics = {
        "model_a": {k: round(v, 4) for k, v in metrics_a.items()},
        "model_b": {k: round(v, 4) for k, v in metrics_b.items()},
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "label_map": LABEL_MAP,
        "note": SYNTHETIC_NOTE,
    }
    (MODELS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nArtifacts written to {MODELS_DIR}/")
    print("  model_a.pkl, model_b.pkl, tfidf_vectorizer.pkl")
    print("  feature_names.json, median_profile.json, metrics.json")


if __name__ == "__main__":
    main()
