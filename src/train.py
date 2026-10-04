"""RiskLabs training script.

Trains and compares risk classifiers on two feature sets:

  Feature set A — 7 financial ratio features only.
  Feature set B — 7 ratio features + 8 lexicon text features + 500 TF-IDF terms
                  (TF-IDF input densified with .toarray(); scale is small).

Classifiers (same stratified 80/20 split, same evaluation for all):
  rf  — RandomForestClassifier (the ORIGINAL RiskLabs baseline; artifacts
        model_a.pkl / model_b.pkl keep these names for backward compatibility)
  lr  — LogisticRegression in a StandardScaler pipeline (simple baseline)
  xgb — XGBoost (stronger tree-based model)

Additionally fits probability calibration (sigmoid, 5-fold CV) around the two
RandomForest models. Calibrated probabilities are saved as separate artifacts
and exposed separately in the dashboard — the base models are kept for SHAP.

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Make `src.*` importable when run as `python src/train.py` from the project root
# (the script's own directory, src/, is what Python puts on sys.path by default).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import (calibrate_model, evaluate_classifier,  # noqa: E402
                          ordered_proba, print_comparison_table)
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
SYNTHETIC_NOTE = "Metrics on SYNTHETIC data — for pipeline validation only."

RF_PARAMS = dict(n_estimators=200, random_state=42, n_jobs=-1)
XGB_PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  random_state=42, n_jobs=-1, eval_metric="mlogloss")


def make_classifier(name: str):
    """Return a FRESH unfitted estimator for *name*.

    A new instance is required for every (classifier, feature-set) fit:
    reusing one instance across fits would silently overwrite the earlier
    model (all dict entries would point at the last-fitted object).
    """
    if name == "rf":
        return RandomForestClassifier(**RF_PARAMS)
    if name == "lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42),
        )
    if name == "xgb":
        return XGBClassifier(**XGB_PARAMS)
    raise ValueError(f"unknown classifier: {name}")


CLASSIFIER_NAMES = ["rf", "lr", "xgb"]


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


def legacy_flat_metrics(full: dict) -> dict:
    """The original 5-key metric dict, kept for backward compatibility."""
    return {k: round(full[k], 4) for k in
            ("accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr")}


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
    feature_sets = {"a": (Xa_train, Xa_test), "b": (Xb_train, Xb_test)}

    classifiers = CLASSIFIER_NAMES
    models, results = {}, {}
    for set_name, (Xtr, Xte) in feature_sets.items():
        for clf_name in classifiers:
            model_id = f"{clf_name}_{set_name}"
            print(f"training {model_id} ...", flush=True)
            model = make_classifier(clf_name).fit(Xtr, y_train)
            proba = model.predict_proba(Xte)
            # Reorder to [Low, Medium, High] for every classifier uniformly.
            proba = np.stack([ordered_proba(model, Xte[i:i + 1]) for i in range(len(Xte))])
            pred = proba.argmax(axis=1)
            results[model_id] = evaluate_classifier(y_test, pred, proba)
            models[model_id] = model

    # Probability calibration for the two shipped RandomForest models.
    # Calibrated wrappers are saved separately; SHAP keeps using the base RF.
    print("calibrating rf_a and rf_b (sigmoid, 5-fold CV) ...", flush=True)
    calibrated = {}
    calibration_report = {}
    for set_name, (Xtr, Xte) in feature_sets.items():
        base = RandomForestClassifier(**RF_PARAMS)
        cal = calibrate_model(base, Xtr, y_train, cv=5, method="sigmoid")
        proba_cal = np.stack([ordered_proba(cal, Xte[i:i + 1]) for i in range(len(Xte))])
        pred_cal = proba_cal.argmax(axis=1)
        full_cal = evaluate_classifier(y_test, pred_cal, proba_cal)
        model_id = f"rf_{set_name}"
        calibrated[model_id] = cal
        calibration_report[model_id] = {
            "ece_before": round(results[model_id]["ece"], 4),
            "ece_after": round(full_cal["ece"], 4),
            "accuracy_after": round(full_cal["accuracy"], 4),
        }

    print_comparison_table(
        results,
        title="RiskLabs — classifier × feature-set comparison (held-out test set)",
        dataset_note=f"train={len(y_train)}  test={len(y_test)}  "
                     "stratified 80/20, random_state=42",
    )

    print("Probability calibration (RandomForest, sigmoid, 5-fold CV):")
    for model_id, rep in calibration_report.items():
        print(f"  {model_id}: ECE {rep['ece_before']:.4f} -> {rep['ece_after']:.4f} "
              f"(accuracy after calibration: {rep['accuracy_after']:.4f})")
    print()
    print("NOTE: Model B (ratios + text) does NOT beat Model A (ratios only) on "
          "this synthetic dataset — see VIVA.md for the investigation. "
          "The result is reported as-is; it is not hidden.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_names = {
        "rf_a": "model_a.pkl",            # ORIGINAL names preserved
        "rf_b": "model_b.pkl",
        "lr_a": "model_a_lr.pkl",
        "lr_b": "model_b_lr.pkl",
        "xgb_a": "model_a_xgb.pkl",
        "xgb_b": "model_b_xgb.pkl",
    }
    for model_id, fname in artifact_names.items():
        joblib.dump(models[model_id], MODELS_DIR / fname)
    joblib.dump(calibrated["rf_a"], MODELS_DIR / "model_a_calibrated.pkl")
    joblib.dump(calibrated["rf_b"], MODELS_DIR / "model_b_calibrated.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")

    tfidf_names = vectorizer.get_feature_names()
    feature_names = {
        "financial": list(RATIO_FEATURES),
        "lexicon": list(LEXICON_FEATURES),
        "tfidf": list(tfidf_names),
        "feature_set_a": list(RATIO_FEATURES),
        "feature_set_b": list(RATIO_FEATURES) + list(LEXICON_FEATURES) + list(tfidf_names),
        "n_features_a": len(RATIO_FEATURES),
        "n_features_b": len(RATIO_FEATURES) + len(LEXICON_FEATURES) + len(tfidf_names),
    }
    (MODELS_DIR / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2), encoding="utf-8")

    median_profile = {c: float(df[c].median()) for c in RAW_FINANCIAL_COLS}
    (MODELS_DIR / "median_profile.json").write_text(
        json.dumps(median_profile, indent=2), encoding="utf-8")

    def rounded(obj):
        if isinstance(obj, float):
            return round(obj, 4)
        if isinstance(obj, dict):
            return {k: rounded(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [rounded(v) for v in obj]
        return obj

    metrics = {
        "models": {mid: rounded(res) for mid, res in results.items()},
        # Legacy flat keys (original 5-metric format) for backward compatibility.
        "model_a": legacy_flat_metrics(results["rf_a"]),
        "model_b": legacy_flat_metrics(results["rf_b"]),
        "calibration": calibration_report,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "label_map": LABEL_MAP,
        "label_names": LABEL_NAMES,
        "classifiers": ["rf", "lr", "xgb"],
        "feature_sets": {"a": "financial ratios only",
                         "b": "ratios + lexicon + TF-IDF"},
        "split": "stratified 80/20, random_state=42 (random-split baseline; "
                 "see src/temporal.py for the out-of-time framework)",
        "note": SYNTHETIC_NOTE,
    }
    (MODELS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nArtifacts written to {MODELS_DIR}/")
    print("  model_a.pkl (RF, ratios), model_b.pkl (RF, ratios+text) — original names kept")
    print("  model_a_lr.pkl, model_b_lr.pkl (LogReg), model_a_xgb.pkl, model_b_xgb.pkl (XGBoost)")
    print("  model_a_calibrated.pkl, model_b_calibrated.pkl (calibrated RF probabilities)")
    print("  tfidf_vectorizer.pkl, feature_names.json, median_profile.json, metrics.json")


if __name__ == "__main__":
    main()
