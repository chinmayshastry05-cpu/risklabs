"""Full evaluation protocol for RiskLabs models.

evaluate_classifier() computes a complete, JSON-serializable metric set for a
3-class risk classifier on a held-out set:

  - accuracy
  - confusion matrix (as nested lists, row=true / col=predicted)
  - per-class precision, recall, F1 (and support)
  - macro and weighted precision/recall/F1
  - ROC-AUC (one-vs-rest)
  - High-risk precision and High-risk recall (the business-critical minority
    signal: a missed High-risk company costs more than a false alarm)
  - sklearn classification_report (string) for human reading
  - expected calibration error (ECE) of predict_proba on the test set

calibrate_model() wraps a fitted classifier in CalibratedClassifierCV
(sigmoid, 5-fold by default). Calibrated probabilities are EXPOSED SEPARATELY
from raw model probabilities and are used for display only — SHAP explanations
still run on the uncalibrated base model (SHAP needs the tree ensemble, not
the calibration wrapper).
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

LABEL_NAMES = ["Low", "Medium", "High"]
HIGH_RISK_LABEL = 2  # index of "High" in LABEL_NAMES


def expected_calibration_error(y_true, y_proba, n_bins: int = 10) -> float:
    """Mean |confidence - accuracy| over equal-width probability bins.

    A well-calibrated model has ECE close to 0. Computed on the max-class
    probability of each sample.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    conf = y_proba.max(axis=1)
    pred = y_proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_true)) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def evaluate_classifier(y_true, y_pred, y_proba, label_names=None) -> dict:
    """Full metric dict for one classifier on one held-out set."""
    label_names = label_names or LABEL_NAMES
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    per_class = {}
    prec = precision_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    rec = recall_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    for i, name in enumerate(label_names):
        per_class[name] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(cm[i].sum()),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(label_names),
        "per_class": per_class,
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc_ovr": float(roc_auc_score(y_true, y_proba, multi_class="ovr")),
        "high_risk_precision": float(per_class["High"]["precision"]),
        "high_risk_recall": float(per_class["High"]["recall"]),
        "ece": expected_calibration_error(y_true, y_proba),
        "classification_report": classification_report(
            y_true, y_pred, target_names=label_names, digits=4, zero_division=0
        ),
    }


def calibrate_model(base_estimator, X_train, y_train, cv: int = 5,
                    method: str = "sigmoid"):
    """Fit CalibratedClassifierCV around an (unfitted or fitted) estimator.

    The returned wrapper is for DISPLAYING calibrated probabilities only.
    Keep the base model for SHAP explanations and feature importances.
    """
    calibrator = CalibratedClassifierCV(estimator=base_estimator, cv=cv, method=method)
    return calibrator.fit(X_train, y_train)


def ordered_proba(model, x_row, label_order=None):
    """predict_proba reordered to [Low, Medium, High] regardless of how the
    model stores classes (ints 0/1/2 or alphabetically ordered strings)."""
    label_order = label_order or LABEL_NAMES
    proba = model.predict_proba(x_row)[0]
    classes = list(model.classes_)
    if all(isinstance(c, str) for c in classes):
        idx = [classes.index(name) for name in label_order]
    else:
        # integer-encoded classes: 0=Low, 1=Medium, 2=High
        name_to_int = {name: i for i, name in enumerate(label_order)}
        idx = [classes.index(name_to_int[name]) for name in label_order]
    return np.asarray(proba, dtype=float)[idx]


def print_comparison_table(results: dict, title: str = "RiskLabs — classifier comparison",
                           dataset_note: str = "") -> None:
    """results: {model_label: evaluate_classifier(...) dict}. Prints a compact table."""
    rows = ["accuracy", "f1_macro", "f1_weighted", "high_risk_recall",
            "high_risk_precision", "roc_auc_ovr", "ece"]
    labels = list(results.keys())
    header = f"{'metric':<20}" + "".join(f"{lab:>16}" for lab in labels)
    print(f"\n{title}")
    if dataset_note:
        print(dataset_note)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row:<20}"
        for lab in labels:
            line += f"{results[lab][row]:>16.4f}"
        print(line)
    print()
    print("NOTE: trained and evaluated on SYNTHETIC data — pipeline validation only.")
