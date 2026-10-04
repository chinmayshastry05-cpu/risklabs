"""NLP ablation study: WHY does Model B (ratios + text) underperform Model A?

Runs RF and scaled-LogReg on feature subsets under the SAME stratified split
as src/train.py, and prints a text-feature diagnostic report. This is analysis
only — it writes no model artifacts and changes no defaults. Findings are
summarized in VIVA.md ("Why doesn't the NLP model beat the financial-only one?").
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import evaluate_classifier  # noqa: E402
from src.features import RATIO_FEATURES, compute_ratios  # noqa: E402
from src.nlp import (LEXICON_FEATURES, RiskTextVectorizer,  # noqa: E402
                     extract_lexicon_features, select_text_features,
                     text_feature_report)

LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}


def main() -> None:
    df = pd.read_csv(PROJECT_ROOT / "data" / "risklabs_synthetic.csv")
    y = df["risk_label"].map(LABEL_MAP).to_numpy(dtype=int)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=y)
    y_tr = train_df["risk_label"].map(LABEL_MAP).to_numpy(dtype=int)
    y_te = test_df["risk_label"].map(LABEL_MAP).to_numpy(dtype=int)

    Xa_tr = compute_ratios(train_df)[RATIO_FEATURES].to_numpy(float)
    Xa_te = compute_ratios(test_df)[RATIO_FEATURES].to_numpy(float)
    Xl_tr = extract_lexicon_features(train_df["report_text"].fillna("").tolist())[LEXICON_FEATURES].to_numpy(float)
    Xl_te = extract_lexicon_features(test_df["report_text"].fillna("").tolist())[LEXICON_FEATURES].to_numpy(float)

    vec = RiskTextVectorizer().fit(train_df["report_text"].fillna("").tolist())
    Xt_tr = vec.transform(train_df["report_text"].fillna("").tolist())
    Xt_te = vec.transform(test_df["report_text"].fillna("").tolist())

    print("=== text feature diagnostics (train texts) ===")
    for k, v in text_feature_report(train_df["report_text"].fillna("").tolist(), vec).items():
        if k == "lexicon_overlap_terms":
            print(f"  {k}: {v[:12]}{'...' if len(v) > 12 else ''}")
        else:
            print(f"  {k}: {v}")

    Xt_tr_k, sel = select_text_features(Xt_tr, y_tr, k=100)
    Xt_te_k = sel.transform(Xt_te)

    variants = {
        "ratios only": (Xa_tr, Xa_te),
        "ratios + lexicon": (np.hstack([Xa_tr, Xl_tr]), np.hstack([Xa_te, Xl_te])),
        "ratios + tfidf": (np.hstack([Xa_tr, Xt_tr.toarray()]), np.hstack([Xa_te, Xt_te.toarray()])),
        "ratios + lexicon + tfidf (Model B)": (
            np.hstack([Xa_tr, Xl_tr, Xt_tr.toarray()]),
            np.hstack([Xa_te, Xl_te, Xt_te.toarray()])),
        "ratios + lexicon + tfidf top-100 chi2": (
            np.hstack([Xa_tr, Xl_tr, Xt_tr_k.toarray()]),
            np.hstack([Xa_te, Xl_te, Xt_te_k.toarray()])),
    }

    rf = lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    lr = lambda: make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=2000, random_state=42))

    print("\n=== ablation: test accuracy / macro-F1 (same split) ===")
    print(f"{'features':<42}{'RF acc':>9}{'RF F1m':>9}{'LR acc':>9}{'LR F1m':>9}")
    for name, (Xtr, Xte) in variants.items():
        m_rf = rf().fit(Xtr, y_tr)
        m_lr = lr().fit(Xtr, y_tr)
        pr, pl = m_rf.predict(Xte), m_lr.predict(Xte)
        print(f"{name:<42}{accuracy_score(y_te, pr):>9.4f}"
              f"{f1_score(y_te, pr, average='macro'):>9.4f}"
              f"{accuracy_score(y_te, pl):>9.4f}"
              f"{f1_score(y_te, pl, average='macro'):>9.4f}")

    # How much of the RF-B importance mass sits on noisy TF-IDF terms?
    Xb_tr = variants["ratios + lexicon + tfidf (Model B)"][0]
    m = rf().fit(Xb_tr, y_tr)
    imp = m.feature_importances_
    tfidf_imp = imp[15:]
    print("\n=== RF Model B feature-importance audit ===")
    print(f"  importance mass on 7 ratios      : {imp[:7].sum():.3f}")
    print(f"  importance mass on 8 lexicon     : {imp[7:15].sum():.3f}")
    print(f"  importance mass on 500 TF-IDF    : {tfidf_imp.sum():.3f}")
    print(f"  TF-IDF terms with importance>0  : {(tfidf_imp > 0).sum()} / 500")
    print(f"  TF-IDF terms with importance>1% : {(tfidf_imp > 0.01).sum()} / 500")
    top_tfidf = np.argsort(tfidf_imp)[::-1][:8]
    names = vec.get_feature_names()
    print("  top TF-IDF terms by importance:",
          [(names[i], round(float(tfidf_imp[i]), 4)) for i in top_tfidf])


if __name__ == "__main__":
    main()
