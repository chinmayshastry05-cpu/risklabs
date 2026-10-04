"""Lightweight NLP features for RiskLabs (CPU-only, no model downloads).

Two feature families:
1. Lexicon features — a hand-built, fully transparent risk lexicon. Scoring is
   stateless: each phrase is counted at most once per document (presence-based),
   matched as a case-insensitive substring on the lowercased text. Overlapping
   phrases (e.g. "default" inside "debt default") each contribute their weight.
2. TF-IDF — a thin joblib-picklable wrapper around sklearn's TfidfVectorizer.

To use sentence-transformers embeddings instead, replace
RiskTextVectorizer.transform with model.encode(...) — keep output 2-D numeric.
"""

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# category -> list of (phrase, weight). Positive weights = risk language,
# negative weights (the "positive" category) = healthy language.
LEXICON = {
    "liquidity": [
        ("liquidity concern", 1.5),
        ("cash crunch", 2.0),
        ("liquidity squeeze", 2.0),
        ("working capital pressure", 1.5),
        ("cash flow strain", 1.5),
        ("delayed payments", 1.0),
    ],
    "leverage": [
        ("debt default", 3.0),
        ("default", 2.5),
        ("covenant breach", 2.5),
        ("overleveraged", 2.0),
        ("refinancing risk", 2.0),
        ("debt burden", 1.5),
    ],
    "profitability": [
        ("mounting losses", 2.5),
        ("declining revenue", 1.5),
        ("revenue decline", 1.5),
        ("falling margins", 1.5),
        ("eroding profitability", 1.5),
        ("margin compression", 1.0),
    ],
    "legal": [
        ("regulatory probe", 2.5),
        ("auditor qualification", 2.5),
        ("lawsuit", 2.0),
        ("litigation", 2.0),
        ("regulatory action", 2.0),
        ("penalty imposed", 1.5),
    ],
    "going_concern": [
        ("going concern", 3.0),
        ("bankruptcy", 3.0),
        ("insolvency", 3.0),
        ("winding up", 2.5),
        ("cease operations", 2.0),
        ("material uncertainty", 2.0),
    ],
    "positive": [
        ("strong liquidity position", -1.5),
        ("healthy cash generation", -1.5),
        ("robust balance sheet", -1.5),
        ("record profit", -1.5),
        ("strong order book", -1.5),
        ("stable demand", -1.0),
        ("low leverage", -1.0),
        ("debt reduction", -1.0),
        ("diversified revenue", -1.0),
        ("prudent working capital management", -1.0),
    ],
}

LEXICON_FEATURES = [
    "lex_total",
    "lex_liquidity",
    "lex_leverage",
    "lex_profitability",
    "lex_legal",
    "lex_going_concern",
    "lex_positive",
    "risk_sentence_count",
]

# Categories whose phrases count as risk language (positive weights).
_RISK_CATEGORIES = ("liquidity", "leverage", "profitability", "legal", "going_concern")
_RISK_PHRASES = tuple(
    phrase for cat in _RISK_CATEGORIES for phrase, _ in LEXICON[cat]
)

_SENTENCE_SPLIT = re.compile(r"[.!?;]+")


def extract_lexicon_features(texts: list) -> pd.DataFrame:
    """Score a list of report texts against LEXICON (stateless).

    For each text, every lexicon phrase matched (case-insensitive substring on
    the lowercased text, counted once per document) contributes its weight to
    its category score; lex_total is the sum over all categories (negative for
    the "positive" category, so lex_total falls for healthy language).
    risk_sentence_count = number of sentences containing at least one
    positive-weight (risk) phrase.
    Non-string inputs are treated as empty text.
    """
    rows = []
    for text in texts:
        lowered = text.lower() if isinstance(text, str) else ""
        row = {}
        total = 0.0
        for category, phrases in LEXICON.items():
            score = sum(weight for phrase, weight in phrases if phrase in lowered)
            row[f"lex_{category}"] = float(score)
            total += score
        row["lex_total"] = float(total)
        sentences = [s for s in _SENTENCE_SPLIT.split(lowered) if s.strip()]
        row["risk_sentence_count"] = int(
            sum(1 for s in sentences if any(p in s for p in _RISK_PHRASES))
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=LEXICON_FEATURES)


def _clean(texts):
    return [t if isinstance(t, str) else "" for t in texts]


class RiskTextVectorizer:
    """Joblib-picklable wrapper around sklearn's TfidfVectorizer.

    To use sentence-transformers embeddings instead, replace
    RiskTextVectorizer.transform with model.encode(...) — keep output 2-D numeric.
    """

    def __init__(self, max_features: int = 500):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2), stop_words="english"
        )

    def fit(self, texts) -> "RiskTextVectorizer":
        self.vectorizer.fit(_clean(texts))
        return self

    def transform(self, texts):
        return self.vectorizer.transform(_clean(texts))

    def get_feature_names(self) -> list:
        return list(self.vectorizer.get_feature_names_out())
