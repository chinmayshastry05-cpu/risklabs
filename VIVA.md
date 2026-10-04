# RiskLabs — Viva Preparation

## Project definition (say it in one breath)

"RiskLabs is an explainable AI-based financial risk prediction system that combines structured financial indicators with unstructured information extracted from financial reports using NLP, and uses machine-learning models to classify a company's financial risk while explaining the key factors behind the prediction."

## Research question

**"Does combining text features with financial indicators improve risk classification vs financial indicators alone?"**

## Model A vs Model B framing

- **Model A (baseline):** RandomForest trained on 7 engineered financial ratios only (`current_ratio`, `debt_to_equity`, `debt_to_assets`, `net_profit_margin`, `roa`, `roe`, `ocf_to_debt`).
- **Model B (full):** RandomForest trained on the 7 ratios **plus** 8 lexicon features (risk-language counts by category) **plus** 500 TF-IDF terms from the report text.
- Both use the same classifier, the same stratified 80/20 train/test split (`random_state=42`), and the same evaluation protocol — so any metric gap isolates the **value of the text features**.
- Compare on: **accuracy, macro precision, macro recall, macro F1, and ROC-AUC (one-vs-rest)**. Macro averaging is used deliberately so a weak minority class is not masked by a strong majority class.

## Likely questions (with answers)

### 1. Why RandomForest and not deep learning?

For this data regime, RandomForest is the right tool, not the cheap one. We have ~1,200 rows with low-dimensional tabular features — far too little data for deep learning to be justified, and neural nets would be harder to audit. RandomForest handles non-linear interactions between ratios without feature scaling, gives probabilistic outputs we need for the 0–100 risk score, and — critically — works natively with SHAP's TreeExplainer, which gives exact, fast explanations instead of sampling-based approximations. A viva-safe framing: *I chose the model family to fit the data and the explainability requirement, and I would revisit deep learning only at real-data scale with unstructured documents long enough to justify it.*

### 2. What is SHAP actually computing?

SHAP decomposes a single prediction into per-feature contributions. For each feature, it asks: *across every possible subset of the other features, how much does the predicted probability change when this feature is added?* — then averages those changes over all subsets. That average is the feature's Shapley value: its fair share of credit for this prediction, grounded in cooperative game theory. In RiskLabs we use `shap.TreeExplainer` on the fitted RandomForest and take the values for the **predicted class** (highest probability), so the top features listed in the dashboard are the ones that actually pushed *this company* toward *this* risk level. Direction is simple: a positive contribution "increases risk", a negative one "reduces risk".

### 3. Why TF-IDF + lexicon instead of just embeddings?

Three reasons. **(a) Auditability:** the lexicon is a fixed, human-readable dictionary of risk phrases ("cash crunch", "covenant breach") with explicit weights — an examiner can read it, critique it, and trace any prediction back to it. **(b) Efficiency and determinism:** TF-IDF with `max_features=500` runs on CPU with no downloads and is fully reproducible; sentence-transformer embeddings add a heavy dependency and opaque 384-dim vectors for unclear gain at this scale. **(c) Complementarity:** the lexicon captures *domain knowledge* (we know going-concern language matters), while TF-IDF captures *data-driven* phrasing the lexicon missed. That said, `src/nlp.py` is written with a clean `transform()` interface precisely so embeddings can be swapped in later — the spec notes the upgrade path.

### 4. Isn't your synthetic data circular? The labels are derived from the same signals the model sees.

Yes — and this is the honest core of the project. The labels in `risklabs_synthetic.csv` are produced by transparent, documented rules that score the same ratios and text signals the models then learn from. So **the metrics validate pipeline correctness, not real-world predictive power**: they prove the feature engineering runs, the A/B experiment is fair, SHAP explains the trained model faithfully, and the dashboard works end-to-end. They do *not* prove RiskLabs would work on real companies. Real labeled data — outcomes like default/bankruptcy filings, credit-rating actions, or auditor qualifications — is the stated next step, and the pipeline is built data-agnostic so nothing but the CSV (and its labels) needs to change.

### 5. How do you handle class imbalance?

The train/test split is **stratified**, so Low/Medium/High proportions are preserved in both sets and evaluation isn't skewed by a lucky split. Evaluation uses **macro-averaged** precision/recall/F1, which weights all three classes equally rather than letting the majority class dominate the score. The generator produces a stratified-ish mix, so imbalance here is mild; with real data — where defaults are genuinely rare — the plan is explicit: class weights (`class_weight="balanced"`), careful thresholding, and PR-AUC on the minority class instead of accuracy. Mentioning this unprompted scores well.

### 6. How do you prevent data leakage between train and test?

The split happens **before** any text vectorization, and the TF-IDF vectorizer is **fit on the training texts only**, then used to transform the test texts. If we fit TF-IDF on all data first, the test set's vocabulary would leak into the training features. Same discipline applies to any future scaling or embedding: fit on train, apply to test. The split itself is fixed with `random_state=42` so results are reproducible.

### 7. Are the category sub-scores (liquidity, leverage, profitability, operational) learned by the model?

No — and the dashboard says so. They are **heuristic** 0–100 sub-scores derived from relevant ratios and lexicon categories (e.g. `current_ratio` + liquidity lexicon feed the liquidity bar), meant to give the user an interpretable breakdown alongside the learned prediction. They are not model outputs and must not be read as such. Keeping them explicitly heuristic is the honest design choice; learning calibrated sub-scores per category is a listed future work.

### 8. What would you do with real data?

Four concrete steps. **(1)** Replace the CSV with real financials + real report text, keeping the same 9 financial columns so the pipeline runs unchanged. **(2)** Derive labels from real outcomes with a **time lag** — e.g. label a company High risk if it defaults within 2 years of the report date — and verify no post-outcome information appears in the inputs (no lookahead bias). **(3)** Re-train and re-run the A/B comparison, then validate **out-of-time** (train on earlier years, test on later years) rather than a random split, because financial risk models must generalize forward in time. **(4)** Tune the decision threshold on the business cost of errors (a missed High-risk company costs more than a false alarm) rather than defaulting to argmax.

### 9. What are the main limitations and future work?

*Limitations:* synthetic data (see Q4); the dataset is a single cross-section, not a time series, so trend information is lost; the lexicon is hand-built and English-only; the dashboard sub-scores are heuristic; and the model sees only the annual report plus financials — no macro-economic context. *Future work:* real labeled data with out-of-time validation (primary); temporal features (year-on-year ratio changes); replacing TF-IDF with sentence-transformer embeddings via the existing `transform()` interface; LLM-generated narrative explanations constrained to cite the SHAP reasons (to avoid hallucinating causes); and calibration of the 0–100 risk score against observed default frequencies.

### 10. Why train two models instead of just shipping the best one?

Because the two-model setup *is* the experiment. A single good model tells you the system works; the A-vs-B comparison answers the research question — it isolates the marginal contribution of the text features under identical classifier, split, and metric protocol. Model A also serves a practical purpose: it keeps the system useful when no report text is available (e.g. the manual-entry mode with an empty excerpt), since it requires only the financial numbers.
