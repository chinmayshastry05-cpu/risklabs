# RiskLabs — Viva Preparation

## Project definition (say it in one breath)

"RiskLabs is an explainable AI-based financial risk prediction system that combines structured financial indicators with unstructured information extracted from financial reports using NLP, and uses machine-learning models to classify a company's financial risk while explaining the key factors behind the prediction."

## Research question

**"Does combining text features with financial indicators improve risk classification vs financial indicators alone?"**

## Model A vs Model B framing

- **Model A (baseline):** RandomForest trained on 7 engineered financial ratios only (`current_ratio`, `debt_to_equity`, `debt_to_assets`, `net_profit_margin`, `roa`, `roe`, `ocf_to_debt`).
- **Model B (full):** RandomForest trained on the 7 ratios **plus** 8 lexicon features (risk-language counts by category) **plus** 500 TF-IDF terms from the report text.
- Both use the same classifier, the same stratified 80/20 train/test split (`random_state=42`), and the same evaluation protocol — so any metric gap isolates the **value of the text features**.
- The comparison is extended with two more classifiers on each feature set — LogisticRegression (simple baseline, with scaling) and XGBoost (stronger tree baseline) — all on the identical split and protocol (`src/train.py`, `src/evaluate.py`).
- Compare on: **accuracy, per-class precision/recall/F1, macro and weighted F1, ROC-AUC (one-vs-rest), High-risk precision/recall, confusion matrices, and expected calibration error**. Macro averaging is used deliberately so a weak minority class is not masked by a strong majority class; High-risk recall is reported because missing a High-risk company is the costliest error.

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

*Limitations:* synthetic data (see Q4); the dataset is a single cross-section, not a time series, so trend information is lost (`src/trends.py` is ready infrastructure awaiting longitudinal data); the lexicon is hand-built and English-only; the dashboard heuristic indicators are not model output; and the model sees only the annual report plus financials — no macro-economic context. *Future work:* real labeled data with out-of-time validation (primary, framework in `src/temporal.py`); temporal features (year-on-year ratio changes); replacing TF-IDF with sentence-transformer embeddings via the existing `transform()` interface (only if the ablation evidence justifies it — the current evidence says fix scaling/selection first); LLM-generated narrative explanations constrained to cite the SHAP reasons (to avoid hallucinating causes); and calibration of the 0–100 risk score against observed default frequencies.

### 10. Why train two models instead of just shipping the best one?

Because the two-model setup *is* the experiment. A single good model tells you the system works; the A-vs-B comparison answers the research question — it isolates the marginal contribution of the text features under identical classifier, split, and metric protocol. Model A also serves a practical purpose: it keeps the system useful when no report text is available (e.g. the manual-entry mode with an empty excerpt), since it requires only the financial numbers.

### 11. Your Model B performs WORSE than Model A. Isn't that a failure?

It's the most instructive result in the project, and we report it openly — with one important nuance. Five verified reasons (`experiments/nlp_ablation.py` reproduces all of this on the same split): **(a)** the synthetic label rule gives text only ~10/100 points by construction — ratios nearly determine the label, so text has almost nothing to add; **(b)** ~39% of Model B's RandomForest importance mass sits on the 500 TF-IDF terms but only 2 terms exceed 1% importance — the trees waste splits on template filler; **(c)** 124 of the 500 TF-IDF terms overlap lexicon phrases (redundant encoding); **(d)** even the lexicon alone slightly hurts RF (0.9583 → 0.9458) while *helping* scaled LogisticRegression (0.9458 → 0.9583) — trees handle this noisy high-dim block worse than a scaled linear model; **(e)** chi2 top-100 feature selection + LogisticRegression reaches 0.9625, the best variant overall — text *can* help with scaling + selection, but the default full-TF-IDF RandomForest is the wrong vehicle.

**The nuance:** the "B < A" result is specific to the RandomForest setup — which is exactly why we added more baselines. XGBoost on the text-enriched feature set scores **0.9708 accuracy vs 0.9542 on ratios alone**: a stronger tree model that learns to ignore noisy features *does* extract value from the text block. So the honest statement is classifier-dependent: with RF, text hurts here; with XGBoost, text helps. The deepest point stands regardless: synthetic texts are template-generated from the same latent tiers as the labels, so they *cannot* contain independent information by construction. Whether real report text helps is an open question for real data.

### 12. How is your evaluation more rigorous than "accuracy"?

Accuracy alone can hide a model that never predicts the minority class. We report the full protocol (`src/evaluate.py`): confusion matrix, per-class precision/recall/F1 with support, macro *and* weighted averages, ROC-AUC (OvR), and — most importantly for risk — **High-risk precision and High-risk recall**, because a missed High-risk company costs more than a false alarm. We also report expected calibration error (ECE) so we know whether the stated probabilities mean what they say. Every classifier × feature-set combination uses the identical stratified split, so comparisons are fair.

### 13. Why add LogisticRegression and XGBoost if RandomForest was already good?

Because a single model family can't tell you whether your results are about the *features* or the *classifier*. LogisticRegression (+ scaler) is the simple baseline: it tests whether the signal is even linear (it isn't, on ratios alone — 0.9458 vs RF's 0.9583, since the label rule is threshold-based). XGBoost is the stronger tree baseline: gradient boosting usually squeezes more out of tabular data and handles noisy high-dimensional text features more gracefully. No deep learning was added — 1,200 rows of tabular features give no technical justification for it.

### 14. What is temporal validation, and why can't you do it on your data?

A random train/test split asks "can the model learn this dataset's patterns?" Temporal (out-of-time) validation asks the harder, honest question: "can a model trained on the PAST predict the FUTURE?" — train on older years, validate on a later period, test on the most recent. It matters because random splits are optimistic for risk models: they allow **look-ahead bias** (post-outcome information leaking into features) and ignore **regime drift** (credit cycles change the relationships). `src/temporal.py` implements the framework — `temporal_split()`, `walk_forward_splits()` — but its validator *refuses* to run on the current dataset because it has no `report_year` column: it's a single synthetic cross-section. Claiming temporal validation on it would be dishonest, so the plan is documented infrastructure for the real-data stage, where labels must be outcomes observed *after* the report date.

### 15. What's the difference between Model Probability, Risk Score, and the Heuristic Indicators?

Four distinct things, labeled explicitly in the dashboard: **Model Probability** is the classifier's raw per-class probability estimate. **Calibrated Model Probability** is that estimate adjusted (sigmoid, 5-fold CV) so stated probabilities better match observed frequencies — shown separately, with ECE reported before/after. **Risk Classification** is just the argmax decision (Low/Medium/High). The **Risk Score (0–100)** is `100 × (0·P(Low) + 0.5·P(Medium) + 1·P(High))` — a weighted summary of the probability distribution, *not* a calibrated real-world default frequency. The **Heuristic Risk Indicators** (liquidity/leverage/profitability/operational) are transparent documented formulas over ratios and lexicon counts — not model output, not probabilities. Conflating these is the most common way risk demos mislead; the UI separates them deliberately.

### 16. How do the SHAP explanations avoid making things up?

Three guards. First, every reason carries its feature *family* (financial / lexicon / tfidf) so ratio effects are never confused with text effects. Second, the plain-English sentences are generated from templates conditioned on the actual feature value *and* the sign of its SHAP contribution — e.g. "High leverage (debt-to-equity = 2.40) increased predicted risk" is only emitted when debt-to-equity is actually high *and* its contribution is positive; otherwise no sentence is produced. Third, TF-IDF word features are *never* narrated: a single word's SHAP contribution isn't interpretable in isolation, so inventing prose for it would be dishonest. The waterfall chart shows all top contributions with their direction; the paragraph only claims what the values support.
