# RiskLabs Build Progress

- [2026-10-03 01:52 IST] Build started. Workspace created at ~/workspace/risklabs/.
- [2026-10-03 01:55 IST] Spec frozen (SPEC.md). 4 parallel workers spawned: (1) dataset+features+NLP, (2) train+explain, (3) pdf+dashboard+requirements, (4) README+VIVA. Dependency install running in background.
- [2026-10-03 01:57 IST] pip blocked by PEP 668 externally-managed environment; retrying install with --break-system-packages.
- [2026-10-03 02:00 IST] Docs done: README.md + VIVA.md written and verified (honesty notice, ASCII architecture, A-vs-B framing, 10 viva Q&As). sklearn/pandas importable; full dependency install still finalizing.
- [2026-10-03 02:05 IST] System site-packages broken by Debian numpy/pandas conflict; building clean venv at ~/workspace/risklabs/.venv with all deps. All running workers redirected to the venv python.
- [2026-10-03 02:12 IST] Clean venv ready at ~/workspace/risklabs/.venv — pandas, numpy, scikit-learn, streamlit, pymupdf, shap, joblib all import OK.
- [2026-10-03 02:14 IST] PDF+dashboard done: src/pdf_parse.py demo extracts 1099 chars from generated sample_report.pdf; app.py compiles and pipeline smoke-tested (awaiting explain.py + trained models from other workers).
- [2026-10-03 02:17 IST] Dataset verified by coordinator: 1200 rows x 12 cols, labels Low 551 / Medium 339 / High 310, zero nulls; compute_ratios and extract_lexicon_features match SPEC with no NaNs.
- [2026-10-03 02:19 IST] Data worker complete: dataset generation deterministic (byte-identical reruns), all feature/NLP verifications pass under venv. Awaiting train+explain worker.
- [2026-10-03 02:28 IST] Training verified by coordinator: ran src/train.py independently — metrics reproduced exactly (A: acc 0.9583 / F1 0.9608 / AUC 0.9962; B: acc 0.9458 / F1 0.9500 / AUC 0.9858). All 6 model artifacts on disk.
- [2026-10-03 02:31 IST] Explainability verified end-to-end: SHAP reasons + plain-English paragraph generated for a test company.
- [2026-10-03 02:33 IST] Dashboard smoke-tested: `streamlit run app.py --server.headless true` serves HTTP 200 on / and /_stcore/health. Server shut down after test.
- [2026-10-03 02:34 IST] requirements.txt pinned from working venv (pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, streamlit 1.64.0, pymupdf 1.28.2, shap 0.52.0, joblib 1.6.0). All modules py_compile clean.
- [2026-10-03 02:35 IST] BUILD COMPLETE

## Improvement pass (2026-10-05) — senior ML review, no rebuild

- [2026-10-05] Assessment: inspected full repo; baseline reproduces exactly
  (RF-A acc 0.9583, RF-B acc 0.9458). Found latent dashboard bug: `ordered_proba`
  indexed string labels against int classes_ → "Analyze Risk" crashed with
  ValueError (smoke test had only hit HTTP 200). Assessment saved at
  ~/workspace/risklabs-improvement/ASSESSMENT.md.
- [2026-10-05] New `src/evaluate.py`: confusion matrix, per-class P/R/F1,
  macro+weighted averages, ROC-AUC OvR, High-risk precision/recall, ECE,
  classification_report, calibration helpers, shared `ordered_proba`.
- [2026-10-05] New baselines in `src/train.py`: LogisticRegression (scaled
  pipeline, simple baseline) and XGBoost (stronger tree baseline), RF kept as
  original baseline. All 6 classifier×feature-set combos use the identical
  stratified split. Probability calibration (sigmoid, 5-fold CV) fitted around
  both RF models; calibrated probabilities saved as separate artifacts.
- [2026-10-05] NLP investigation (`experiments/nlp_ablation.py`): Model B < A
  for RF because (1) synthetic label rule gives text only ~10/100 points,
  (2) ~39% of RF-B importance mass spreads over 500 noisy TF-IDF terms,
  (3) 124/500 TF-IDF terms redundantly overlap lexicon phrases,
  (4) scaled LogReg handles the text block better than trees,
  (5) chi2 top-100 selection + LR reaches 0.9625. Nuance: XGBoost DOES gain
  from text (0.9542 → 0.9708) — the effect is classifier-dependent.
- [2026-10-05] New `src/temporal.py`: out-of-time validation framework
  (temporal_split, walk_forward_splits); refuses to run without a real
  report_year column — documented infrastructure, not a claim.
- [2026-10-05] New `src/trends.py`: multi-year Year → Risk Score → Risk Class
  profiling + increasing/stable/decreasing classification (OLS slope, ±2.0
  band); needs real longitudinal data, none fabricated.
- [2026-10-05] `src/explain.py`: feature-family tags, traceable factor
  sentences (value + SHAP-sign conditioned; TF-IDF never narrated),
  matplotlib SHAP waterfall figure, canonical probability-term labels.
- [2026-10-05] `src/pdf_parse.py` hardened: extract_report() with tables,
  text cleaning, header/footer stripping, empty-page/encrypted-file handling,
  max-pages/chars guards; informative ValueErrors instead of crashes.
- [2026-10-05] `app.py`: fixed ordered_proba crash; 3 tabs (Analyze company /
  Model comparison / Risk trend); labels "Model Probability",
  "Calibrated Model Probability", "Risk Classification", "Risk Score (0–100)",
  "Heuristic Risk Indicators"; SHAP waterfall; comparison table + confusion
  matrices + classification reports + calibration results in dashboard.
- [2026-10-05] New `tests/test_pipeline.py`: 24 pytest smoke tests covering
  features, NLP, PDF, evaluation, temporal, trends, explain, artifacts
  (incl. feature-count consistency regression test).
- [2026-10-05] requirements.txt: added xgboost 3.4.1, matplotlib 3.11.2,
  pytest 9.1.1 (pip install initially failed: disk full; cleared pip cache and
  set TMPDIR to home overlay).
- [2026-10-05] Docs: README (new architecture, evaluation methodology,
  baselines, calibration, NLP findings, temporal plan), VIVA (Q11–Q16:
  B<A investigation, rigorous evaluation, baselines, temporal validation,
  probability-term distinctions, anti-hallucination guards), SPEC amendment
  header, this log.
- [2026-10-05] Bug found during verification: train.py reused one classifier
  instance across feature sets, so A-models were silently overwritten by B
  fits (model_a.pkl expected 515 features). Fixed with per-fit factory;
  retrained; regression test added. Final metrics: RF-A 0.9583 / RF-B 0.9458
  (unchanged), XGB-B 0.9708 (best), calibration ECE RF-B 0.0853 → 0.0332.
  All 24 tests pass; dashboard analysis path verified end-to-end in-process;
  Streamlit serves HTTP 200 headless.
