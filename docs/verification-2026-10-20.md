# Verification log — 2026-10-20 (independent-review fixes)

Environment: CPython 3.12.3, Linux x86_64. All installs below were made
into **fresh virtualenvs** created for this check (no reuse of the project
`.venv`, no system site-packages).

## 1. Full requirements (synthetic + real)

```
python3 -m venv rl-clean
TMPDIR=<on-disk tmp> rl-clean/bin/pip install --no-cache-dir -r requirements.txt
```
Result: **install succeeds**; every pin resolves on Python 3.12
(numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.2,
xgboost 3.4.1, streamlit 1.64.0, pymupdf 1.28.2, shap 0.52.0,
joblib 1.6.0, pytest 9.1.1).

- All `models/*.pkl` artifacts load under scikit-learn 1.9.1 with
  warnings escalated to errors — **no InconsistentVersionWarning** (the
  warning a 1.7.2 environment produces, as the independent review found).
- Full test suite in the clean env: **43 passed**.

Note: the pins do **not** support Python 3.10 (e.g. pandas 3.0.6 and
matplotlib 3.11.2 publish no cp310 wheels). Supported Python is 3.12 and
is documented as such in the README — that is the exact supported
version + resolvable pins option, not an accident of the dev machine.

## 2. Real-data-only requirements

```
python3 -m venv rl-realonly
rl-realonly/bin/pip install -r requirements-real.txt
```
Result: install succeeds (numpy, pandas, pymupdf, scikit-learn,
streamlit, pytest only — no xgboost/shap/matplotlib).

Proof it needs no synthetic pickles: the repo was copied to a scratch
tree **without `models/`** and, under this venv —

- `tests/test_case_study.py` + `tests/test_nim_llm.py`: **19 passed**;
- the dashboard ran under Streamlit's AppTest with **no exception** and
  the real case-study view rendered (the synthetic tabs report the
  missing artifacts instead of blocking the app).

## 3. Dashboard default view + screenshots

Headless Chromium (Playwright) against the running app verified the
first tab is "Real companies (case study)" with `aria-selected=true`,
and captured the persisted screenshots in `docs/screenshots/`:

- `case-study-bajaj-auto-fy2023-24.png` — Bajaj view; MD&A signals
  (incl. the standalone Table 7 current-ratio quote) are tagged
  "excerpt basis: standalone", the auditor signal "consolidated", and
  the Basis caution appears under Limitations.
- `case-study-vodafone-idea-fy2023-24.png` — Vodafone Idea view with
  per-signal basis tags, excerpt scope section, and limitations.

## 4. NIM validation hardening (regression-tested)

`tests/test_nim_llm.py` grew from 5 to 14 tests covering the review's
cases: `signals: null` / non-list / missing → `NimError` (was TypeError);
`going_concern_flag: "false"` → `False` (was truthy `True`); bool and
float severities rejected (was: bool passed as int); non-string
`overall_assessment` discarded; an assessment with **no surviving
verified signal** is withheld and flagged `assessment_unsupported`
instead of being presented as a conclusion. Re-validating the committed
LLM cache under the strict validator leaves the committed features
unchanged (all recorded flags/severities were already correctly typed);
4 chunk-level free-text assessments that the loose validator would have
passed are now withheld by design.

## 5. Run-log accounting (corrected description)

`data/real/llm_run_log.json` covers **17 chunks in total: 7 live calls
and 10 served from cache** in the final recorded run — not 17 fresh live
calls. The cache holds responses recorded live earlier the same day.
