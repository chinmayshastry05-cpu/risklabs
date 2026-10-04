"""Temporal / out-of-time validation framework for RiskLabs.

Why this exists
---------------
A random train/test split answers: "can the model learn the patterns in this
dataset?" For financial risk, the harder and more honest question is: "can a
model trained on the PAST predict the FUTURE?" Random splits are optimistic
for risk models because:

  1. **Look-ahead bias** — if the data contain post-outcome information
     (e.g. a report written after a default), a random split lets the model
     peek at the future.
  2. **Regime change** — financial relationships drift (credit cycles, policy
     changes). A model that only needs to interpolate within one period may
     fail when the regime shifts.

Out-of-time validation — train on older years, validate on a later period,
test on the most recent period — directly measures forward-generalization and
is the standard discipline in credit-risk and default modeling.

Status in this repo
-------------------
The shipped synthetic dataset (data/risklabs_synthetic.csv) is a SINGLE
CROSS-SECTION: it has no `report_year` column and represents no real time
period. These utilities therefore CANNOT produce real temporal validation
today. They are the framework that will be used the day real longitudinal
data (company × year rows with a report_year column) is introduced — see
README "Where real data plugs in".

Usage (with real longitudinal data)
-----------------------------------
    df = pd.read_csv("real_data.csv")          # must contain report_year
    train_df, val_df, test_df = temporal_split(df, year_col="report_year",
                                               train_years=(2015, 2018),
                                               val_years=(2019, 2020),
                                               test_years=(2021, 2022))
    for fold in walk_forward_splits(df, year_col="report_year",
                                    train_window=4, test_window=1,
                                    start_year=2015, end_year=2022):
        ... train on fold["train"], evaluate on fold["test"] ...
"""

import pandas as pd

MIN_YEARS_PER_SPLIT = 1


def validate_temporal_columns(df: pd.DataFrame, year_col: str = "report_year") -> None:
    """Raise a clear error if temporal validation is not possible on *df*.

    This is intentionally strict: silently "temporal-splitting" a dataset with
    no real time dimension would be dishonest.
    """
    if year_col not in df.columns:
        raise ValueError(
            f"Temporal validation requires a '{year_col}' column with the year "
            "each observation's report belongs to. The current dataset has no "
            "such column (it is a single synthetic cross-section), so only the "
            "random-split baseline is valid for it."
        )
    years = pd.to_numeric(df[year_col], errors="coerce")
    if years.isna().any():
        raise ValueError(f"'{year_col}' contains non-numeric year values.")
    if years.nunique() < 3:
        raise ValueError(
            f"'{year_col}' has only {years.nunique()} distinct year(s); "
            "temporal validation needs at least 3 distinct years "
            "(train / validation / test)."
        )


def temporal_split(df: pd.DataFrame, year_col: str = "report_year",
                   train_years=(None, None), val_years=(None, None),
                   test_years=(None, None)):
    """Split a longitudinal frame into train / validation / test by year.

    Each *_years is an inclusive (start, end) tuple. Years outside all three
    windows are dropped. Returns (train_df, val_df, test_df) in chronological
    order. Raises if any split is empty.
    """
    validate_temporal_columns(df, year_col)
    years = pd.to_numeric(df[year_col])

    def window(mask_years):
        lo, hi = mask_years
        mask = pd.Series(True, index=df.index)
        if lo is not None:
            mask &= years >= lo
        if hi is not None:
            mask &= years <= hi
        return df.loc[mask].copy()

    train_df, val_df, test_df = (window(w) for w in (train_years, val_years, test_years))
    for name, part in (("train", train_df), ("validation", val_df), ("test", test_df)):
        if len(part) == 0:
            raise ValueError(
                f"Temporal split produced an empty '{name}' set — "
                "check the year windows against the data's year range."
            )
    return train_df, val_df, test_df


def walk_forward_splits(df: pd.DataFrame, year_col: str = "report_year",
                        train_window: int = 4, test_window: int = 1,
                        start_year: int = None, end_year: int = None):
    """Yield expanding-window (train) / rolling (test) folds over years.

    Each fold: {"train_years": (lo, hi), "test_years": (lo, hi),
    "train": df_train, "test": df_test}. The training window EXPANDS (all
    history up to the test window) — standard for risk models, which should
    use all available past data.
    """
    validate_temporal_columns(df, year_col)
    years = sorted(pd.to_numeric(df[year_col]).unique().tolist())
    start_year = start_year if start_year is not None else years[0]
    end_year = end_year if end_year is not None else years[-1]

    folds = []
    test_lo = start_year + train_window
    while test_lo + test_window - 1 <= end_year:
        train_lo, train_hi = start_year, test_lo - 1
        test_hi = test_lo + test_window - 1
        mask_years = pd.to_numeric(df[year_col])
        train = df.loc[(mask_years >= train_lo) & (mask_years <= train_hi)].copy()
        test = df.loc[(mask_years >= test_lo) & (mask_years <= test_hi)].copy()
        if len(train) and len(test):
            folds.append({
                "train_years": (train_lo, train_hi),
                "test_years": (test_lo, test_hi),
                "train": train,
                "test": test,
            })
        test_lo += test_window
    if not folds:
        raise ValueError("No walk-forward folds could be built from the year range.")
    return folds


def describe_temporal_plan() -> str:
    """Human-readable summary of the intended real-data validation protocol."""
    return (
        "Temporal validation plan (requires real longitudinal data with a "
        "'report_year' column):\n"
        "  1. Sort company-year observations by report year.\n"
        "  2. Train on older years, tune on a later validation window, "
        "report final metrics on the most recent test window.\n"
        "  3. Additionally run walk-forward folds (expanding train window, "
        "rolling 1-year test window) to measure stability across regimes.\n"
        "  4. Labels must be outcomes OBSERVED AFTER the report date (e.g. "
        "default within 2 years) so no post-outcome information leaks into "
        "features — this is the look-ahead-bias guard.\n"
        "The current synthetic dataset has no year column, so this plan is "
        "documented infrastructure, not an executed experiment."
    )
