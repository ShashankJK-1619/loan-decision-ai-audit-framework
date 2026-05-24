"""
Feature engineering for Home Credit Default Risk.

Two layers live here:

1. `engineer_domain_features` — credit-risk specific derived features. These
   are interpretable by a credit officer and produce SHAP reason codes that
   read like "high debt-to-income ratio" rather than "AMT_CREDIT high". That
   interpretability is the whole reason the compliance team blocks
   auto-decisioning.

2. `build_preprocessor` — a sklearn ColumnTransformer that turns the
   engineered frame into a model-ready matrix (impute → encode). It's a
   `Pipeline`-compatible artifact so the exact same transformation runs in
   training, batch scoring, and the Streamlit audit dashboard.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.utils.config import load_config
from src.utils.logging import get_logger

_log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Domain feature engineering
# ──────────────────────────────────────────────────────────────────────────────
def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add credit-risk-meaningful derived features.

    All operations are vectorised and safe under NaN (we avoid `np.divide`
    asserts). Division-by-zero is suppressed and surfaced as NaN so the
    imputer downstream handles it consistently.
    """
    df = df.copy()
    _log.info("Engineering domain features on %d rows", len(df))

    # ── Income & affordability ratios ────────────────────────────────────────
    # Debt-to-income: a foundational underwriting metric. The credit amount
    # the applicant is requesting relative to their reported income.
    df["CREDIT_INCOME_RATIO"] = _safe_divide(df.get("AMT_CREDIT"), df.get("AMT_INCOME_TOTAL"))

    # Annuity (monthly payment) as a fraction of income — proxy for
    # affordability of the requested loan.
    df["ANNUITY_INCOME_RATIO"] = _safe_divide(df.get("AMT_ANNUITY"), df.get("AMT_INCOME_TOTAL"))

    # Implicit loan term in years (credit / annuity, annualised). A long
    # implicit term on a small credit can indicate restructuring risk.
    df["CREDIT_TERM_YEARS"] = _safe_divide(df.get("AMT_CREDIT"), df.get("AMT_ANNUITY") * 12)

    # Down-payment proxy: goods price covered by the credit. <1 means
    # applicant is funding part of the purchase themselves.
    df["CREDIT_GOODS_RATIO"] = _safe_divide(df.get("AMT_CREDIT"), df.get("AMT_GOODS_PRICE"))

    # ── Demographics in interpretable units ──────────────────────────────────
    # Home Credit stores ages and tenure as negative day counts relative to
    # the application date. Converting to years makes SHAP reason codes
    # human-readable ("AGE_YEARS = 28" beats "DAYS_BIRTH = -10220").
    if "DAYS_BIRTH" in df:
        df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).astype("float32")
    if "DAYS_EMPLOYED" in df:
        df["EMPLOYED_YEARS"] = (-df["DAYS_EMPLOYED"] / 365.25).astype("float32")
    if "DAYS_REGISTRATION" in df:
        df["YEARS_REGISTRATION"] = (-df["DAYS_REGISTRATION"] / 365.25).astype("float32")
    if "DAYS_ID_PUBLISH" in df:
        df["YEARS_ID_PUBLISH"] = (-df["DAYS_ID_PUBLISH"] / 365.25).astype("float32")

    # Employment-to-age ratio: portion of life the applicant has been
    # employed. Stable, monotonic with credit-worthiness in most studies.
    df["EMPLOYED_AGE_RATIO"] = _safe_divide(df.get("EMPLOYED_YEARS"), df.get("AGE_YEARS"))

    # ── External credit scores ───────────────────────────────────────────────
    # EXT_SOURCE_{1,2,3} are anonymised bureau scores; combining them
    # reduces missingness and yields one of the strongest single predictors
    # in published Home Credit baselines.
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    if ext_cols:
        df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
        df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)
        df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)
        df["EXT_SOURCE_MAX"] = df[ext_cols].max(axis=1)
        df["EXT_SOURCE_NA_COUNT"] = df[ext_cols].isna().sum(axis=1).astype("int8")

    # ── Family burden ────────────────────────────────────────────────────────
    # Income per family member — captures household-level affordability that
    # raw income misses.
    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(df.columns):
        df["INCOME_PER_PERSON"] = _safe_divide(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])
    if {"CNT_CHILDREN", "CNT_FAM_MEMBERS"}.issubset(df.columns):
        df["CHILDREN_RATIO"] = _safe_divide(df["CNT_CHILDREN"], df["CNT_FAM_MEMBERS"])

    _log.info("Domain features added; new shape: %s", df.shape)
    return df


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Division that returns NaN on zero/None denominators instead of inf/error."""
    if numerator is None or denominator is None:
        return pd.Series(np.nan, index=getattr(numerator, "index", None))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing pipeline
# ──────────────────────────────────────────────────────────────────────────────
def split_feature_types(
    df: pd.DataFrame, exclude: List[str] | None = None
) -> Tuple[List[str], List[str]]:
    """Partition columns into numeric vs categorical, excluding ID/target."""
    exclude = set(exclude or [])
    numeric_cols = [c for c in df.columns
                    if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns
                        if c not in exclude and not pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """Construct the train/serve preprocessor.

    Choices and why:
    - Median imputation for numerics: robust to the long right tails common in
      monetary amounts (AMT_INCOME_TOTAL has billionaires in the dataset).
    - Most-frequent imputation for categoricals: simple and stable; a future
      iteration could swap in a "MISSING" indicator category.
    - OneHotEncoder with `handle_unknown="ignore"`: production inference will
      encounter unseen categories; we must not raise on them.
    - No scaling: XGBoost is scale-invariant and SHAP values are easier to
      reason about on raw units.
    """
    cfg = load_config()["preprocessing"]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=cfg["numeric_imputer"])),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=cfg["categorical_imputer"], fill_value="MISSING")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=0.01)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        # `remainder="drop"` is explicit so a stray column can't sneak into
        # training without being declared.
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor
