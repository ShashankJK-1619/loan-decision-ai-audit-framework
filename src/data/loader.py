"""
Data loading utilities for the Home Credit Default Risk dataset.

Why this lives in a module instead of inline in the notebook:
- Inference paths (Streamlit app, batch scoring) must load data the same way
  training does. Centralising load + dtype logic removes train/serve skew.
- Memory matters: application_train alone is ~166 MB in pandas' default
  dtypes. Downcasting halves that and lets reviewers actually run the
  notebook on a laptop.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from src.utils.config import get_paths, load_config
from src.utils.logging import get_logger


_log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Memory optimisation
# ──────────────────────────────────────────────────────────────────────────────
def downcast_numeric(df: pd.DataFrame, exclude: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Downcast int/float columns to the smallest dtype that preserves values.

    XGBoost converts to float32 internally anyway, so persisting float32 on
    disk is correct, not lossy. Excluded columns (e.g. IDs, target) are kept
    untouched so we don't accidentally lose precision on identifiers.
    """
    exclude = set(exclude or [])
    before = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.columns:
        if col in exclude:
            continue
        col_type = df[col].dtype

        if pd.api.types.is_integer_dtype(col_type):
            # signed vs unsigned matters for storage size; pick based on min
            if df[col].min() >= 0:
                df[col] = pd.to_numeric(df[col], downcast="unsigned")
            else:
                df[col] = pd.to_numeric(df[col], downcast="integer")

        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="float")

    after = df.memory_usage(deep=True).sum() / 1024 ** 2
    _log.info("Downcast: %.1f MB → %.1f MB (%.0f%% reduction)",
              before, after, 100 * (1 - after / before))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Schema-aware loading
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_csv_path(name: str) -> Path:
    """Map a logical dataset name from config to its absolute CSV path."""
    cfg = load_config()
    paths = get_paths(cfg)
    fname = cfg["datasets"][name]
    return paths.data_raw / fname


def load_application_train(downcast: bool = True) -> pd.DataFrame:
    """Load application_train.csv with consistent dtypes and sentinel handling.

    Returns the raw applicant snapshot — the primary table for Step 1. Bureau
    and previous_application aggregates are deferred to later steps so this
    step stays readable.
    """
    path = _resolve_csv_path("application_train")
    if not path.exists():
        raise FileNotFoundError(
            f"Expected Home Credit data at {path}. Download instructions are "
            f"in README.md (Kaggle dataset: c/home-credit-default-risk)."
        )

    _log.info("Loading application_train from %s", path)
    df = pd.read_csv(path)

    cfg = load_config()
    sentinel = cfg["sentinels"]["days_employed_anomaly"]
    xna_values = cfg["sentinels"]["xna_categories"]
    id_col = cfg["project"]["id_col"]
    target_col = cfg["project"]["target_col"]

    # Collect every column we want to replace or add into a dict, then
    # build the result frame in a single `pd.concat`. This is the pandas-
    # idiomatic way to avoid the "DataFrame is highly fragmented" warning
    # that fires under pandas 3.x when columns are added one at a time.
    replacements: dict[str, pd.Series] = {}

    # Sentinel: DAYS_EMPLOYED uses 365243 (~1000 years) to mean "no record".
    # We replace with NaN so downstream imputation handles it explicitly
    # instead of XGBoost learning a spurious "1000-year tenure" signal.
    if "DAYS_EMPLOYED" in df.columns:
        n_anom = int((df["DAYS_EMPLOYED"] == sentinel).sum())
        cleaned = df["DAYS_EMPLOYED"].replace(sentinel, np.nan)
        replacements["DAYS_EMPLOYED"] = cleaned
        # The fact a value was sentinel is itself signal — keep an explicit
        # flag (employment-status missingness correlates with default).
        replacements["DAYS_EMPLOYED_ANOM"] = cleaned.isna().astype("int8")
        _log.info("Replaced %d DAYS_EMPLOYED sentinels (%.2f%% of rows)",
                  n_anom, 100 * n_anom / len(df))

    # 'XNA' is Home Credit's literal string for "not applicable" in
    # categoricals. Convert to NaN for consistency with other missing values.
    # Include both legacy 'object' and pandas 3.x's dedicated 'str' dtype.
    obj_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in obj_cols:
        replacements[col] = df[col].replace(xna_values, np.nan)

    # Rebuild the frame in one shot: keep untouched columns + concat new ones.
    untouched = df.drop(columns=[c for c in replacements if c in df.columns])
    new_block = pd.concat(replacements.values(), axis=1, keys=replacements.keys())
    df = pd.concat([untouched, new_block], axis=1)

    if downcast:
        # Keep ID and target as-is for downstream joins/labels.
        df = downcast_numeric(df, exclude=[id_col, target_col])

    _log.info("application_train shape: %s", df.shape)
    return df


def summarise_missingness(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Return a per-column missingness report sorted desc by missing fraction."""
    miss = df.isna().mean().sort_values(ascending=False)
    report = pd.DataFrame({
        "missing_frac": miss,
        "missing_count": df.isna().sum().reindex(miss.index),
        "dtype": df.dtypes.reindex(miss.index).astype(str),
    })
    return report.head(top_n)


def dtype_inventory(df: pd.DataFrame) -> Dict[str, int]:
    """Return a count of columns by dtype family. Used in EDA section header."""
    families = {
        "numeric_int": int(df.select_dtypes(include=["integer"]).shape[1]),
        "numeric_float": int(df.select_dtypes(include=["floating"]).shape[1]),
        "categorical_object": int(df.select_dtypes(include=["object", "category"]).shape[1]),
        "boolean": int(df.select_dtypes(include=["bool"]).shape[1]),
    }
    return families


def load_engineered_train() -> pd.DataFrame:
    """Load the post-feature-engineering DataFrame saved by Step 1.

    Single-line wrapper so downstream steps don't reach into raw paths
    directly. If Step 1 hasn't run yet, raises with a clear pointer.
    """
    paths = get_paths()
    path = paths.data_processed / "application_train_engineered.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected engineered parquet at {path}. Run Step 1 notebook first."
        )
    _log.info("Loading engineered training frame from %s", path)
    df = pd.read_parquet(path)
    _log.info("Engineered frame shape: %s", df.shape)
    return df
