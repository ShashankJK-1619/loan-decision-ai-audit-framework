"""
TreeSHAP explainability for the loan-default XGBoost classifier.

Why TreeSHAP specifically (and not LIME, permutation importance, or
KernelSHAP):

- Exact rather than sampled. Compliance auditors do not want to be told that
  a reason code is "approximately" attributed to a feature.
- Polynomial in tree depth × leaves, which on a 325-tree booster runs in
  seconds on tens of thousands of rows.
- Native XGBoost integration via `shap.TreeExplainer` — no model-agnostic
  wrapper required, so attribution is grounded in the tree structure itself.
- SHAP values are *additive*: for any applicant, sum of feature SHAPs +
  expected value = the model's logit output. That additive property is
  what makes per-decision reason codes legally defensible — every column on
  the explanation adds up to the decision, with no residual to hand-wave.

The plotting helpers return matplotlib figures so callers can both
`plt.show()` in a notebook and `mlflow.log_figure(...)` for the audit
trail.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.utils.logging import get_logger

_log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Computation
# ──────────────────────────────────────────────────────────────────────────────
def compute_shap_values(
    booster: xgb.Booster,
    X: np.ndarray,
    feature_names: List[str],
) -> Tuple[np.ndarray, float]:
    """Compute TreeSHAP values for every row of X.

    Returns
    -------
    shap_values : ndarray of shape (n_rows, n_features)
        Per-row feature contributions on the logit scale.
    expected_value : float
        The model's baseline (average prediction over the training set on
        the logit scale). Per-row SHAP sums plus this baseline equal the
        model's raw output for that row.
    """
    _log.info("Computing TreeSHAP values on %d rows × %d features", X.shape[0], X.shape[1])
    # `shap.TreeExplainer` consumes a Booster directly; no need to wrap in a
    # sklearn estimator. `feature_perturbation="tree_path_dependent"` is the
    # default and the right choice for credit risk — it doesn't require a
    # background dataset and respects the tree's actual decision paths.
    explainer = shap.TreeExplainer(booster)
    # Pass a DataFrame so the shap library carries the column names through.
    X_df = pd.DataFrame(X, columns=feature_names)
    shap_values = explainer.shap_values(X_df)
    expected_value = float(explainer.expected_value)
    _log.info("Done. expected_value (logit baseline) = %.4f", expected_value)
    return shap_values, expected_value


# ──────────────────────────────────────────────────────────────────────────────
# Global feature importance
# ──────────────────────────────────────────────────────────────────────────────
def top_features_by_mean_abs_shap(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Rank features by mean absolute SHAP across all rows.

    Mean |SHAP| is the cleanest global importance metric — it measures
    average per-decision contribution magnitude in the model's native units.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    })
    return df.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)


def plot_top_features_bar(
    top_features: pd.DataFrame,
    title: str = "Top features by mean |SHAP|",
) -> plt.Figure:
    """Horizontal bar chart of the global importance table."""
    df = top_features.iloc[::-1]  # so largest is at the top of the chart
    fig, ax = plt.subplots(figsize=(7, max(4, len(df) * 0.3)))
    ax.barh(df["feature"], df["mean_abs_shap"], color="#4C72B0")
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(title)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Summary plot (beeswarm)
# ──────────────────────────────────────────────────────────────────────────────
def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    max_display: int = 15,
    title: Optional[str] = "SHAP summary (validation set)",
) -> plt.Figure:
    """Beeswarm summary plot — every point is one applicant on one feature.

    Reads as: x-axis is contribution to the prediction; color is feature
    value (red = high, blue = low). Useful for spotting non-monotone
    relationships at a glance.
    """
    X_df = pd.DataFrame(X, columns=feature_names)
    shap.summary_plot(
        shap_values, X_df, feature_names=feature_names,
        max_display=max_display, show=False,
    )
    fig = plt.gcf()
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Local explanations (per-applicant waterfall)
# ──────────────────────────────────────────────────────────────────────────────
def plot_shap_waterfall(
    shap_values: np.ndarray,
    expected_value: float,
    X: np.ndarray,
    feature_names: List[str],
    idx: int,
    max_display: int = 12,
    title: Optional[str] = None,
) -> plt.Figure:
    """Per-applicant waterfall — the reason code a compliance officer reads."""
    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=expected_value,
        data=X[idx] if not hasattr(X, "iloc") else X.iloc[idx].values,
        feature_names=feature_names,
    )
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def per_applicant_reason_codes(
    shap_values: np.ndarray,
    feature_names: List[str],
    idx: int,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return the `top_n` strongest reason codes for a single applicant.

    Sorted by absolute contribution; sign of the SHAP value tells you
    whether the feature pushed the prediction toward default (+) or away
    from default (-).
    """
    row = shap_values[idx]
    df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": row,
        "abs_shap": np.abs(row),
        "direction": np.where(row > 0, "increases default risk", "decreases default risk"),
    })
    return (df.sort_values("abs_shap", ascending=False)
              .head(top_n)
              .drop(columns="abs_shap")
              .reset_index(drop=True))
