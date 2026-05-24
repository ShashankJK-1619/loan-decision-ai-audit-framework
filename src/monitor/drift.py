"""
Drift monitoring for the loan-default classifier.

Two things this module does:

1. **PSI (Population Stability Index)** — the industry-standard univariate
   drift metric in credit risk. PSI = sum over bins of
   `(p_current - p_reference) * ln(p_current / p_reference)`. Interpretation
   thresholds are conventional:
     - PSI < 0.10  : no significant change
     - 0.10–0.25   : moderate drift, monitor
     - PSI > 0.25  : significant drift, investigate
   PSI is computed per feature; rolling it up across features is not
   meaningful (different features have different scales).

2. **Synthetic time-window generation** — simulates six months of
   production data by sampling from the validation set with progressively
   stronger biases applied. The goal is to make drift detection
   demonstrable on a static dataset: real production traffic would drift
   on its own, but we have a snapshot, so we inject controlled shifts.

The bias schedule is deliberate, not random: month 1 is near-baseline,
months 2-3 add mild income drift, months 4-5 add age + employment drift,
month 6 adds severe combined drift. A PSI tracker watching this should
show a clean monotonic trend, validating the detection logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

_log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# PSI — Population Stability Index
# ──────────────────────────────────────────────────────────────────────────────
def _bin_edges_from_reference(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Compute quantile-based bin edges on the reference distribution.

    Quantile binning is the right choice for PSI: each reference bin
    contains the same fraction of mass, so any deviation in the current
    distribution is comparable across bins. Equal-width binning would
    over-weight the tails for skewed features like income.
    """
    ref = pd.Series(reference).dropna().values
    # Use unique quantiles to avoid degenerate (zero-width) bins on highly
    # discrete features like FLAG_DOCUMENT_3.
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    # Extend the outermost edges to absorb any current-period extremes.
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def psi_numeric(reference: np.ndarray, current: np.ndarray,
                n_bins: int = 10, smoothing: float = 1e-4) -> float:
    """Compute PSI between a reference and current numeric distribution.

    Smoothing avoids division-by-zero when a bin is empty in one period;
    1e-4 is a conventional regularization weight used in credit-risk PSI
    tooling.
    """
    ref = pd.Series(reference).dropna().values
    cur = pd.Series(current).dropna().values
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")

    edges = _bin_edges_from_reference(ref, n_bins=n_bins)
    if len(edges) < 3:
        # Too few unique values — PSI is undefined.
        return float("nan")

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_pct = (ref_counts + smoothing) / (ref_counts.sum() + smoothing * len(ref_counts))
    cur_pct = (cur_counts + smoothing) / (cur_counts.sum() + smoothing * len(cur_counts))

    psi = float(((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum())
    return psi


def psi_categorical(reference: pd.Series, current: pd.Series,
                    smoothing: float = 1e-4) -> float:
    """PSI on a categorical column. Treats each unique level as its own bin."""
    ref = pd.Series(reference).dropna()
    cur = pd.Series(current).dropna()
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    levels = sorted(set(ref.unique()) | set(cur.unique()), key=str)
    ref_counts = ref.value_counts().reindex(levels, fill_value=0).values
    cur_counts = cur.value_counts().reindex(levels, fill_value=0).values
    ref_pct = (ref_counts + smoothing) / (ref_counts.sum() + smoothing * len(levels))
    cur_pct = (cur_counts + smoothing) / (cur_counts.sum() + smoothing * len(levels))
    return float(((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum())


def psi_table(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Optional[List[str]] = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute PSI for every column. Auto-routes to numeric vs categorical."""
    columns = columns or list(set(reference.columns) & set(current.columns))
    rows = []
    for col in columns:
        if pd.api.types.is_numeric_dtype(reference[col]):
            psi = psi_numeric(reference[col].values, current[col].values, n_bins=n_bins)
        else:
            psi = psi_categorical(reference[col], current[col])
        rows.append({
            "feature": col,
            "psi": psi,
            "severity": _psi_severity(psi),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def _psi_severity(psi: float) -> str:
    if pd.isna(psi):
        return "n/a"
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "significant"


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic time-window generator
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class MonthlyBatch:
    month: int
    label: str
    description: str
    df: pd.DataFrame
    target: pd.Series


def generate_synthetic_months(
    df: pd.DataFrame,
    target: pd.Series,
    n_per_month: int = 5000,
    seed: int = 42,
) -> List[MonthlyBatch]:
    """Carve six monthly batches out of a held-out frame with progressive drift.

    Returns a list of `MonthlyBatch` objects, indices 0..5 corresponding to
    months 1..6. Each batch's `description` documents the bias applied,
    making the drift narrative auditable rather than hidden inside a
    sampling function.
    """
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    target = target.reset_index(drop=True)

    def _sample_with_bias(weights: np.ndarray, n: int) -> pd.Index:
        """Weighted sample without replacement; n must be <= len(weights)."""
        w = weights / weights.sum()
        idx = rng.choice(len(df), size=n, replace=False, p=w)
        return pd.Index(idx)

    batches: List[MonthlyBatch] = []

    # ── Month 1 — baseline, no bias ─────────────────────────────────────
    idx = pd.Index(rng.choice(len(df), size=n_per_month, replace=False))
    batches.append(MonthlyBatch(
        month=1, label="2026-01",
        description="Baseline — uniform sample, no injected drift.",
        df=df.loc[idx].copy(), target=target.loc[idx].copy(),
    ))

    # ── Month 2 — mild income skew (slightly higher-income applicants) ──
    income = df["AMT_INCOME_TOTAL"].fillna(df["AMT_INCOME_TOTAL"].median()).values
    w = np.exp((income - income.mean()) / income.std() * 0.3)  # mild
    idx = _sample_with_bias(w, n_per_month)
    batches.append(MonthlyBatch(
        month=2, label="2026-02",
        description="Mild income skew upward (marketing channel shift).",
        df=df.loc[idx].copy(), target=target.loc[idx].copy(),
    ))

    # ── Month 3 — stronger income skew + younger applicants ─────────────
    age_days = df["DAYS_BIRTH"].fillna(df["DAYS_BIRTH"].median()).values
    # Younger = less negative DAYS_BIRTH; weight those higher.
    w_age = np.exp((age_days - age_days.mean()) / age_days.std() * 0.4)
    w = np.exp((income - income.mean()) / income.std() * 0.5) * w_age
    idx = _sample_with_bias(w, n_per_month)
    batches.append(MonthlyBatch(
        month=3, label="2026-03",
        description="Income skew + younger applicants (product expansion).",
        df=df.loc[idx].copy(), target=target.loc[idx].copy(),
    ))

    # ── Month 4 — employment drift (more sentinel/unemployed) ────────────
    anom = df.get("DAYS_EMPLOYED_ANOM", pd.Series(0, index=df.index)).fillna(0).values
    w = 1.0 + 2.0 * anom  # 3× weight on sentinel rows
    idx = _sample_with_bias(w, n_per_month)
    batches.append(MonthlyBatch(
        month=4, label="2026-04",
        description="Employment-record drift (more unemployed/retiree applicants).",
        df=df.loc[idx].copy(), target=target.loc[idx].copy(),
    ))

    # ── Month 5 — lower external scores (riskier population) ────────────
    ext = df.get("EXT_SOURCE_2", pd.Series(0.5, index=df.index)).fillna(0.5).values
    w = np.exp(-(ext - ext.mean()) / max(ext.std(), 1e-6) * 1.0)
    idx = _sample_with_bias(w, n_per_month)
    batches.append(MonthlyBatch(
        month=5, label="2026-05",
        description="Bureau-score drift downward (acquisition into lower-prime).",
        df=df.loc[idx].copy(), target=target.loc[idx].copy(),
    ))

    # ── Month 6 — severe combined drift (broken acquisition funnel) ─────
    w_ext = np.exp(-(ext - ext.mean()) / max(ext.std(), 1e-6) * 1.5)
    w_anom = 1.0 + 3.0 * anom
    w_income_low = np.exp(-(income - income.mean()) / income.std() * 0.5)
    w = w_ext * w_anom * w_income_low
    idx = _sample_with_bias(w, n_per_month)
    batches.append(MonthlyBatch(
        month=6, label="2026-06",
        description="Severe combined drift (low scores + unemployed + low income).",
        df=df.loc[idx].copy(), target=target.loc[idx].copy(),
    ))

    for b in batches:
        _log.info("Month %d (%s) — %d rows, base rate %.2f%% — %s",
                  b.month, b.label, len(b.df), 100 * b.target.mean(), b.description)
    return batches


# ──────────────────────────────────────────────────────────────────────────────
# Performance tracking over time
# ──────────────────────────────────────────────────────────────────────────────
def performance_by_month(
    batches: List[MonthlyBatch],
    score_func,
    decision_threshold: float,
) -> pd.DataFrame:
    """Compute headline performance metrics for each monthly batch.

    `score_func` is a callable taking a feature DataFrame and returning
    predicted probabilities. Decoupling lets the same monitor work over the
    raw XGBoost model, a retrained model, or a stub for testing.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    from src.models.evaluate import recall_at_precision

    rows = []
    for b in batches:
        scores = score_func(b.df)
        y = b.target.values
        if y.sum() > 0:
            pr_auc = float(average_precision_score(y, scores))
            roc_auc = float(roc_auc_score(y, scores))
            recall_p50, _ = recall_at_precision(y, scores, 0.50)
        else:
            pr_auc = roc_auc = recall_p50 = float("nan")
        approval_rate = float((scores < decision_threshold).mean())
        rows.append({
            "month": b.month,
            "label": b.label,
            "n": len(b.df),
            "base_rate": float(b.target.mean()),
            "approval_rate": approval_rate,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "recall_at_p50": recall_p50,
            "mean_score": float(np.mean(scores)),
        })
    return pd.DataFrame(rows)
