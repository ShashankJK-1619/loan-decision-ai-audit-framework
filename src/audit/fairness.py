"""
Fairness audit by segment.

This module slices the validation metrics by protected and proxy attributes
and reports:

- **Approval-rate parity**: at a fixed decision threshold, does each group
  get approved/rejected at similar rates?
- **Recall parity**: among true defaulters in each group, what fraction
  does the model flag?
- **Calibration parity**: does a score of 0.X mean roughly the same default
  rate across groups?

These three metrics jointly cover the EEOC- and ECOA-style fairness
analyses regulators expect on consumer credit models. None of them is
sufficient on its own; together they triangulate whether the model exhibits
disparate impact.

Note on the current Step 2 model: it was trained with `CODE_GENDER` and
`NAME_FAMILY_STATUS` *as features*. In regulated lending, that is generally
not acceptable — protected attributes should be used only for audit, not
for prediction. The audit framework will surface the resulting disparity
regardless; remediation (retraining without the protected attribute) is a
future step.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.evaluate import recall_at_precision
from src.utils.logging import get_logger

_log = get_logger(__name__)

# Below this row count, a slice is considered too small to report
# meaningful metrics on. Surfacing the count alongside the metric is the
# right call when the cohort is borderline.
MIN_SEGMENT_SIZE: int = 200


def metrics_by_segment(
    y_true: np.ndarray,
    y_score: np.ndarray,
    segments: pd.Series,
    decision_threshold: float = 0.5,
    precision_target: float = 0.50,
) -> pd.DataFrame:
    """Compute per-segment fairness metrics.

    Returns a row per unique value in `segments` with the count, base rate,
    approval rate at the threshold, recall at the given precision target,
    and the mean predicted score (a quick calibration proxy).

    Segments with fewer than `MIN_SEGMENT_SIZE` rows are reported but
    flagged in a `small_segment` column — caller decides whether to drop or
    annotate them in plots.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows: List[Dict] = []
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    for value in sorted(pd.Series(segments).dropna().unique(), key=str):
        mask = (segments == value).values
        n = int(mask.sum())
        if n == 0:
            continue
        y_t = y_true[mask]
        y_s = y_score[mask]

        # `approval rate` = share where the model decides "not default" at
        # the given threshold. If decision_threshold is high, we approve
        # more applicants (predicted default rate is lower).
        predicted_default = (y_s >= decision_threshold).astype(int)
        approval_rate = float(1 - predicted_default.mean())

        # Recall at fixed precision — but only computed when there are
        # positives in the segment.
        if y_t.sum() > 0:
            recall, _ = recall_at_precision(y_t, y_s, precision_target)
            pr_auc = float(average_precision_score(y_t, y_s))
            try:
                roc_auc = float(roc_auc_score(y_t, y_s))
            except ValueError:
                roc_auc = float("nan")
        else:
            recall = float("nan")
            pr_auc = float("nan")
            roc_auc = float("nan")

        rows.append({
            "segment": value,
            "n": n,
            "small_segment": n < MIN_SEGMENT_SIZE,
            "base_rate": float(y_t.mean()) if n else float("nan"),
            "approval_rate": approval_rate,
            f"recall_at_p{int(precision_target * 100)}": recall,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "mean_score": float(y_s.mean()) if n else float("nan"),
        })

    return pd.DataFrame(rows)


def disparity_summary(
    seg_metrics: pd.DataFrame,
    reference_segment: Optional[str] = None,
    metric_col: str = "approval_rate",
) -> pd.DataFrame:
    """Compute disparity ratios relative to a reference segment.

    The classic four-fifths rule in EEOC analysis says any group's approval
    rate should be at least 80% of the reference group's. This function
    computes that ratio so a reviewer can scan for violations directly.
    """
    if reference_segment is None:
        # Default: take the largest segment as reference.
        reference_segment = seg_metrics.loc[seg_metrics["n"].idxmax(), "segment"]
    ref_row = seg_metrics[seg_metrics["segment"] == reference_segment]
    if ref_row.empty:
        raise ValueError(f"reference_segment={reference_segment!r} not in metrics")
    ref_value = float(ref_row[metric_col].iloc[0])

    out = seg_metrics.copy()
    out["disparity_ratio"] = out[metric_col] / ref_value if ref_value else float("nan")
    out["four_fifths_violation"] = out["disparity_ratio"] < 0.80
    out.attrs["reference_segment"] = reference_segment
    out.attrs["metric_col"] = metric_col
    return out


def income_decile(income: pd.Series) -> pd.Series:
    """Bucket an income column into decile labels (D1 = lowest, D10 = highest).

    `qcut` handles ties and missing values; the result is a categorical
    Series that groupby can use directly.
    """
    deciles = pd.qcut(income.rank(method="first"), 10,
                       labels=[f"D{i+1}" for i in range(10)])
    return deciles.astype("string")


def plot_segment_metric(
    seg_metrics: pd.DataFrame,
    metric: str,
    title: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of one metric across segments, with small-segment markers."""
    df = seg_metrics.copy()
    if "small_segment" not in df:
        df["small_segment"] = False

    colors = ["#C44E52" if small else "#4C72B0" for small in df["small_segment"]]
    fig, ax = plt.subplots(figsize=(7, max(3, len(df) * 0.4)))
    bars = ax.barh(df["segment"].astype(str), df[metric], color=colors)
    # Annotate each bar with its value and sample size.
    for bar, value, n in zip(bars, df[metric], df["n"]):
        if pd.notna(value):
            ax.text(bar.get_width(),
                    bar.get_y() + bar.get_height() / 2,
                    f"  {value:.3f}  (n={n:,})",
                    va="center", fontsize=9)
    ax.set_xlabel(metric)
    ax.set_title(title or f"{metric} by segment")
    ax.set_xlim(0, max(df[metric].dropna().max() * 1.25, 0.1))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
