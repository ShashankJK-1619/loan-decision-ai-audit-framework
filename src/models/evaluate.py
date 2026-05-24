"""
Evaluation metrics & plotting for the loan-default classifier.

The framing is deliberate: for an 8%-positive-class credit-risk problem,
accuracy is meaningless and ROC-AUC is misleading. The metrics that matter
are:

- **PR-AUC** — area under the precision-recall curve. Sensitive to minority
  class performance in a way ROC-AUC isn't.
- **Recall at a fixed precision threshold** — the operational metric. The
  business doesn't care about a "good" AUC; they care that "at 90% precision
  we catch 35% of true defaulters".
- **Calibration** — predicted probabilities should match observed default
  rates. A model that scores 0.8 should default ~80% of the time. Required
  for downstream decisioning thresholds and capital-reserve calculations.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def recall_at_precision(y_true: np.ndarray, y_score: np.ndarray,
                        precision_target: float) -> Tuple[float, float]:
    """Highest recall achievable while maintaining `precision_target`.

    Returns (recall, decision_threshold). If the precision target is
    unattainable on this dataset, returns (0.0, 1.0).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # `precision_recall_curve` returns one fewer threshold than points
    # (the last point is the (recall=0, precision=1) anchor with no threshold).
    valid = precisions[:-1] >= precision_target
    if not valid.any():
        return 0.0, 1.0
    # Among thresholds meeting the precision floor, pick the one with the
    # highest recall (= lowest threshold).
    best_idx = int(np.argmax(recalls[:-1] * valid))
    return float(recalls[best_idx]), float(thresholds[best_idx])


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray,
                    precision_targets: Tuple[float, ...] = (0.95, 0.90, 0.80, 0.50)
                    ) -> Dict[str, float]:
    """Return the full metric panel as a flat dict (MLflow-loggable)."""
    metrics: Dict[str, float] = {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }
    for p in precision_targets:
        recall, thresh = recall_at_precision(y_true, y_score, p)
        # MLflow metric names must be safe (no '%'); use integer percent.
        metrics[f"recall_at_p{int(p*100)}"] = recall
        metrics[f"threshold_at_p{int(p*100)}"] = thresh
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers — each returns the Figure so callers can mlflow.log_figure
# ──────────────────────────────────────────────────────────────────────────────
def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray,
                  title: str = "Precision-Recall curve") -> plt.Figure:
    """Precision-recall curve with the PR-AUC annotated."""
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    baseline = float(y_true.mean())  # PR-AUC of random classifier

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recalls, precisions, lw=2, color="#4C72B0", label=f"model (AP = {ap:.3f})")
    ax.axhline(baseline, ls="--", color="grey", lw=1,
               label=f"random baseline ({baseline:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray,
                   title: str = "ROC curve") -> plt.Figure:
    """ROC curve with AUC annotated. Less informative than PR for imbalanced data
    but a standard expected visual on any classifier write-up."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, lw=2, color="#C44E52", label=f"model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="random")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def plot_calibration(y_true: np.ndarray, y_score: np.ndarray,
                     n_bins: int = 10,
                     title: str = "Calibration") -> plt.Figure:
    """Reliability diagram. A well-calibrated model sits on the diagonal."""
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=n_bins,
                                            strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(mean_pred, frac_pos, marker="o", lw=2, color="#55A868", label="model")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="perfect calibration")
    ax.set_xlabel("mean predicted probability (bin)")
    ax.set_ylabel("observed default rate (bin)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def plot_confusion(y_true: np.ndarray, y_score: np.ndarray,
                   threshold: float,
                   title: Optional[str] = None) -> plt.Figure:
    """Confusion matrix at a specific decision threshold."""
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred: repaid", "pred: defaulted"])
    ax.set_yticklabels(["true: repaid", "true: defaulted"])
    ax.set_title(title or f"Confusion matrix @ threshold = {threshold:.3f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
