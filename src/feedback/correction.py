"""
Analyst-correction feedback loop for the loan-default classifier.

The simulation framing:

1. The baseline model from Step 2 scores a held-out "production pool" of
   applicants we held back from training.
2. Of those, the model's most-confident *wrong* predictions are the cases
   a human analyst would catch on review. False positives at scores ≥ 0.5
   (model said "default", applicant repaid) and false negatives at scores
   ≤ 0.05 (model said "safe", applicant defaulted) are the two failure modes
   compliance cares about most.
3. The simulation adds those analyst-verified rows back into the training set
   with a higher sample weight (representing "human-confirmed labels are
   more trustworthy than auto-generated ones"), retrains XGBoost, and
   measures the accuracy delta against a fresh, never-touched test set.

That accuracy-delta-per-correction-batch is the headline number the
business cares about: "for every 100 analyst hours spent reviewing, how
much does the model improve?"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

_log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Identify candidates for analyst review
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class CorrectionCandidate:
    """A model error a human analyst would flag for review."""
    indices: np.ndarray            # row indices in the production pool
    error_type: str                # 'false_positive' or 'false_negative'
    score_range: Tuple[float, float]
    n: int


def find_high_confidence_errors(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fp_threshold: float = 0.50,
    fn_threshold: float = 0.05,
) -> Tuple[CorrectionCandidate, CorrectionCandidate]:
    """Surface high-confidence false positives and false negatives.

    These are the two failure modes:

    - False positives at score >= fp_threshold: model loudly predicted
      "default" but the applicant repaid. Each one is a denied-then-paid
      loan, lost revenue, and a potentially aggrieved customer.
    - False negatives at score <= fn_threshold: model loudly predicted
      "safe" but the applicant defaulted. Each one is an extended-then-
      defaulted loan, a credit loss the model missed entirely.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    fp_mask = (y_true == 0) & (y_score >= fp_threshold)
    fn_mask = (y_true == 1) & (y_score <= fn_threshold)

    fp = CorrectionCandidate(
        indices=np.where(fp_mask)[0],
        error_type="false_positive",
        score_range=(fp_threshold, float(y_score[fp_mask].max() if fp_mask.any() else fp_threshold)),
        n=int(fp_mask.sum()),
    )
    fn = CorrectionCandidate(
        indices=np.where(fn_mask)[0],
        error_type="false_negative",
        score_range=(float(y_score[fn_mask].min() if fn_mask.any() else 0.0), fn_threshold),
        n=int(fn_mask.sum()),
    )
    _log.info("Correction candidates: %d false positives, %d false negatives",
              fp.n, fn.n)
    return fp, fn


def sample_correction_batch(
    candidates: List[CorrectionCandidate],
    batch_size: int,
    seed: int = 42,
) -> np.ndarray:
    """Pick a balanced sample of correction candidates for one review batch.

    Splits roughly evenly between false positives and false negatives so
    the analyst's time covers both failure modes. Returns row indices into
    the production pool.
    """
    rng = np.random.default_rng(seed)
    per_type = batch_size // 2
    selected: List[np.ndarray] = []
    for cand in candidates:
        if cand.n == 0:
            continue
        n_take = min(per_type, cand.n)
        chosen = rng.choice(cand.indices, size=n_take, replace=False)
        selected.append(chosen)
    return np.concatenate(selected) if selected else np.array([], dtype=int)


# ──────────────────────────────────────────────────────────────────────────────
# Retraining with corrections
# ──────────────────────────────────────────────────────────────────────────────
def build_augmented_training_set(
    X_train_original: pd.DataFrame,
    y_train_original: pd.Series,
    X_corrections: pd.DataFrame,
    y_corrections: pd.Series,
    correction_weight: float = 3.0,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Concatenate the original training set with the analyst-verified rows.

    The correction rows get a higher sample weight (default 3×). The
    reasoning: analyst-confirmed labels are higher-quality than the raw
    historical labels, so the loss function should pay more attention to
    them. Three is conventional but tunable.
    """
    X_aug = pd.concat([X_train_original, X_corrections], axis=0, ignore_index=True)
    y_aug = pd.concat([y_train_original, y_corrections], axis=0, ignore_index=True)

    weights = np.concatenate([
        np.ones(len(X_train_original), dtype="float32"),
        np.full(len(X_corrections), correction_weight, dtype="float32"),
    ])
    _log.info(
        "Augmented training set: %d original + %d corrections "
        "(weight=%.1f×) = %d total rows",
        len(X_train_original), len(X_corrections), correction_weight, len(X_aug),
    )
    return X_aug, y_aug, weights
