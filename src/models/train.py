"""
XGBoost training for the loan-default classifier.

Design choices encoded here:

- Histogram-based tree method (`hist`) — 5–10× faster than `exact` on our
  300K-row training set with negligible accuracy loss. Default in modern
  XGBoost for a reason.
- `scale_pos_weight` for class imbalance instead of resampling. SMOTE-style
  over-sampling distorts SHAP values (the model trains on a different
  distribution than it serves on); scaling the loss preserves both
  calibration and explainability.
- Early stopping on validation PR-AUC, not log loss. PR-AUC is the right
  objective for imbalanced binary classification — log-loss happily declares
  victory when the model predicts ~0 for everyone.
- All hyperparameters are returned as a plain dict so they round-trip
  cleanly through MLflow's `log_params`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import xgboost as xgb

from src.utils.config import load_config
from src.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class TrainResult:
    """Container for the artifacts of a single training run."""
    model: xgb.Booster
    best_iteration: int
    best_score: float
    eval_history: Dict[str, Dict[str, list]]
    params: Dict[str, Any]


def default_xgb_params(y_train: Optional[np.ndarray] = None,
                       use_scale_pos_weight: bool = False) -> Dict[str, Any]:
    """Return a sensible XGBoost parameter set for binary credit-risk classification.

    `scale_pos_weight` is OFF by default — for compliance-driven decisioning
    we prioritise probability calibration over upweighted recall. Set
    `use_scale_pos_weight=True` to opt back in (the value is then computed
    from `y_train`'s class distribution). The returned dict is
    JSON-serialisable so MLflow can log it directly.
    """
    cfg = load_config()
    params: Dict[str, Any] = {
        "objective": "binary:logistic",
        # Track both metrics, but PR-AUC must be LAST: XGBoost uses the
        # last-listed eval_metric for early stopping. Logloss as the first
        # metric is shown alongside for monitoring; we stop on val PR-AUC.
        "eval_metric": ["logloss", "aucpr"],
        "tree_method": "hist",
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "seed": cfg["project"]["random_seed"],
        "verbosity": 1,
    }
    if use_scale_pos_weight and y_train is not None:
        # scale_pos_weight = #negatives / #positives. Tells the loss function
        # to weight each positive example proportionally more. Off by default
        # because it distorts probability calibration; opt in only when the
        # downstream decisioning genuinely needs higher recall and is willing
        # to recalibrate scores separately.
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        if n_pos > 0:
            params["scale_pos_weight"] = float(n_neg / n_pos)
    return params


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 1500,
    early_stopping_rounds: int = 50,
    feature_names: Optional[list] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> TrainResult:
    """Train an XGBoost binary classifier with early stopping on validation PR-AUC.

    `sample_weight` (optional) lets callers up-weight specific training
    rows — used by the feedback-loop pipeline to give analyst-verified
    labels more influence than raw historical labels.

    Returns a `TrainResult` carrying the booster, the iteration at which
    early-stopping kicked in, and the full eval history (useful for plotting
    the learning curves).
    """
    params = params or default_xgb_params(y_train)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight,
                         feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    eval_history: Dict[str, Dict[str, list]] = {}

    _log.info("Training XGBoost: %d boost rounds, early stop = %d",
              num_boost_round, early_stopping_rounds)

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=eval_history,
        verbose_eval=50,
    )

    # XGBoost stores best_iteration as an attribute on the booster.
    best_iter = int(getattr(booster, "best_iteration", num_boost_round - 1))
    # Best score is the PR-AUC at the best iteration on validation.
    best_score = float(eval_history["val"]["aucpr"][best_iter])

    _log.info("Training complete. Best iteration: %d, val PR-AUC: %.4f",
              best_iter, best_score)

    return TrainResult(
        model=booster,
        best_iteration=best_iter,
        best_score=best_score,
        eval_history=eval_history,
        params=params,
    )


def predict_proba(model: xgb.Booster, X: np.ndarray,
                  feature_names: Optional[list] = None,
                  iteration_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Predict positive-class probabilities. Uses the best iteration by default."""
    dmat = xgb.DMatrix(X, feature_names=feature_names)
    if iteration_range is None:
        # Use the booster's best_iteration if available (set by early stopping).
        best = getattr(model, "best_iteration", None)
        if best is not None:
            iteration_range = (0, int(best) + 1)
    return model.predict(dmat, iteration_range=iteration_range)
