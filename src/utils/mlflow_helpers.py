"""
MLflow tracking helpers.

Centralising the tracking URI and experiment name here means:
- A reviewer changing where runs are stored edits one file, not every notebook.
- The Streamlit dashboard in Step 6 resolves model versions through the same
  `mlflow.MlflowClient` configuration as training.
"""
from __future__ import annotations

import os
from typing import Optional

import mlflow

from src.utils.config import get_paths, load_config
from src.utils.logging import get_logger

_log = get_logger(__name__)

DEFAULT_EXPERIMENT = "loan-default-baseline"


def configure_mlflow(experiment: Optional[str] = None) -> str:
    """Set the tracking URI and select / create an experiment.

    Uses a SQLite backend for run metadata (the file-based store is
    deprecated as of MLflow 3.x) and a local folder for artifact storage.
    Returns the experiment id.
    """
    paths = get_paths()
    tracking_uri = paths.mlflow_tracking_uri
    artifact_root = paths.mlflow_artifact_root

    # Ensure the parent dir for the SQLite db exists, and the artifact root.
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)

    exp_name = experiment or DEFAULT_EXPERIMENT
    # Create the experiment with an explicit artifact_location the first time
    # it appears. set_experiment() alone doesn't accept artifact_location.
    existing = mlflow.get_experiment_by_name(exp_name)
    if existing is None:
        mlflow.create_experiment(
            exp_name,
            artifact_location=f"file://{artifact_root}",
        )
    exp = mlflow.set_experiment(exp_name)

    _log.info("MLflow tracking URI: %s | artifact root: %s | experiment: %s (id=%s)",
              tracking_uri, artifact_root, exp_name, exp.experiment_id)
    return exp.experiment_id
