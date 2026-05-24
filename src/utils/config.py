"""
Configuration loading utilities.

Centralising config access means every notebook, module, and test resolves
paths and hyperparameters through the same code path. This eliminates
"works on my machine" reproducibility issues that compliance teams flag.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


# Project root resolved once at import time so callers don't have to fiddle
# with relative paths regardless of cwd (notebook, CLI, Streamlit, tests).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class Paths:
    """Strongly-typed access to filesystem locations declared in config.yaml."""
    data_raw: Path
    data_interim: Path
    data_processed: Path
    artifacts: Path
    mlflow_tracking_uri: str
    mlflow_artifact_root: Path


@lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> Dict[str, Any]:
    """Load and cache the YAML config. Cached because it's read repeatedly."""
    cfg_path = Path(path) if path else CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def get_paths(cfg: Dict[str, Any] | None = None) -> Paths:
    """Resolve all configured paths to absolute Path objects."""
    cfg = cfg or load_config()
    raw = cfg["paths"]
    # MLflow tracking URI: if it ends with .db, treat as SQLite backend and
    # emit a sqlite:// URI. Otherwise pass through unchanged (so a remote
    # tracking server URL or filesystem path both work).
    tracking_raw = raw["mlflow_tracking_uri"]
    tracking_abs = PROJECT_ROOT / tracking_raw
    if str(tracking_raw).endswith(".db"):
        tracking_uri = f"sqlite:///{tracking_abs}"
    else:
        tracking_uri = str(tracking_abs)
    return Paths(
        data_raw=PROJECT_ROOT / raw["data_raw"],
        data_interim=PROJECT_ROOT / raw["data_interim"],
        data_processed=PROJECT_ROOT / raw["data_processed"],
        artifacts=PROJECT_ROOT / raw["artifacts"],
        mlflow_tracking_uri=tracking_uri,
        mlflow_artifact_root=PROJECT_ROOT / raw.get("mlflow_artifact_root", "artifacts/mlruns"),
    )
