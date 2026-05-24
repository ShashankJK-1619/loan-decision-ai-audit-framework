"""
Structured logging setup.

A single configured logger is shared across notebooks and modules so log lines
are uniformly formatted — important when an MLflow run captures stdout/stderr
as artifacts and a reviewer reads them later in isolation.
"""
from __future__ import annotations

import logging
import sys
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "loan_audit", level: int = logging.INFO) -> logging.Logger:
    """Return a singleton-style configured logger.

    Idempotent — calling twice with the same name reuses the existing handler
    instead of stacking duplicates (a common gotcha when re-running notebook
    cells).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured — return as is to avoid duplicate log lines.
        return logger

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
