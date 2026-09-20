"""Metrics — including the ones a headline number hides.

ADR-001 decided to keep the 965 rows pinned at the ``$500,001`` cap rather than
drop them. That decision is only honest if every evaluation reports where the
error lives, because **a model cannot be right about a censored row**: the true
value is unknown and above the cap, so any prediction below it is wrong by an
unknowable amount and any prediction above it is penalised for being closer to
the truth.

A single RMSE over the whole test set averages that unanswerable 4.7% together
with the 95.3% the model can actually be judged on. Both numbers are reported
here, always, side by side.

The second split is geographic. ``INLAND`` is a third of the data and a
different price regime; a model that is excellent on the coast and poor inland
has a flattering average and a specific, describable weakness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config

__all__ = [
    "METRIC_NAMES",
    "regression_metrics",
    "censored_mask",
    "inland_mask",
    "segment_table",
    "compare",
]

METRIC_NAMES = ("rmse", "mae", "r2", "n")


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """RMSE, MAE, R² and the row count, as a plain dict.

    ``n`` is included deliberately: every segment table below reports metrics
    over subsets, and an RMSE computed on five rows deserves to be read
    differently from one computed on four thousand. A table that omits the
    count invites exactly that mistake.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        # R² is undefined for a single row (zero variance); report NaN rather
        # than sklearn's warning-plus-0.0, which reads as a real score.
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "n": int(len(y_true)),
    }


def censored_mask(y) -> np.ndarray:
    """True where the target sits on the cap or the floor (ADR-001)."""
    values = np.asarray(y, dtype=float)
    return (values >= config.TARGET_CAP) | (values <= config.TARGET_FLOOR)


def inland_mask(X: pd.DataFrame) -> np.ndarray:
    """True for the ``INLAND`` price regime."""
    return (X["ocean_proximity"] == "INLAND").to_numpy()


def segment_table(X: pd.DataFrame, y_true, y_pred) -> pd.DataFrame:
    """Metrics overall and by the two segments that carry a decision.

    Returns one row per segment with ``METRIC_NAMES`` as columns. The ``all``
    row is first so the headline is readable, and everything after it explains
    where that headline came from.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    censored = censored_mask(y_true)
    inland = inland_mask(X)

    segments: dict[str, np.ndarray] = {
        "all": np.ones(len(y_true), dtype=bool),
        "uncensored": ~censored,
        "censored": censored,
        "inland": inland,
        "coastal": ~inland,
    }
    rows = {
        name: regression_metrics(y_true[mask], y_pred[mask])
        for name, mask in segments.items()
    }
    return pd.DataFrame(rows).T[list(METRIC_NAMES)].astype(
        {"n": int}
    )


def compare(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Several models' metrics as one table, best RMSE first.

    Sorted rather than left in insertion order, because a comparison table
    whose ordering is arbitrary invites the reader to compare adjacent rows
    instead of the whole.
    """
    table = pd.DataFrame(results).T
    if "rmse" in table.columns:
        table = table.sort_values("rmse")
    return table
