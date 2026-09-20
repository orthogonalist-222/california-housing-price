"""What the model learned — permutation importance and partial dependence.

Two deliberate choices.

**Permutation importance, not the tree's own ``feature_importances_``.** An
impurity-based importance is computed on the *training* data and is biased
toward high-cardinality features — which, in a pipeline that emits twenty
continuous cluster-similarity columns beside four binary one-hot columns, is a
bias pointed straight at the features this project engineered. Permutation
importance measures what the score actually loses when a column is scrambled,
on data the model did not fit.

**It is measured on a validation slice of the TRAINING split, never on the test
set.** ADR-003 spends the test set once, on scores. An importance plot computed
from it would be a second look.

Both outputs are figures rather than tables because their job is to be argued
with, and the shape of a partial-dependence curve carries information no
three-column table does.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.pipeline import Pipeline

from . import config
from .preprocess.assemble import feature_names

__all__ = ["importance_table", "plot_importance", "plot_partial_dependence"]

DEFAULT_OUTPUT_DIR = Path("artifacts/interpret")

_WATER = "#2a6f97"
_ACCENT = "#9d0208"


def importance_table(
    fitted: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_repeats: int = 10,
    seed: int = config.RANDOM_SEED,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Permutation importance over the **raw input columns**.

    Permuting the raw columns rather than the assembled features is the honest
    unit here: ``total_rooms`` feeds three branches, and scrambling only one of
    its derived features would leave the other two intact and report the column
    as unimportant. Permuting the source asks the question a reader means —
    *what does this measurement contribute?*
    """
    result = permutation_importance(
        fitted,
        X,
        y,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="neg_root_mean_squared_error",
        n_jobs=n_jobs,
    )
    return (
        pd.DataFrame(
            {
                "column": X.columns,
                "rmse_increase": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("rmse_increase", ascending=False)
        .reset_index(drop=True)
    )


def plot_importance(table: pd.DataFrame, out: Path, *, title: str = "") -> Path:
    """Horizontal bars, with the error bars kept.

    An importance whose spread crosses zero is an importance the data does not
    support, and a bar chart without error bars invites exactly that reading.
    """
    path = Path(out) / "01-permutation-importance.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered = table.sort_values("rmse_increase")
    colors = [
        _ACCENT if (m - s) <= 0 else _WATER
        for m, s in zip(ordered["rmse_increase"], ordered["std"])
    ]

    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(ordered) + 1.6))
    ax.barh(
        ordered["column"],
        ordered["rmse_increase"],
        xerr=ordered["std"],
        color=colors,
        error_kw={"ecolor": "#444", "elinewidth": 0.9},
    )
    ax.axvline(0, color="#444", lw=0.9)
    ax.set_xlabel("RMSE increase when the column is shuffled ($)")
    ax.set_title(
        title or "Permutation importance (red: spread reaches zero)",
        loc="left",
        fontsize=10.5,
    )
    ax.grid(axis="y", visible=False)
    fig.savefig(path, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return path


def plot_partial_dependence(
    fitted: Pipeline,
    X: pd.DataFrame,
    features: list[str],
    out: Path,
) -> Path:
    """Partial dependence for a couple of columns worth arguing about."""
    path = Path(out) / "02-partial-dependence.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(features), figsize=(4.3 * len(features), 3.6))
    axes = np.atleast_1d(axes)
    PartialDependenceDisplay.from_estimator(
        fitted,
        X,
        features=features,
        ax=axes,
        line_kw={"color": _WATER, "lw": 2},
    )
    for ax, name in zip(axes, features):
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Partial dependence — the shape the model learned",
        x=0.01,
        ha="left",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return path
