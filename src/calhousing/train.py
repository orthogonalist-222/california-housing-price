"""Search preprocessing and model together — Spark's ``ParamGridBuilder``.

The point of this module is one line of configuration::

    space = {**PREPROCESS_SEARCH_SPACE, **spec.search_space}

``RandomizedSearchCV`` receives one estimator — the whole pipeline — and a
space that mixes ``model__max_depth`` with ``preprocess__geo__n_clusters`` and
``preprocess__heavy__deskew``. Every draw refits the imputers, the encoders,
the bin edges and the KMeans centroids from the fold's own training rows, and
the number of geographic clusters is chosen by the same evidence that chooses
the tree depth.

A pipeline assembled by hand out of ``fit_transform`` calls can tune the model.
It cannot tune the preprocessing, because by the time the model is being tuned
the preprocessing has already happened. That is the practical reason the
discipline is worth its cost, and it is more convincing than the leakage
argument — which, measured on this dataset, is worth 0.05% (see
``tools/measure_leakage.py``).

Run with::

    uv run python -m calhousing.train --model hgb --n-iter 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from . import config
from .data import load_raw
from .evaluate import segment_table
from .models import (
    PREPROCESS_SEARCH_SPACE,
    REGISTRY,
    build_cv,
    get_model,
)
from .preprocess.assemble import build_pipeline
from .splits import split_train_test

__all__ = ["search_space_for", "run_search", "DEFAULT_OUTPUT_DIR"]

DEFAULT_OUTPUT_DIR = Path("artifacts/search")


def search_space_for(name: str) -> dict[str, Any]:
    """The joint space: this model's parameters **and** the preprocessing's."""
    return {**PREPROCESS_SEARCH_SPACE, **get_model(name).search_space}


def run_search(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_iter: int = 30,
    n_splits: int = 5,
    seed: int = config.RANDOM_SEED,
    n_jobs: int = -1,
    verbose: bool = True,
) -> RandomizedSearchCV:
    """Randomised search over the whole pipeline.

    Randomised rather than exhaustive: the joint space here is over 4 000
    combinations, and a grid would spend its budget proving that
    ``n_clusters=5`` is bad four hundred times. With a fixed seed the draw is
    reproducible, which is what makes the budget a *recorded* choice rather
    than a thing that happened.
    """
    spec = get_model(name)
    space = search_space_for(name)
    search = RandomizedSearchCV(
        estimator=build_pipeline(spec.factory()),
        param_distributions=space,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=build_cv(n_splits, seed),
        random_state=seed,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
        error_score="raise",  # a failed draw is a defect, not a low score
    )
    started = time.time()
    search.fit(X, y)
    elapsed = time.time() - started

    if verbose:
        print(f"[{name}] {n_iter} draws x {n_splits} folds in {elapsed:,.0f}s")
        print(f"[{name}] best CV RMSE {-search.best_score_:,.0f}")
        for key, value in sorted(search.best_params_.items()):
            print(f"        {key} = {_short(value)}")
    return search


def _short(value: Any) -> str:
    """Render a parameter value without a wall of estimator repr."""
    if hasattr(value, "__class__") and hasattr(value, "get_params"):
        return value.__class__.__name__
    return repr(value)


def leaderboard(search: RandomizedSearchCV, top: int = 5) -> pd.DataFrame:
    """The best draws, with train and test scores side by side.

    Both columns, always: a draw whose train RMSE is half its test RMSE won the
    search by overfitting, and a leaderboard showing only the test column hides
    that.
    """
    frame = pd.DataFrame(search.cv_results_)
    frame = frame.assign(
        cv_rmse=-frame["mean_test_score"],
        train_rmse=-frame["mean_train_score"],
        cv_rmse_std=frame["std_test_score"],
    )
    columns = ["cv_rmse", "cv_rmse_std", "train_rmse", "mean_fit_time"]
    return frame.nsmallest(top, "cv_rmse")[columns].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search a model jointly with the pipeline.")
    parser.add_argument("--model", default="hgb", choices=sorted(REGISTRY))
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--csv", default=None)
    args = parser.parse_args(argv)

    frame = load_raw(args.csv)
    train, _test = split_train_test(frame, seed=args.seed)
    # `_test` is deliberately unused and deliberately named. The test split is
    # scored exactly once, in M3-S4, per ADR-003.
    X = train.drop(columns=[config.TARGET])
    y = train[config.TARGET]

    search = run_search(
        args.model,
        X,
        y,
        n_iter=args.n_iter,
        n_splits=args.n_splits,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    print()
    print(leaderboard(search).round(1).to_string())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, out / f"{args.model}-best.joblib")
    (out / f"{args.model}-best.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "cv_rmse": float(-search.best_score_),
                "n_iter": args.n_iter,
                "n_splits": args.n_splits,
                "seed": args.seed,
                "best_params": {k: _short(v) for k, v in search.best_params_.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[{args.model}] saved to {out}")

    # In-sample segment table on the TRAINING split, to see where the error
    # lives before the test set is ever touched.
    predictions = search.best_estimator_.predict(X)
    print()
    print("training-split segments (not a test score):")
    print(segment_table(X, y, predictions).round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
