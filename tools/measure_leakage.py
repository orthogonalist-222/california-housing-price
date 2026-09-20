"""Measure what leakage actually costs on this dataset, instead of asserting it.

Every tutorial says "fit your preprocessing inside the fold or you will leak".
That is true. It is also frequently presented with an implied magnitude that
nobody checks, and on this dataset the honest answer is **small** — which is
worth knowing, because a reader who is told to expect a dramatic number and
then measures 0.05% concludes the whole lesson was theatre.

This script runs four comparisons and prints what it observes:

A. Preprocessing fitted once on everything vs. inside each fold.
B. The same, as the training set shrinks (the leak scales with how much the
   learned statistics move).
C. A column derived from the target, added to the input frame.
D. Feature selection on the target, fitted outside the fold.

Run with::

    uv run python tools/measure_leakage.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from calhousing import config
from calhousing.data import load_raw
from calhousing.preprocess.assemble import build_pipeline, build_preprocessor

SCORING = "neg_root_mean_squared_error"


def _cv(estimator, X, y, seed: int) -> tuple[float, float]:
    scores = -cross_val_score(
        estimator, X, y, cv=KFold(5, shuffle=True, random_state=seed), scoring=SCORING
    )
    return float(scores.mean()), float(scores.std())


def part_a(frame: pd.DataFrame, seed: int) -> None:
    print("A. Preprocessing inside the fold vs. fitted once on everything")
    X, y = frame.drop(columns=[config.TARGET]), frame[config.TARGET]
    honest, honest_sd = _cv(build_pipeline(Ridge()), X, y, seed)
    leaked_matrix = build_preprocessor().fit(X).transform(X)
    leaked, leaked_sd = _cv(Ridge(), leaked_matrix, y, seed)
    delta = honest - leaked
    print(f"   honest  RMSE {honest:>10,.0f}  +/- {honest_sd:,.0f}")
    print(f"   leaked  RMSE {leaked:>10,.0f}  +/- {leaked_sd:,.0f}")
    print(f"   leak         {delta:>+10,.0f}  ({100 * delta / honest:+.3f}% of honest)")
    print(f"   -> the leak is {abs(delta) / honest_sd:.2f}x the fold-to-fold spread")
    print()


def part_b(frame: pd.DataFrame, seed: int) -> None:
    print("B. The same leak, as the training set shrinks")
    print(f"   {'n':>8}  {'honest':>10}  {'leaked':>10}  {'leak':>10}")
    for n in (300, 1000, 5000, len(frame)):
        subset = frame.sample(n, random_state=0) if n < len(frame) else frame
        X, y = subset.drop(columns=[config.TARGET]), subset[config.TARGET]
        honest, _ = _cv(build_pipeline(Ridge()), X, y, seed)
        leaked, _ = _cv(Ridge(), build_preprocessor().fit(X).transform(X), y, seed)
        print(
            f"   {n:>8,}  {honest:>10,.0f}  {leaked:>10,.0f}  "
            f"{honest - leaked:>+10,.0f}"
        )
    print()


def part_c(frame: pd.DataFrame, seed: int) -> None:
    print("C. A column derived from the target, added to the input")
    rng = np.random.default_rng(seed)
    X, y = frame.drop(columns=[config.TARGET]), frame[config.TARGET]
    poisoned = X.assign(appraisal=y * 0.98 + rng.normal(0, 5_000, len(y)))

    honest, _ = _cv(build_pipeline(Ridge()), X, y, seed)
    through_assembler, _ = _cv(build_pipeline(Ridge()), poisoned, y, seed)

    # ...and the same column, forced into the model past the assembler.
    matrix = build_preprocessor().fit(X).transform(X)
    matrix["appraisal"] = poisoned["appraisal"].to_numpy()
    forced, _ = _cv(Ridge(), matrix, y, seed)

    print(f"   clean frame                       RMSE {honest:>10,.0f}")
    print(f"   leaky column, through assembler   RMSE {through_assembler:>10,.0f}")
    print(f"   leaky column, forced past it      RMSE {forced:>10,.0f}")
    print("   -> remainder='drop' silently discarded the leaky column: the two")
    print("      upper numbers are the same run. The assembler is a whitelist.")
    print()


def part_d(frame: pd.DataFrame, seed: int) -> None:
    print("D. Feature selection on the target, fitted outside the fold")
    rng = np.random.default_rng(seed)
    X, y = frame.drop(columns=[config.TARGET]), frame[config.TARGET]
    noise = pd.DataFrame(
        rng.normal(size=(len(frame), 200)),
        columns=[f"noise_{i}" for i in range(200)],
        index=frame.index,
    )
    matrix = pd.concat([build_preprocessor().fit(X).transform(X), noise], axis=1)

    inside = Pipeline([("select", SelectKBest(f_regression, k=10)), ("model", Ridge())])
    honest, _ = _cv(inside, matrix, y, seed)
    chosen = SelectKBest(f_regression, k=10).fit(matrix, y).get_support()
    leaked, _ = _cv(Ridge(), matrix.loc[:, chosen], y, seed)
    print(f"   selected inside the fold   RMSE {honest:>10,.0f}")
    print(f"   selected on all the data   RMSE {leaked:>10,.0f}")
    print(f"   leak                            {honest - leaked:>+10,.0f}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args(argv)

    frame = load_raw(args.csv)
    print()
    for part in (part_a, part_b, part_c, part_d):
        part(frame, args.seed)
    print("Read the numbers, not the folklore.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
