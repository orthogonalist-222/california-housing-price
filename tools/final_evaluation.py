"""The single test-set evaluation. Run once. Read ADR-003 first.

This script exists so that the one-shot check is a **recipe**, not an
afternoon. It scores every arm named in ADR-003 inside one invocation, against
one split, and writes what it observed. Running it twice and keeping the better
output is the failure the whole protocol exists to prevent — so it refuses to
overwrite its own record unless explicitly told to.

The arms, fixed in ADR-003 before any of them existed:

1. ``dummy`` — the floor.
2. ``ridge`` — tuned linear baseline.
3. ``rf``   — best tree ensemble from M3-S2 (re-searched in M3-S3).
4. ``lgbm`` — best gradient-boosting arm from M3-S3.
5. ``stack``— the stacked ensemble.

Each records the **full segment table**, not a headline: overall, uncensored,
censored, inland and coastal, each with its row count. ADR-001 kept the
censored rows on exactly that condition.

Run with::

    uv run python tools/final_evaluation.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from calhousing import config
from calhousing.data import load_raw
from calhousing.evaluate import segment_table
from calhousing.interpret import (
    importance_table,
    plot_importance,
    plot_partial_dependence,
)
from calhousing.models import STACK_MEMBERS, STACK_PREPROCESS, TUNED, tuned_pipeline
from calhousing.splits import split_train_test

#: Exactly the arms ADR-003 named. Not a parameter.
ARMS = ("dummy", "ridge", "rf", "lgbm", "stack")

DEFAULT_OUTPUT_DIR = Path("artifacts/final")
RECORD_NAME = "test-set-evaluation.json"


def run(
    frame: pd.DataFrame,
    out: Path,
    *,
    seed: int = config.RANDOM_SEED,
    interpret: bool = True,
) -> dict:
    """Fit every arm on train, score the test split once, record everything."""
    train, test = split_train_test(frame, seed=seed)
    X_train = train.drop(columns=[config.TARGET])
    y_train = train[config.TARGET]
    X_test = test.drop(columns=[config.TARGET])
    y_test = test[config.TARGET]

    print(f"train {len(train):,}  test {len(test):,}  seed {seed}")
    print()

    tables: dict[str, pd.DataFrame] = {}
    timings: dict[str, float] = {}
    fitted_models = {}

    for arm in ARMS:
        started = time.time()
        pipeline = tuned_pipeline(arm)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        timings[arm] = time.time() - started
        tables[arm] = segment_table(X_test, y_test, predictions)
        fitted_models[arm] = pipeline
        print(
            f"  {arm:6s} test RMSE {tables[arm].loc['all', 'rmse']:>9,.0f}"
            f"   fit+score {timings[arm]:>6.1f}s"
        )

    headline = (
        pd.DataFrame(
            {arm: table.loc["all"] for arm, table in tables.items()}
        )
        .T.sort_values("rmse")
    )

    print()
    print("headline (test set, scored once):")
    print(headline.round(1).to_string())
    print()
    for arm, table in tables.items():
        print(f"{arm} - by segment:")
        print(table.round(1).to_string())
        print()

    out.mkdir(parents=True, exist_ok=True)
    record = {
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": seed,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "arms": list(ARMS),
        "stack_members": list(STACK_MEMBERS),
        "stack_preprocess": STACK_PREPROCESS,
        "tuned_params": TUNED,
        "timings_seconds": {k: round(v, 2) for k, v in timings.items()},
        "segments": {arm: table.to_dict(orient="index") for arm, table in tables.items()},
    }
    (out / RECORD_NAME).write_text(json.dumps(record, indent=2), encoding="utf-8")
    headline.to_csv(out / "headline.csv")
    print(f"recorded -> {out / RECORD_NAME}")

    if interpret:
        best = headline.index[0]
        print(f"\ninterpreting the best arm ({best}) on a TRAINING slice, not the test set")
        # A validation slice of the TRAINING split. ADR-003 spends the test set
        # on scores; an importance plot computed from it would be a second look.
        slice_X = X_train.sample(min(2000, len(X_train)), random_state=seed)
        slice_y = y_train.loc[slice_X.index]
        table = importance_table(fitted_models[best], slice_X, slice_y, n_repeats=5)
        print(table.round(1).to_string(index=False))
        plot_importance(table, out, title=f"Permutation importance — {best}")
        plot_partial_dependence(
            fitted_models[best], slice_X, ["median_income", "housing_median_age"], out
        )
        record["importance"] = table.to_dict(orient="records")
        (out / RECORD_NAME).write_text(json.dumps(record, indent=2), encoding="utf-8")

    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--csv", default=None)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--no-interpret", action="store_true")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help=(
            "Overwrite an existing record. Read ADR-003 before using this: the "
            "test set is a one-shot check, and a second run whose output "
            "replaces the first is indistinguishable from tuning on it."
        ),
    )
    args = parser.parse_args(argv)

    out = Path(args.out)
    existing = out / RECORD_NAME
    if existing.exists() and not args.rerun:
        previous = json.loads(existing.read_text(encoding="utf-8"))
        print(
            f"REFUSING: {existing} already records a run at {previous.get('ran_at')}.\n"
            "The test set is a one-shot check (ADR-003). A second run that\n"
            "replaces the first is indistinguishable from tuning on it.\n"
            "If you genuinely need to re-run - the code changed, the previous\n"
            "run is being superseded on the record - pass --rerun and say why\n"
            "in the story that does it.",
            file=sys.stderr,
        )
        return 2

    run(load_raw(args.csv), out, seed=args.seed, interpret=not args.no_interpret)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
