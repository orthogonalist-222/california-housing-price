"""Stacking, interpretation, and the guard on the one-shot evaluation.

The most important test in this file is
``test_a_second_run_is_refused``. Everything the protocol says about one-shot
checks is worth exactly as much as that refusal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from calhousing import config
from calhousing.interpret import (
    importance_table,
    plot_importance,
    plot_partial_dependence,
)
from calhousing.models import (
    REGISTRY,
    STACK_MEMBERS,
    STACK_PREPROCESS,
    TUNED,
    tuned_estimator,
    tuned_pipeline,
)


@pytest.fixture
def xy(housing: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    shuffled = housing.sample(frac=1.0, random_state=4)
    return shuffled.drop(columns=[config.TARGET]), shuffled[config.TARGET]


# --- the recorded tuning -----------------------------------------------------


def test_tuned_params_are_recorded_in_code_not_read_from_artifacts() -> None:
    """``artifacts/`` is gitignored and may be empty on any checkout.

    A stack that loaded ``artifacts/search/*.joblib`` would be a recipe that
    cannot rebuild itself. The winning configurations are transcribed into
    ``TUNED`` instead.
    """
    for name, entry in TUNED.items():
        assert name in REGISTRY
        assert "model" in entry and "preprocess" in entry


@pytest.mark.parametrize("name", sorted(TUNED))
def test_every_tuned_configuration_builds_and_fits(name: str, xy) -> None:
    """A transcription error in ``TUNED`` must fail here, not in the one-shot
    run. A typo discovered during the final evaluation costs the test set."""
    X, y = xy
    pipeline = tuned_pipeline(name)
    pipeline.fit(X, y)
    assert np.isfinite(pipeline.predict(X)).all()


@pytest.mark.parametrize("name", sorted(TUNED))
def test_tuned_params_are_actually_applied(name: str) -> None:
    """Recorded-but-not-applied is the quiet way this goes wrong."""
    estimator = tuned_estimator(name)
    for key, value in TUNED[name]["model"].items():
        assert estimator.get_params()[key] == value


# --- the stack ---------------------------------------------------------------


def test_stack_members_are_genuinely_different_learners() -> None:
    """Four copies of one inductive bias give a meta-learner nothing to
    arbitrate."""
    classes = {n: REGISTRY[n].factory().__class__.__name__ for n in STACK_MEMBERS}
    assert len(set(classes.values())) == len(classes), classes


def test_random_forest_is_deliberately_excluded() -> None:
    """289 MB and 115s per fit for a score indistinguishable from LightGBM's
    6.8 MB and 5.1s (RT-018) - and inside a stack that cost is multiplied by
    the internal CV. Pinned so the exclusion is a decision, not an oversight."""
    assert "rf" not in STACK_MEMBERS


def test_the_stack_shares_one_preprocessing_and_says_which() -> None:
    """A real simplification, recorded rather than hidden: the base learners
    are NOT exactly the pipelines that were tuned."""
    assert STACK_PREPROCESS == TUNED["lgbm"]["preprocess"]


def test_stack_fits_and_predicts(xy) -> None:
    X, y = xy
    pipeline = tuned_pipeline("stack")
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()


def test_stack_preprocesses_once_not_once_per_member(xy) -> None:
    """The base learners are bare estimators inside one pipeline.

    Nesting a full pipeline per member would mean six preprocessing fits per
    stack fit. This asserts the shape that avoids it.
    """
    from sklearn.pipeline import Pipeline

    stack = REGISTRY["stack"].factory()
    for _, member in stack.estimators:
        assert not isinstance(member, Pipeline), (
            "a stack member is itself a Pipeline - the preprocessing would run "
            "once per member per fold"
        )


def test_stack_meta_learner_can_downweight_a_member() -> None:
    """A fixed average cannot say a base learner adds nothing. RidgeCV can."""
    from sklearn.linear_model import RidgeCV

    assert isinstance(REGISTRY["stack"].factory().final_estimator, RidgeCV)


# --- interpretation ----------------------------------------------------------


def test_importance_is_over_raw_columns(xy) -> None:
    """``total_rooms`` feeds three branches. Permuting one derived feature
    would leave the other two intact and report the column as unimportant."""
    X, y = xy
    fitted = tuned_pipeline("ridge").fit(X, y)
    table = importance_table(fitted, X.iloc[:200], y.iloc[:200], n_repeats=2)
    assert set(table["column"]) == set(X.columns)


def test_importance_ranks_a_signal_column_above_a_noise_column(xy) -> None:
    """The measure has to be able to tell them apart, or the plot is decor.

    The synthetic target is built from ``median_income``; ``housing_median_age``
    is not in the formula at all.
    """
    X, y = xy
    fitted = tuned_pipeline("lgbm").fit(X, y)
    table = importance_table(fitted, X.iloc[:300], y.iloc[:300], n_repeats=3)
    ranks = {row.column: index for index, row in enumerate(table.itertuples())}
    assert ranks["median_income"] < ranks["housing_median_age"]


def test_importance_figures_are_written(xy, tmp_path: Path) -> None:
    X, y = xy
    fitted = tuned_pipeline("ridge").fit(X, y)
    table = importance_table(fitted, X.iloc[:200], y.iloc[:200], n_repeats=2)
    first = plot_importance(table, tmp_path)
    second = plot_partial_dependence(
        fitted, X.iloc[:200], ["median_income", "housing_median_age"], tmp_path
    )
    for path in (first, second):
        assert path.is_file() and path.stat().st_size > 0


def test_importance_figures_are_deterministic(xy, tmp_path: Path) -> None:
    import hashlib

    X, y = xy
    fitted = tuned_pipeline("ridge").fit(X, y)
    table = importance_table(fitted, X.iloc[:200], y.iloc[:200], n_repeats=2)
    a = plot_importance(table, tmp_path / "a")
    b = plot_importance(table, tmp_path / "b")
    assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()


# --- the one-shot guard ------------------------------------------------------


def test_the_arms_are_exactly_what_adr_003_named() -> None:
    """ADR-003 fixed five arms before any of them existed. Adding a sixth
    afterwards and testing it is a second test run wearing a disguise."""
    from final_evaluation import ARMS

    assert ARMS == ("dummy", "ridge", "rf", "lgbm", "stack")


def test_a_second_run_is_refused(housing: pd.DataFrame, tmp_path: Path) -> None:
    """**The guard the whole protocol rests on.**

    The test set is a one-shot check. A second run whose output replaces the
    first is indistinguishable from tuning on it, so the script refuses rather
    than trusting anybody to remember.
    """
    from final_evaluation import RECORD_NAME, main

    csv = tmp_path / "housing.csv"
    housing.to_csv(csv, index=False)
    out = tmp_path / "final"

    assert main(["--csv", str(csv), "--out", str(out), "--no-interpret"]) == 0
    record = json.loads((out / RECORD_NAME).read_text(encoding="utf-8"))
    first_ran_at = record["ran_at"]

    # Second run: refused, and the original record is untouched.
    assert main(["--csv", str(csv), "--out", str(out), "--no-interpret"]) == 2
    again = json.loads((out / RECORD_NAME).read_text(encoding="utf-8"))
    assert again["ran_at"] == first_ran_at


def test_the_override_exists_and_works(housing: pd.DataFrame, tmp_path: Path) -> None:
    """A guard with no escape hatch gets worked around by deleting the file,
    which leaves no trace. ``--rerun`` is explicit and greppable."""
    from final_evaluation import main

    csv = tmp_path / "housing.csv"
    housing.to_csv(csv, index=False)
    out = tmp_path / "final"
    assert main(["--csv", str(csv), "--out", str(out), "--no-interpret"]) == 0
    assert main(["--csv", str(csv), "--out", str(out), "--no-interpret", "--rerun"]) == 0


def test_the_record_is_readable_without_unpickling(
    housing: pd.DataFrame, tmp_path: Path
) -> None:
    """The evidence a reviewer walks must not require loading a model."""
    from final_evaluation import RECORD_NAME, main

    csv = tmp_path / "housing.csv"
    housing.to_csv(csv, index=False)
    out = tmp_path / "final"
    main(["--csv", str(csv), "--out", str(out), "--no-interpret"])

    record = json.loads((out / RECORD_NAME).read_text(encoding="utf-8"))
    for key in ("ran_at", "seed", "n_train", "n_test", "arms", "segments", "tuned_params"):
        assert key in record
    for arm in record["arms"]:
        # Every arm records the FULL segment table, per ADR-001's condition.
        assert set(record["segments"][arm]) == {
            "all",
            "uncensored",
            "censored",
            "inland",
            "coastal",
        }
