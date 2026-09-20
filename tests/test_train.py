"""The search. Small and synthetic — the real runs are recorded in the PR.

What is asserted here is that the search is **joint**: that it can and does
reach preprocessing parameters as well as model ones. That is the claim this
whole project makes about pipelines, and it is the one thing a test can check
in two seconds that would otherwise take a twenty-minute run to discover.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from calhousing import config
from calhousing.models import PREPROCESS_SEARCH_SPACE, REGISTRY
from calhousing.train import leaderboard, run_search, search_space_for


@pytest.fixture
def xy(housing: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Shuffled, so the index is NOT 0..n-1. F-006 hid behind a RangeIndex.
    shuffled = housing.sample(frac=1.0, random_state=2)
    return shuffled.drop(columns=[config.TARGET]), shuffled[config.TARGET]


def test_the_space_mixes_model_and_preprocessing_parameters() -> None:
    """**The claim of the project, as an assertion.**

    Spark's ``ParamGridBuilder`` searches a ``Pipeline``'s stages. So does
    this. A space containing only ``model__*`` keys would mean the
    preprocessing had been fixed by taste and only the estimator tuned — which
    is what a hand-rolled ``fit_transform`` chain forces you to do.
    """
    space = search_space_for("ridge")
    assert any(key.startswith("model__") for key in space)
    assert any(key.startswith("preprocess__") for key in space)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registered_model_has_a_usable_space(name: str) -> None:
    space = search_space_for(name)
    assert space, f"{name} has an empty search space"
    for key, values in space.items():
        assert key.startswith(("model__", "preprocess__")), key
        assert len(values) > 0, key


def test_search_runs_and_refits(xy) -> None:
    X, y = xy
    search = run_search("ridge", X, y, n_iter=3, n_splits=2, n_jobs=1, verbose=False)
    assert hasattr(search, "best_estimator_")
    assert search.best_score_ < 0  # neg RMSE
    assert len(search.predict(X)) == len(X)


def test_search_actually_varies_preprocessing(xy) -> None:
    """A search that only ever drew the default preprocessing would satisfy
    every other test here while proving nothing."""
    X, y = xy
    search = run_search("ridge", X, y, n_iter=6, n_splits=2, n_jobs=1, verbose=False)
    drawn = {
        tuple(sorted((k, str(v)) for k, v in params.items() if k.startswith("preprocess__")))
        for params in search.cv_results_["params"]
    }
    assert len(drawn) > 1, "every draw used the same preprocessing"


def test_a_failed_draw_is_an_error_not_a_low_score(xy) -> None:
    """``error_score`` defaults to ``np.nan``, which turns a broken
    configuration into a merely unlucky one — and the search reports a winner
    while silently discarding half its budget. F-006 was exactly that shape."""
    X, y = xy
    search = run_search("ridge", X, y, n_iter=2, n_splits=2, n_jobs=1, verbose=False)
    assert search.error_score == "raise"


def test_leaderboard_shows_train_beside_test(xy) -> None:
    """A draw whose train RMSE is half its test RMSE won by overfitting, and a
    leaderboard with only the test column hides that."""
    X, y = xy
    search = run_search("ridge", X, y, n_iter=4, n_splits=2, n_jobs=1, verbose=False)
    board = leaderboard(search, top=3)
    assert list(board.columns) == [
        "cv_rmse",
        "cv_rmse_std",
        "train_rmse",
        "mean_fit_time",
    ]
    assert len(board) == 3
    assert board["cv_rmse"].is_monotonic_increasing


def test_search_is_reproducible(xy) -> None:
    X, y = xy
    first = run_search("ridge", X, y, n_iter=4, n_splits=2, n_jobs=1, verbose=False)
    second = run_search("ridge", X, y, n_iter=4, n_splits=2, n_jobs=1, verbose=False)
    assert first.best_score_ == second.best_score_
    assert first.best_params_.keys() == second.best_params_.keys()


def test_cli_writes_a_model_and_a_readable_record(housing: pd.DataFrame, tmp_path) -> None:
    """The recipe must rebuild the artefact, and the record must be readable
    without unpickling anything."""
    from calhousing.train import main

    csv = tmp_path / "housing.csv"
    housing.to_csv(csv, index=False)
    out = tmp_path / "search"

    assert main(
        ["--model", "ridge", "--n-iter", "2", "--n-splits", "2",
         "--n-jobs", "1", "--csv", str(csv), "--out", str(out)]
    ) == 0

    assert (out / "ridge-best.joblib").is_file()
    record = json.loads((out / "ridge-best.json").read_text(encoding="utf-8"))
    assert record["model"] == "ridge"
    assert record["cv_rmse"] > 0
    assert record["n_iter"] == 2
    assert record["best_params"], "the winning configuration must be recorded"


def test_the_cli_never_touches_the_test_split(housing: pd.DataFrame, tmp_path) -> None:
    """ADR-003. The test split is scored exactly once, in M3-S4.

    Asserted structurally: the search is fitted on fewer rows than the file
    holds, which is only true because the test split was set aside.
    """
    from calhousing.train import main
    import calhousing.train as train_module

    captured: dict[str, int] = {}
    original = train_module.run_search

    def spy(name, X, y, **kwargs):
        captured["n"] = len(X)
        return original(name, X, y, **kwargs)

    train_module.run_search = spy
    try:
        csv = tmp_path / "housing.csv"
        housing.to_csv(csv, index=False)
        main(["--model", "ridge", "--n-iter", "2", "--n-splits", "2",
              "--n-jobs", "1", "--csv", str(csv), "--out", str(tmp_path / "s")])
    finally:
        train_module.run_search = original

    assert captured["n"] < len(housing)
    assert captured["n"] == pytest.approx(0.8 * len(housing), abs=2)
