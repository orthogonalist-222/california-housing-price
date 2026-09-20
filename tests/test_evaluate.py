"""Metrics and the CV harness.

The segment table is the thing being protected here. ADR-001 kept the censored
rows on the condition that every evaluation says where the error lives, so a
change that quietly drops a segment has to fail a test rather than a review.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calhousing import config
from calhousing.evaluate import (
    METRIC_NAMES,
    censored_mask,
    compare,
    inland_mask,
    regression_metrics,
    segment_table,
)
from calhousing.models import (
    PREPROCESS_SEARCH_SPACE,
    REGISTRY,
    build_cv,
    cross_validate_pipeline,
    get_model,
)


# --- metrics -----------------------------------------------------------------


def test_perfect_predictions_score_perfectly() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = regression_metrics(y, y)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["n"] == 4


def test_rmse_punishes_one_large_error_more_than_mae() -> None:
    """The reason RMSE is primary, asserted rather than asserted-in-prose."""
    y = np.zeros(100)
    spread = np.full(100, 10.0)
    concentrated = np.zeros(100)
    concentrated[0] = 1000.0

    spread_m = regression_metrics(y, spread)
    conc_m = regression_metrics(y, concentrated)
    assert conc_m["mae"] == pytest.approx(spread_m["mae"])
    assert conc_m["rmse"] > spread_m["rmse"]


def test_r2_is_nan_for_a_single_row_not_zero() -> None:
    """sklearn returns 0.0 with a warning, which reads as a real score."""
    metrics = regression_metrics([5.0], [5.0])
    assert np.isnan(metrics["r2"])
    assert metrics["n"] == 1


def test_empty_input_does_not_raise() -> None:
    """A segment can be empty — an island-free test fold, for instance."""
    metrics = regression_metrics([], [])
    assert metrics["n"] == 0
    assert np.isnan(metrics["rmse"])


# --- masks -------------------------------------------------------------------


def test_censored_mask_catches_both_ends() -> None:
    """ADR-001: the target is censored at the cap **and** the floor."""
    y = np.array(
        [config.TARGET_FLOOR, 100_000.0, config.TARGET_CAP, 250_000.0]
    )
    np.testing.assert_array_equal(censored_mask(y), [True, False, True, False])


def test_inland_mask_reads_the_category(housing: pd.DataFrame) -> None:
    mask = inland_mask(housing)
    assert mask.sum() == (housing["ocean_proximity"] == "INLAND").sum()


# --- the segment table -------------------------------------------------------


def test_segment_table_reports_every_segment(housing: pd.DataFrame) -> None:
    """The obligation ADR-001 took on, pinned as a test."""
    y = housing[config.TARGET]
    predictions = np.full(len(y), float(y.median()))
    table = segment_table(housing, y, predictions)

    assert list(table.index) == ["all", "uncensored", "censored", "inland", "coastal"]
    assert list(table.columns) == list(METRIC_NAMES)
    assert table.loc["all", "n"] == len(y)


def test_segments_partition_the_rows(housing: pd.DataFrame) -> None:
    """Each pair must add back up to the whole, or a segment is being dropped."""
    y = housing[config.TARGET]
    table = segment_table(housing, y, np.full(len(y), float(y.mean())))
    total = table.loc["all", "n"]
    assert table.loc["uncensored", "n"] + table.loc["censored", "n"] == total
    assert table.loc["inland", "n"] + table.loc["coastal", "n"] == total


def test_the_table_can_separate_a_model_that_is_bad_in_one_segment(
    housing: pd.DataFrame,
) -> None:
    """The table has to be capable of showing what it exists to show.

    A prediction that is perfect on the coast and wrong inland must produce a
    visibly worse inland RMSE — otherwise the segment split is decoration.
    """
    y = housing[config.TARGET].to_numpy(dtype=float)
    predictions = y.copy()
    predictions[inland_mask(housing)] += 50_000.0

    table = segment_table(housing, y, predictions)
    assert table.loc["coastal", "rmse"] == pytest.approx(0.0)
    assert table.loc["inland", "rmse"] > 1_000.0


def test_compare_sorts_by_rmse() -> None:
    table = compare(
        {
            "worse": {"rmse": 100.0, "mae": 90.0},
            "better": {"rmse": 10.0, "mae": 9.0},
        }
    )
    assert list(table.index) == ["better", "worse"]


# --- the harness -------------------------------------------------------------


def test_build_cv_is_the_same_object_every_time() -> None:
    """Two arms scored under different folds are not comparable."""
    first, second = build_cv(), build_cv()
    # A splitter is not an estimator, so there is no `get_params` to compare -
    # its attributes are the interface.
    assert (first.n_splits, first.shuffle, first.random_state) == (
        second.n_splits,
        second.shuffle,
        second.random_state,
    )
    assert first.shuffle is True
    assert first.random_state == config.RANDOM_SEED


def test_unknown_model_is_refused_with_the_known_names() -> None:
    with pytest.raises(KeyError, match="Known models"):
        get_model("catboost")


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registered_model_cross_validates(name: str, housing: pd.DataFrame) -> None:
    X, y = housing.drop(columns=[config.TARGET]), housing[config.TARGET]
    result = cross_validate_pipeline(name, X, y, n_splits=3)
    for metric in ("rmse", "mae", "r2"):
        assert np.isfinite(result[metric])
        assert np.isfinite(result[f"{metric}_std"])
        assert np.isfinite(result[f"{metric}_train"])


def test_errors_are_reported_as_positive_numbers(housing: pd.DataFrame) -> None:
    """sklearn maximises, so its error scorers are negated. Every number a
    human reads here must already be flipped back."""
    X, y = housing.drop(columns=[config.TARGET]), housing[config.TARGET]
    result = cross_validate_pipeline("ridge", X, y, n_splits=3)
    assert result["rmse"] > 0
    assert result["mae"] > 0
    assert result["mae"] < result["rmse"], "MAE cannot exceed RMSE"


def test_a_real_model_beats_the_dummy(housing: pd.DataFrame) -> None:
    """The floor exists to be cleared. If ridge ever failed to clear it, the
    pipeline is broken in a way no other test would name."""
    X, y = housing.drop(columns=[config.TARGET]), housing[config.TARGET]
    dummy = cross_validate_pipeline("dummy", X, y, n_splits=3)["rmse"]
    ridge = cross_validate_pipeline("ridge", X, y, n_splits=3)["rmse"]
    assert ridge < dummy


def test_harness_is_reproducible(housing: pd.DataFrame) -> None:
    X, y = housing.drop(columns=[config.TARGET]), housing[config.TARGET]
    first = cross_validate_pipeline("ridge", X, y, n_splits=3)
    second = cross_validate_pipeline("ridge", X, y, n_splits=3)
    assert first["rmse"] == second["rmse"]


def test_search_space_paths_all_resolve() -> None:
    """A search space key that names no real step targets nothing, silently.

    This is the failure mode that wastes a whole search: ``RandomizedSearchCV``
    raises on an unknown path, but only once it starts fitting — which on a
    real budget is minutes in and several arms deep.
    """
    from sklearn.linear_model import Ridge

    from calhousing.preprocess.assemble import build_pipeline

    pipeline = build_pipeline(Ridge())
    available = set(pipeline.get_params(deep=True))
    space = {**PREPROCESS_SEARCH_SPACE, **REGISTRY["ridge"].search_space}
    unknown = [key for key in space if key not in available]
    assert not unknown, f"search-space keys that name nothing: {unknown}"


def test_every_search_space_value_is_settable(housing: pd.DataFrame) -> None:
    """The paths resolving is not enough — the values have to be accepted."""
    from sklearn.base import clone
    from sklearn.linear_model import Ridge

    from calhousing.preprocess.assemble import build_pipeline

    X, y = housing.drop(columns=[config.TARGET]), housing[config.TARGET]
    for key, values in PREPROCESS_SEARCH_SPACE.items():
        pipeline = build_pipeline(Ridge())
        pipeline.set_params(**{key: clone(values[0], safe=False)})
        pipeline.fit(X.iloc[:200], y.iloc[:200])


def test_registry_entries_explain_themselves() -> None:
    """A model in the comparison with no stated reason is a model nobody can
    defend leaving in or taking out."""
    for name, spec in REGISTRY.items():
        assert spec.name == name
        assert len(spec.why) > 40, f"{name} has no real justification"
