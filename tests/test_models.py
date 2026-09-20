"""The boosting libraries, and the interop problems they bring with them.

Two external libraries now sit inside the pipeline. The tests here are about
the seam: a name XGBoost will accept, a version range that overlaps Kaggle's
image, and a search space that actually fits. The modelling itself is measured
in the PR, not asserted here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from calhousing import config
from calhousing.models import REGISTRY, cross_validate_pipeline, get_model
from calhousing.preprocess.assemble import (
    SanitiseFeatureNames,
    build_pipeline,
    feature_names,
)

BOOSTERS = ["xgb", "lgbm"]


@pytest.fixture
def xy(housing: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    shuffled = housing.sample(frac=1.0, random_state=3)
    return shuffled.drop(columns=[config.TARGET]), shuffled[config.TARGET]


# --- the feature-name seam ---------------------------------------------------


def test_the_dataset_really_does_contain_an_unsafe_name(xy) -> None:
    """The sanitiser is not speculative — the raw category forces it.

    ``ocean_proximity`` has the level ``"<1H OCEAN"``. Through the one-hot
    encoder that becomes a feature name containing ``<``, which XGBoost
    refuses::

        ValueError: feature_names must be string, and may not contain [, ] or <
    """
    X, _ = xy
    assert any("<" in str(v) for v in X["ocean_proximity"].unique())


@pytest.mark.parametrize("name", BOOSTERS)
def test_boosters_fit_inside_the_pipeline(name: str, xy) -> None:
    X, y = xy
    pipeline = build_pipeline(clone(get_model(name).factory()))
    pipeline.fit(X, y)
    assert np.isfinite(pipeline.predict(X)).all()


def test_sanitiser_removes_every_character_xgboost_rejects(xy) -> None:
    X, y = xy
    names = feature_names(build_pipeline(clone(get_model("xgb").factory())).fit(X, y))
    for forbidden in ("<", ">", "[", "]"):
        assert not any(forbidden in name for name in names), forbidden


def test_sanitised_names_are_still_readable(xy) -> None:
    """The point of not just handing XGBoost a bare ndarray.

    ``ocean_proximity_lt1H OCEAN`` is obvious to a reader; ``f17`` is not.
    """
    X, y = xy
    names = feature_names(build_pipeline(clone(get_model("xgb").factory())).fit(X, y))
    assert any("ocean_proximity" in name for name in names)
    assert any("geo_cluster" in name for name in names)
    assert all(name.count("__") >= 1 for name in names)


def test_sanitiser_refuses_to_merge_two_features() -> None:
    """A collision is worse than the character it was avoiding."""
    colliding = pd.DataFrame({"a<b": [1.0], "a>b": [2.0]})
    # `<` -> lt and `>` -> gt keep these apart...
    SanitiseFeatureNames().fit(colliding)
    # ...but two names that differ only by a rewritten character must refuse.
    with pytest.raises(ValueError, match="collisions"):
        SanitiseFeatureNames().fit(pd.DataFrame({"a<b": [1.0], "altb": [2.0]}))


def test_sanitiser_is_a_no_op_on_clean_names() -> None:
    clean = pd.DataFrame({"alpha": [1.0], "beta": [2.0]})
    step = SanitiseFeatureNames().fit(clean)
    pd.testing.assert_frame_equal(step.transform(clean), clean)


# --- registry hygiene --------------------------------------------------------


@pytest.mark.parametrize("name", BOOSTERS)
def test_booster_search_spaces_resolve(name: str) -> None:
    """A key naming nothing raises only once a search starts fitting."""
    from calhousing.train import search_space_for

    pipeline = build_pipeline(clone(get_model(name).factory()))
    available = set(pipeline.get_params(deep=True))
    unknown = [k for k in search_space_for(name) if k not in available]
    assert not unknown, unknown


@pytest.mark.parametrize("name", BOOSTERS)
def test_boosters_are_reproducible(name: str, xy) -> None:
    X, y = xy
    first = cross_validate_pipeline(name, X, y, n_splits=2)["rmse"]
    second = cross_validate_pipeline(name, X, y, n_splits=2)["rmse"]
    assert first == second


@pytest.mark.parametrize("name", BOOSTERS)
def test_boosters_run_single_threaded(name: str) -> None:
    """The search parallelises across draws. A thread pool inside a process
    pool oversubscribes the machine and is reliably slower than either."""
    estimator = get_model(name).factory()
    assert estimator.get_params()["n_jobs"] == 1


def test_lightgbm_does_not_print_a_banner_per_fit() -> None:
    """25 draws x 5 folds is 125 banners burying the one line anybody reads."""
    assert get_model("lgbm").factory().get_params()["verbosity"] == -1


def test_the_widened_ranges_are_not_pinned_at_a_boundary() -> None:
    """M3-S2's winners both chose the TOP of their ranges.

    A boundary winner means the search was cut off, not converged - the range
    had not been tested, only truncated. The ranges were widened in M3-S3, and
    this test records the rule rather than the symptom.
    """
    from calhousing.models import PREPROCESS_SEARCH_SPACE

    clusters = PREPROCESS_SEARCH_SPACE["preprocess__geo__n_clusters"]
    bins = PREPROCESS_SEARCH_SPACE["preprocess__binned__bin__n_bins"]
    assert max(clusters) > 20, "n_clusters=20 won in M3-S2; the range must extend past it"
    assert max(bins) > 8, "n_bins=8 won in M3-S2; the range must extend past it"


def test_every_model_in_the_registry_is_distinct() -> None:
    """Five arms that are four copies of one inductive bias make a stack that
    cannot disagree with itself."""
    classes = {name: spec.factory().__class__.__name__ for name, spec in REGISTRY.items()}
    assert len(set(classes.values())) == len(classes), classes


def test_versions_overlap_the_declared_range() -> None:
    """The kernel's fallback install resolves no dependencies, so a version
    this project needs but Kaggle lacks is a published traceback."""
    import lightgbm
    import xgboost

    assert int(xgboost.__version__.split(".")[0]) >= 2
    assert int(lightgbm.__version__.split(".")[0]) >= 4
