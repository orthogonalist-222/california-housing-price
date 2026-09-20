"""The assembled pipeline, and the leak defence that actually earns its keep.

`tools/measure_leakage.py` measured what leakage costs on this dataset, and the
answer reshaped these tests. Fitting the preprocessing outside the fold is worth
**-0.135%** of RMSE here - 0.06x the fold-to-fold spread, with a sign that flips
as the sample changes. It is noise.

What is *not* noise is a leaked **column**: forcing a target-derived feature past
the assembler takes RMSE from 63 811 to 5 121. And the thing that prevents that
is `remainder="drop"` - the assembler is a whitelist.

So the tests below assert the **mechanism**, never a score. Scores belong in the
measurement script, where they can be rerun and argued with.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from calhousing import config
from calhousing.preprocess.assemble import (
    BRANCHES,
    build_pipeline,
    build_preprocessor,
    feature_names,
)


@pytest.fixture
def xy(housing: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return housing.drop(columns=[config.TARGET]), housing[config.TARGET]


# --- assembly ----------------------------------------------------------------


def test_pipeline_is_one_estimator_that_fits_and_predicts(xy) -> None:
    X, y = xy
    pipeline = build_pipeline(Ridge())
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()


def test_every_declared_branch_is_present(xy) -> None:
    X, _ = xy
    preprocessor = build_preprocessor().fit(X)
    assert tuple(name for name, _, _ in preprocessor.transformers_ if name != "remainder") == BRANCHES


def test_feature_names_are_readable_unique_and_branch_prefixed(xy) -> None:
    """A model whose importances read ``x17`` cannot be interrogated."""
    X, y = xy
    pipeline = build_pipeline(Ridge()).fit(X, y)
    names = feature_names(pipeline)
    assert len(names) == len(set(names)), "duplicate feature names"
    assert all(name.split("__")[0] in BRANCHES for name in names), names[:5]
    assert any("geo_cluster" in name for name in names)
    assert any("rooms_per_household" in name for name in names)


def test_names_match_the_transformed_width(xy) -> None:
    X, _ = xy
    preprocessor = build_preprocessor().fit(X)
    assert preprocessor.transform(X).shape[1] == len(feature_names(preprocessor))


def test_a_column_may_feed_more_than_one_branch(xy) -> None:
    """A ColumnTransformer is a fan-out, not a partition.

    ``total_rooms`` is a denominator in ``ratios`` and a de-skewed magnitude in
    ``heavy``. That is the property that makes it the right analogue for
    ``VectorAssembler``.
    """
    X, _ = xy
    names = feature_names(build_preprocessor().fit(X))
    assert any(name.startswith("heavy__total_rooms") for name in names)
    assert any(name.startswith("ratios__rooms_per_household") for name in names)


# --- the leak defence that actually matters ---------------------------------


def test_the_assembler_is_a_whitelist(xy) -> None:
    """**The defence with a large measured payoff.**

    An undeclared column - however tempting, however target-shaped - never
    reaches the model. Measured: forcing a target-derived column past the
    assembler takes RMSE from 63 811 to 5 121; routed *through* the assembler it
    changes nothing, because it is dropped.
    """
    X, y = xy
    poisoned = X.assign(appraisal=y * 0.98)

    clean_names = feature_names(build_preprocessor().fit(X))
    poisoned_names = feature_names(build_preprocessor().fit(poisoned))

    assert clean_names == poisoned_names
    assert not any("appraisal" in name for name in poisoned_names)


def test_the_whitelist_survives_a_perfect_leak(xy) -> None:
    """A column that IS the target, verbatim, still does not reach the model."""
    X, y = xy
    pipeline = build_pipeline(DecisionTreeRegressor(random_state=0))
    honest = pipeline.fit(X, y).predict(X)
    with_leak = clone(pipeline).fit(X.assign(cheat=y), y).predict(X.assign(cheat=y))
    np.testing.assert_allclose(honest, with_leak)


def test_remainder_is_drop_not_passthrough(xy) -> None:
    """Pinned explicitly, because the failure is silent.

    ``remainder="passthrough"`` is how a leaked column reaches a model without
    anybody choosing to put it there - no error, no warning, just a better
    score.
    """
    X, _ = xy
    assert build_preprocessor().fit(X).remainder == "drop"


# --- refitting inside the fold ----------------------------------------------


def test_every_stateful_step_is_refit_from_its_training_rows(xy) -> None:
    """The mechanism the whole project is about, asserted structurally.

    Fitting on two different subsets must produce different learned parameters.
    If it did not, ``cross_val_score`` over the pipeline would be doing nothing
    that ``cross_val_score`` over a pre-transformed matrix does not.
    """
    X, y = xy
    first = build_pipeline(Ridge()).fit(X.iloc[:300], y.iloc[:300])
    second = build_pipeline(Ridge()).fit(X.iloc[300:], y.iloc[300:])

    def geo_centroids(pipeline: Pipeline) -> np.ndarray:
        return pipeline.named_steps["preprocess"].named_transformers_["geo"].centroids_

    def ratio_medians(pipeline: Pipeline) -> np.ndarray:
        branch = pipeline.named_steps["preprocess"].named_transformers_["ratios"]
        return branch.named_steps["impute"].statistics_

    assert not np.allclose(
        np.sort(geo_centroids(first), axis=0), np.sort(geo_centroids(second), axis=0)
    )
    assert not np.allclose(ratio_medians(first), ratio_medians(second))


def test_transforming_a_row_alone_matches_transforming_it_in_company(xy) -> None:
    """End-to-end version of the per-block check from M2-S1 and M2-S3."""
    X, y = xy
    train, test = X.iloc[:400], X.iloc[400:]
    preprocessor = build_preprocessor().fit(train)
    together = preprocessor.transform(test)
    for position in (0, 5, len(test) - 1):
        alone = preprocessor.transform(test.iloc[[position]])
        np.testing.assert_allclose(
            alone.to_numpy(), together.iloc[[position]].to_numpy(), atol=1e-10
        )


# --- robustness regressions --------------------------------------------------


def test_an_institutional_block_group_does_not_blow_up_predictions(xy) -> None:
    """Regression test for the defect found while assembling this story.

    The real dataset contains block groups that are institutions, not
    neighbourhoods - one has 6 households and 7 460 people, so
    ``population_per_household`` is 1 243 against a median of 2.8. Unclipped and
    standard-scaled, that row is a z-score in the hundreds, and on a 5 000-row
    subsample one fold came back at **RMSE 1 805 055** beside neighbours around
    65 000.

    Asserted as a property, not a number: predictions must stay inside a sane
    multiple of the observed target range.
    """
    X, y = xy
    planted = X.copy()
    planted.iloc[0, planted.columns.get_loc("households")] = 6.0
    planted.iloc[0, planted.columns.get_loc("population")] = 7460.0
    planted.iloc[0, planted.columns.get_loc("total_rooms")] = 19.0

    pipeline = build_pipeline(Ridge()).fit(planted, y)
    predictions = pipeline.predict(planted)
    span = y.max() - y.min()
    assert predictions.min() > y.min() - 3 * span
    assert predictions.max() < y.max() + 3 * span


def test_clipping_can_be_switched_off(xy) -> None:
    """The knob has to exist, or the clip is dogma rather than a decision."""
    X, y = xy
    unclipped = build_pipeline(Ridge(), clip=0.0).fit(X, y)
    assert np.isfinite(unclipped.predict(X)).all()


def test_an_unseen_category_survives_the_whole_pipeline(xy) -> None:
    """The M2-S2 red-team, end to end."""
    X, y = xy
    island_free = X[X["ocean_proximity"] != "ISLAND"]
    islands = X[X["ocean_proximity"] == "ISLAND"]
    assert len(islands) > 0

    pipeline = build_pipeline(Ridge()).fit(island_free, y.loc[island_free.index])
    predictions = pipeline.predict(islands)
    assert np.isfinite(predictions).all()


def test_nulls_never_reach_the_model(xy) -> None:
    X, _ = xy
    assert X["total_bedrooms"].isna().any()
    out = build_preprocessor().fit_transform(X)
    assert not out.isna().any().any()


# --- searchability -----------------------------------------------------------


def test_preprocessing_parameters_are_addressable_by_path(xy) -> None:
    """M3-S2 sets these by name. A renamed step empties a search space
    silently, so the paths are pinned here."""
    X, y = xy
    pipeline = build_pipeline(Ridge())
    pipeline.set_params(
        preprocess__heavy__deskew="passthrough",
        preprocess__geo__n_clusters=4,
        model__alpha=10.0,
    )
    pipeline.fit(X, y)
    assert pipeline.named_steps["preprocess"].named_transformers_["geo"].n_clusters == 4
    assert sum(
        1 for name in feature_names(pipeline) if name.startswith("geo__")
    ) == 4


def test_pipeline_clones_cleanly(xy) -> None:
    pipeline = build_pipeline(Ridge(), n_clusters=3, encoder="ordinal")
    copy = clone(pipeline)
    assert copy.get_params()["preprocess__geo__n_clusters"] == 3


@pytest.mark.parametrize("encoder", ["onehot", "ordinal"])
@pytest.mark.parametrize("deskew", ["none", "log"])
def test_configurations_the_search_will_visit_all_build_and_fit(
    encoder: str, deskew: str, xy
) -> None:
    """Every combination M3-S2 can draw must actually run.

    A search that dies two hours in on a combination nobody tried is the
    expensive way to discover this.
    """
    X, y = xy
    pipeline = build_pipeline(Ridge(), encoder=encoder, deskew=deskew)
    pipeline.fit(X, y)
    assert np.isfinite(pipeline.predict(X)).all()


def test_real_data_assembles_end_to_end(real_housing: pd.DataFrame) -> None:
    """Skipped on CI. Locally: the whole thing, on the actual file."""
    X = real_housing.drop(columns=[config.TARGET])
    y = real_housing[config.TARGET]
    pipeline = build_pipeline(Ridge()).fit(X, y)
    names = feature_names(pipeline)
    assert len(names) == len(set(names))
    assert np.isfinite(pipeline.predict(X)).all()
