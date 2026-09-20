"""Engineered features — ratios, and the transformer where leakage actually bites.

The central test here is ``test_centroids_do_not_move_when_test_rows_are_transformed``.
Everything else supports it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from calhousing import config
from calhousing.preprocess.features import (
    RATIO_COLUMNS,
    ZERO_DENOMINATOR_FILL,
    ClusterSimilarity,
    RatioFeatures,
    safe_divide,
)

COUNTS = ["total_rooms", "total_bedrooms", "population", "households"]


@pytest.fixture
def counts(housing: pd.DataFrame) -> pd.DataFrame:
    return housing[COUNTS].fillna(housing[COUNTS].median())


@pytest.fixture
def geo(housing: pd.DataFrame) -> pd.DataFrame:
    return housing[["latitude", "longitude", "population"]]


# --- ratios ------------------------------------------------------------------


def test_ratios_are_appended_with_their_names(counts: pd.DataFrame) -> None:
    out = RatioFeatures().fit_transform(counts)
    assert list(out.columns) == COUNTS + RATIO_COLUMNS
    assert len(out) == len(counts)


def test_ratios_can_replace_the_totals(counts: pd.DataFrame) -> None:
    out = RatioFeatures(keep_original=False).fit_transform(counts)
    assert list(out.columns) == RATIO_COLUMNS


def test_get_feature_names_out_matches_the_output(counts: pd.DataFrame) -> None:
    for keep in (True, False):
        transformer = RatioFeatures(keep_original=keep).fit(counts)
        out = transformer.transform(counts)
        assert list(transformer.get_feature_names_out()) == list(out.columns)


def test_ratios_are_arithmetically_what_they_claim(counts: pd.DataFrame) -> None:
    out = RatioFeatures().fit_transform(counts)
    np.testing.assert_allclose(
        out["rooms_per_household"], counts["total_rooms"] / counts["households"]
    )
    np.testing.assert_allclose(
        out["bedrooms_per_room"], counts["total_bedrooms"] / counts["total_rooms"]
    )


def test_ratios_are_stateless_across_rows(counts: pd.DataFrame) -> None:
    """A row's ratio must not depend on any other row."""
    transformer = RatioFeatures().fit(counts.iloc[:100])
    together = transformer.transform(counts)
    alone = transformer.transform(counts.iloc[[7]])
    np.testing.assert_allclose(alone.to_numpy(), together.iloc[[7]].to_numpy())


def test_wrong_columns_are_refused_by_name(housing: pd.DataFrame) -> None:
    """A ColumnTransformer routing mistake must say so.

    Without this the failure is a ``KeyError: 'households'`` raised from inside
    a transform, several frames from the ``ColumnTransformer`` entry that
    actually chose the columns.
    """
    with pytest.raises(ValueError, match="RatioFeatures needs"):
        RatioFeatures().fit(housing[["latitude", "longitude"]])


# --- the zero-denominator guard ---------------------------------------------


def test_zero_denominator_yields_the_fill_not_inf() -> None:
    """**Synthetic rows on purpose.**

    No row in the real dataset has a zero denominator — ``households`` and
    ``total_rooms`` are never 0, measured on the raw file. This guard is not
    fixing an observed defect; it is refusing to depend on a property of one
    CSV. Saying the dataset forced it would be a nicer story and a false one.
    """
    planted = pd.DataFrame(
        {
            "total_rooms": [100.0, 0.0, 50.0],
            "total_bedrooms": [20.0, 5.0, 10.0],
            "population": [300.0, 100.0, 150.0],
            "households": [50.0, 25.0, 0.0],
        }
    )
    out = RatioFeatures().fit_transform(planted)
    values = out[RATIO_COLUMNS].to_numpy()
    assert np.isfinite(values).all(), "an inf or NaN escaped the ratio block"
    assert out.loc[2, "rooms_per_household"] == ZERO_DENOMINATOR_FILL
    assert out.loc[1, "bedrooms_per_room"] == ZERO_DENOMINATOR_FILL


def test_the_unguarded_division_really_does_produce_inf() -> None:
    """The other half: show what the guard prevents.

    If plain division ever stopped producing ``inf`` here, ``safe_divide``
    would be unnecessary and should be deleted.
    """
    numerator = pd.Series([1.0])
    denominator = pd.Series([0.0])
    assert np.isinf((numerator / denominator)).all()
    assert np.isfinite(safe_divide(numerator, denominator)).all()


def test_nan_denominator_is_also_handled() -> None:
    assert safe_divide(pd.Series([1.0]), pd.Series([np.nan])).iloc[0] == (
        ZERO_DENOMINATOR_FILL
    )


# --- ClusterSimilarity: the leakage test ------------------------------------


def test_centroids_do_not_move_when_test_rows_are_transformed(
    geo: pd.DataFrame,
) -> None:
    """**THE test for this module.**

    The centroids are learned parameters. Fit on train, then transform test —
    the centres must be byte-for-byte what they were before the test data was
    ever seen. A version of this written as a preprocessing *function* over the
    whole frame would fail here, and would score better while doing it.
    """
    train, test = geo.iloc[:400], geo.iloc[400:]
    transformer = ClusterSimilarity(n_clusters=5).fit(train)
    before = transformer.centroids_.copy()

    transformer.transform(test)
    transformer.transform(geo)

    np.testing.assert_array_equal(transformer.centroids_, before)


def test_fitting_on_everything_moves_the_centroids(geo: pd.DataFrame) -> None:
    """The vacuity check for the test above.

    If fitting on train+test produced the same centres as fitting on train, the
    leakage test could not detect anything and would be quietly worthless.
    """
    train = geo.iloc[:400]
    honest = ClusterSimilarity(n_clusters=5).fit(train).centroids_
    leaked = ClusterSimilarity(n_clusters=5).fit(geo).centroids_
    assert not np.allclose(np.sort(honest, axis=0), np.sort(leaked, axis=0))


def test_transform_of_a_row_alone_matches_in_company(geo: pd.DataFrame) -> None:
    train, test = geo.iloc[:400], geo.iloc[400:]
    transformer = ClusterSimilarity(n_clusters=6).fit(train)
    together = transformer.transform(test)
    for position in (0, 4, len(test) - 1):
        alone = transformer.transform(test.iloc[[position]])
        np.testing.assert_allclose(
            alone.to_numpy(), together.iloc[[position]].to_numpy()
        )


def test_output_shape_names_and_range(geo: pd.DataFrame) -> None:
    transformer = ClusterSimilarity(n_clusters=7).fit(geo)
    out = transformer.transform(geo)
    assert out.shape == (len(geo), 7)
    assert list(out.columns) == list(transformer.get_feature_names_out())
    assert all(name.startswith("geo_cluster_") for name in out.columns)
    values = out.to_numpy()
    assert values.min() >= 0.0 and values.max() <= 1.0


def test_similarity_peaks_at_its_own_centre(geo: pd.DataFrame) -> None:
    """The feature must mean what its name says.

    A row placed exactly on centre *k* must be more similar to *k* than to any
    other centre. Asserted as a ranking, not a value — the actual number
    depends on gamma.
    """
    transformer = ClusterSimilarity(n_clusters=5, gamma=1.0).fit(geo)
    for index, (lat, lon) in enumerate(transformer.centroids_):
        at_centre = pd.DataFrame(
            {"latitude": [lat], "longitude": [lon], "population": [1.0]}
        )
        row = transformer.transform(at_centre).to_numpy()[0]
        assert int(np.argmax(row)) == index
        np.testing.assert_allclose(row[index], 1.0)


def test_gamma_controls_locality(geo: pd.DataFrame) -> None:
    """Large gamma → narrow bumps. A small gamma makes every column approach 1
    and carry no information, which is a real way to search this into
    uselessness."""
    narrow = ClusterSimilarity(n_clusters=5, gamma=5.0).fit(geo).transform(geo)
    broad = ClusterSimilarity(n_clusters=5, gamma=0.01).fit(geo).transform(geo)
    assert narrow.to_numpy().mean() < broad.to_numpy().mean()


def test_population_weighting_changes_the_centres(geo: pd.DataFrame) -> None:
    """``weight_by`` must actually do something, or it is a decorative knob."""
    weighted = ClusterSimilarity(n_clusters=5, weight_by="population").fit(geo)
    unweighted = ClusterSimilarity(n_clusters=5, weight_by=None).fit(geo)
    assert not np.allclose(
        np.sort(weighted.centroids_, axis=0), np.sort(unweighted.centroids_, axis=0)
    )


def test_missing_weight_column_is_tolerated(geo: pd.DataFrame) -> None:
    """The ColumnTransformer may route only the coordinates here."""
    coords = geo[["latitude", "longitude"]]
    transformer = ClusterSimilarity(n_clusters=4, weight_by="population").fit(coords)
    assert transformer.transform(coords).shape == (len(coords), 4)


def test_is_reproducible(geo: pd.DataFrame) -> None:
    a = ClusterSimilarity(n_clusters=5).fit(geo).centroids_
    b = ClusterSimilarity(n_clusters=5).fit(geo).centroids_
    np.testing.assert_array_equal(a, b)


def test_missing_coordinates_are_refused_by_name(counts: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="ClusterSimilarity needs"):
        ClusterSimilarity().fit(counts)


def test_composes_into_a_pipeline_and_clones(geo: pd.DataFrame) -> None:
    """Both must survive ``clone``, or a search silently drops their settings."""
    pipeline = Pipeline([("geo", ClusterSimilarity(n_clusters=4))])
    clone(pipeline)
    out = pipeline.fit_transform(geo)
    assert out.shape[1] == 4

    ratios = clone(RatioFeatures(keep_original=False))
    assert ratios.keep_original is False


def test_real_geography_produces_usable_regions(real_housing: pd.DataFrame) -> None:
    """Skipped on CI. Locally: the centres must land inside California."""
    transformer = ClusterSimilarity(n_clusters=10).fit(real_housing)
    centres = transformer.centroids_
    assert ((centres[:, 0] > 32.0) & (centres[:, 0] < 43.0)).all()
    assert ((centres[:, 1] > -125.0) & (centres[:, 1] < -113.0)).all()
