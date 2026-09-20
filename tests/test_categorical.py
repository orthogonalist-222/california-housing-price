"""Categorical block — and the unseen-category red-team this dataset demands.

``ocean_proximity`` has a level with five rows. Every test below that fits on a
frame with the island removed is simulating something that genuinely happens in
a cross-validation fold, not an exotic hypothetical.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from calhousing import config
from calhousing.preprocess.categorical import (
    ENCODERS,
    UNKNOWN_VALUE,
    UnsafeEncoderError,
    build_binned_block,
    build_categorical_block,
    make_onehot_encoder,
    make_ordinal_encoder,
)

CAT = config.CATEGORICAL_COLUMNS
ISLAND = "ISLAND"


@pytest.fixture
def cat_frame(housing: pd.DataFrame) -> pd.DataFrame:
    return housing[CAT]


@pytest.fixture
def island_free(cat_frame: pd.DataFrame) -> pd.DataFrame:
    """A training fold that happened not to contain the rare level."""
    out = cat_frame[cat_frame["ocean_proximity"] != ISLAND]
    assert ISLAND not in set(out["ocean_proximity"])
    return out


@pytest.fixture
def island_only() -> pd.DataFrame:
    return pd.DataFrame({"ocean_proximity": [ISLAND]})


# --- the unseen-category red-team -------------------------------------------


@pytest.mark.parametrize("encoder", ENCODERS)
def test_a_fit_that_never_saw_the_island_still_transforms_one(
    encoder: str, island_free: pd.DataFrame, island_only: pd.DataFrame
) -> None:
    """THE requirement. Neither encoder may raise on the rare level."""
    block = build_categorical_block(encoder=encoder).fit(island_free)
    out = block.transform(island_only)
    assert len(out) == 1
    assert np.isfinite(np.asarray(out, dtype=float)).all()


def test_the_naive_encoder_does_raise(
    island_free: pd.DataFrame, island_only: pd.DataFrame
) -> None:
    """The other half: the default configuration is genuinely unsafe here.

    ``handle_unknown='error'`` is scikit-learn's default. On this dataset that
    is a pipeline which trains happily and then dies at scoring time on five
    rows out of 20 640.
    """
    naive = OneHotEncoder(handle_unknown="error", sparse_output=False)
    naive.fit(island_free)
    with pytest.raises(ValueError, match="Found unknown categories"):
        naive.transform(island_only)


def test_declared_categories_encode_a_level_the_fold_never_saw(
    island_free: pd.DataFrame, island_only: pd.DataFrame
) -> None:
    """The ordinal encoder's real advantage, and it is not about ordering.

    Because the levels come from the *schema* rather than the sample, ``ISLAND``
    gets its correct code — not an unknown marker — even though the fit never
    saw one.
    """
    encoder = make_ordinal_encoder().fit(island_free)
    code = float(encoder.transform(island_only)[0, 0])
    assert code == config.OCEAN_PROXIMITY_ORDER.index(ISLAND)
    assert code != UNKNOWN_VALUE


def test_undeclared_encoder_marks_it_unknown_instead(
    island_free: pd.DataFrame, island_only: pd.DataFrame
) -> None:
    """``declared=False`` is what a StringIndexer does — and it loses the level."""
    encoder = make_ordinal_encoder(declared=False).fit(island_free)
    assert float(encoder.transform(island_only)[0, 0]) == UNKNOWN_VALUE


def test_a_genuinely_novel_level_is_still_marked_unknown(
    cat_frame: pd.DataFrame,
) -> None:
    """Declaring the schema handles known levels, not unknowable ones."""
    encoder = make_ordinal_encoder().fit(cat_frame)
    novel = pd.DataFrame({"ocean_proximity": ["NEAR LAKE"]})
    assert float(encoder.transform(novel)[0, 0]) == UNKNOWN_VALUE


# --- the drop='first' collision ---------------------------------------------


def test_drop_with_tolerant_unknown_handling_is_refused() -> None:
    """Red-team: scikit-learn permits a configuration that silently merges two
    different categories into one vector."""
    with pytest.raises(UnsafeEncoderError) as exc:
        make_onehot_encoder(drop="first", handle_unknown="ignore")
    message = str(exc.value)
    assert "dropped reference level" in message
    assert "drop=None" in message


def test_drop_is_allowed_when_unknowns_are_impossible() -> None:
    """The guard must not ban the safe use of ``drop``."""
    encoder = make_onehot_encoder(drop="first", handle_unknown="error")
    assert encoder.drop == "first"


def test_the_collision_the_guard_prevents_is_real(
    island_free: pd.DataFrame, island_only: pd.DataFrame
) -> None:
    """Demonstrate the defect directly, with the guard bypassed.

    If a future scikit-learn ever stopped collapsing these two to the same
    vector, this test fails and ``UnsafeEncoderError`` becomes unnecessary
    strictness that should be deleted. The guard is only justified while this
    holds.
    """
    unsafe = OneHotEncoder(
        drop="first", handle_unknown="ignore", sparse_output=False
    ).fit(island_free)
    reference_level = unsafe.categories_[0][0]

    with pytest.warns(UserWarning, match="unknown categories"):
        unknown_row = unsafe.transform(island_only)
    reference_row = unsafe.transform(
        pd.DataFrame({"ocean_proximity": [reference_level]})
    )
    assert np.array_equal(unknown_row, reference_row), (
        "drop='first' no longer collapses unknown into the reference level; "
        "UnsafeEncoderError is now unnecessary and should be removed"
    )


def test_without_drop_the_two_are_distinguishable(
    island_free: pd.DataFrame, island_only: pd.DataFrame
) -> None:
    """Why ``drop=None`` is the default: the row sum carries the distinction.

    Note the asymmetry in scikit-learn's own reporting — it emits its
    ``will be encoded as all zeros`` UserWarning **only when ``drop`` is set**.
    Here, with no drop, the unknown row is encoded silently. The loud case is
    the safe one; the quiet case is the one that needs this check.
    """
    encoder = make_onehot_encoder(min_frequency=None, handle_unknown="ignore")
    encoder.fit(island_free)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        unknown_row = encoder.transform(island_only)
    assert not caught, f"sklearn now warns without drop: {[str(w.message) for w in caught]}"

    known_row = encoder.transform(pd.DataFrame({"ocean_proximity": ["INLAND"]}))
    assert unknown_row.sum() == 0.0
    assert known_row.sum() == 1.0


def test_min_frequency_gives_an_unknown_a_positive_signal(
    cat_frame: pd.DataFrame, island_only: pd.DataFrame
) -> None:
    """``infrequent_if_exist`` is a no-op without a bucket to map into.

    The bucket exists only because the rare level is present **at fit time** and
    falls under the threshold — which is exactly what M1-S2's split guarantees.
    Fitted here on the full frame (island included), as every real fold is.
    """
    assert (cat_frame["ocean_proximity"] == ISLAND).sum() < 10
    encoder = make_onehot_encoder(min_frequency=10).fit(cat_frame)
    names = list(encoder.get_feature_names_out())
    assert any("infrequent" in name for name in names), names

    # The rare level, and a level nobody has ever seen, both land in the bucket
    # as a PRESENCE rather than an absence.
    assert encoder.transform(island_only).sum() == 1.0
    novel = pd.DataFrame({"ocean_proximity": ["NEAR LAKE"]})
    assert encoder.transform(novel).sum() == 1.0


def test_the_infrequent_bucket_needs_the_rare_level_at_fit_time(
    island_free: pd.DataFrame, island_only: pd.DataFrame
) -> None:
    """The precondition, stated as a test rather than trusted as a habit.

    Without a below-threshold category in the training data there is no
    infrequent column at all, and ``min_frequency`` silently buys nothing. This
    is why ``splits.stratification_key`` guaranteeing ISLAND on the training
    side is load-bearing for the ENCODER, not only for the split — and if that
    guarantee were ever removed, this test says what would quietly break.
    """
    encoder = make_onehot_encoder(min_frequency=10).fit(island_free)
    assert not any("infrequent" in name for name in encoder.get_feature_names_out())
    assert encoder.transform(island_only).sum() == 0.0


# --- ordinary invariants -----------------------------------------------------


@pytest.mark.parametrize("encoder", ENCODERS)
def test_block_emits_pandas_with_usable_names(
    encoder: str, cat_frame: pd.DataFrame
) -> None:
    block = build_categorical_block(encoder=encoder)
    out = block.fit_transform(cat_frame)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(cat_frame)
    assert all(isinstance(name, str) and name for name in out.columns)


def test_onehot_widens_and_ordinal_does_not(cat_frame: pd.DataFrame) -> None:
    wide = build_categorical_block(encoder="onehot").fit_transform(cat_frame)
    narrow = build_categorical_block(encoder="ordinal").fit_transform(cat_frame)
    assert wide.shape[1] > 1
    assert narrow.shape[1] == len(CAT)


def test_ordinal_codes_follow_the_declared_distance_order(
    cat_frame: pd.DataFrame,
) -> None:
    """The ordering must assert something true, or an ordinal code is noise.

    ISLAND (surrounded by water) must code below INLAND (farthest from it).
    """
    block = build_categorical_block(encoder="ordinal").fit(cat_frame)
    codes = {
        level: float(
            block.transform(pd.DataFrame({"ocean_proximity": [level]})).iloc[0, 0]
        )
        for level in config.OCEAN_PROXIMITY_ORDER
    }
    assert list(codes.values()) == sorted(codes.values())
    assert codes[ISLAND] < codes["INLAND"]


def test_unknown_encoder_name_is_refused(cat_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="expected one of"):
        build_categorical_block(encoder="target")


def test_nulls_in_the_string_column_are_imputed(cat_frame: pd.DataFrame) -> None:
    """The dataset has none. A block that assumes that breaks on the next extract."""
    holed = cat_frame.copy()
    holed.iloc[:3, 0] = np.nan
    out = build_categorical_block(encoder="ordinal").fit_transform(holed)
    assert not out.isna().any().any()


# --- the binned block (Bucketizer) ------------------------------------------


@pytest.mark.parametrize("encoder", ENCODERS)
@pytest.mark.parametrize("strategy", ["quantile", "uniform"])
def test_binned_block_turns_a_number_into_categories(
    encoder: str, strategy: str, housing: pd.DataFrame
) -> None:
    frame = housing[["median_income", "housing_median_age"]]
    out = build_binned_block(n_bins=5, strategy=strategy, encoder=encoder).fit_transform(
        frame
    )
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(frame)
    if encoder == "onehot":
        assert out.shape[1] > frame.shape[1]
        np.testing.assert_allclose(out.sum(axis=1).to_numpy(), 1.0 * frame.shape[1])
    else:
        assert out.shape[1] == frame.shape[1]
        assert out.to_numpy().min() >= 0


def test_binning_is_reproducible(housing: pd.DataFrame) -> None:
    """``KBinsDiscretizer`` subsamples by default, which makes the learned bin
    edges depend on a random draw. ``subsample=None`` is what stops that."""
    frame = housing[["median_income"]]
    a = build_binned_block().fit_transform(frame)
    b = build_binned_block().fit_transform(frame)
    pd.testing.assert_frame_equal(a, b)


def test_binned_block_refuses_unknown_options(housing: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="expected one of"):
        build_binned_block(encoder="hashing")
    with pytest.raises(ValueError, match="expected one of"):
        build_binned_block(strategy="jenks")


def test_bin_edges_come_from_train_only(housing: pd.DataFrame) -> None:
    """Leakage, again as a property: transforming a row alone matches
    transforming it in company."""
    frame = housing[["median_income"]]
    train, test = frame.iloc[:400], frame.iloc[400:]
    block = build_binned_block(encoder="ordinal").fit(train)
    together = block.transform(test)
    for position in (0, 3, len(test) - 1):
        alone = block.transform(test.iloc[[position]])
        np.testing.assert_allclose(
            alone.to_numpy(), together.iloc[[position]].to_numpy()
        )


def test_real_file_island_survives_a_fold_without_it(real_housing: pd.DataFrame) -> None:
    """Skipped on CI. The same red-team, on the actual five rows."""
    frame = real_housing[CAT]
    island_free = frame[frame["ocean_proximity"] != ISLAND]
    islands = frame[frame["ocean_proximity"] == ISLAND]
    assert len(islands) == 5
    for encoder in ENCODERS:
        block = build_categorical_block(encoder=encoder).fit(island_free)
        out = block.transform(islands)
        assert len(out) == 5
        assert np.isfinite(np.asarray(out, dtype=float)).all()
