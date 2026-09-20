"""The split's invariants — the properties that must hold for every seed.

Nothing here asserts a row count or a score. What it asserts is that the split
partitions the data, that it is reproducible, and that the rare category
reaches both sides **whatever seed is used** — which is the entire reason
``stratification_key`` collapses levels instead of taking the obvious path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from calhousing import config
from calhousing.splits import (
    coverage_report,
    income_band,
    split_train_test,
    stratification_key,
)

from conftest import make_housing

SEEDS = [0, 1, 7, 42, 2024]


def test_split_is_a_partition(housing: pd.DataFrame) -> None:
    train, test = split_train_test(housing)
    assert len(train) + len(test) == len(housing)
    assert set(train.index).isdisjoint(test.index)
    assert set(train.index) | set(test.index) == set(housing.index)


def test_split_is_reproducible(housing: pd.DataFrame) -> None:
    a_train, a_test = split_train_test(housing, seed=config.RANDOM_SEED)
    b_train, b_test = split_train_test(housing, seed=config.RANDOM_SEED)
    pd.testing.assert_frame_equal(a_train, b_train)
    pd.testing.assert_frame_equal(a_test, b_test)


def test_different_seeds_give_different_splits(housing: pd.DataFrame) -> None:
    """Guards against a stratification key so coarse it pins the split."""
    a, _ = split_train_test(housing, seed=0)
    b, _ = split_train_test(housing, seed=1)
    assert set(a.index) != set(b.index)


@pytest.mark.parametrize("seed", SEEDS)
def test_rare_category_reaches_both_sides_at_every_seed(seed: int) -> None:
    """THE invariant this module exists for.

    Stratifying on income band alone makes this a lottery — on the real file,
    seed 7 gives the test set zero islands and seed 42 gives train only two of
    five. Collapsing the level into its own stratum makes it deterministic.
    """
    frame = make_housing(n_island=5)
    train, test = split_train_test(frame, seed=seed)
    report = coverage_report(train, test)
    assert set(report.index) == set(frame["ocean_proximity"].unique())
    assert (report["train"] > 0).all(), f"a level is missing from train:\n{report}"
    assert (report["test"] > 0).all(), f"a level is missing from test:\n{report}"


def test_naive_band_stratification_is_the_lottery_this_replaces() -> None:
    """Red-team the alternative: show the simpler design actually failing.

    This is the planted defect for the split. Over a sweep of seeds the
    band-only strategy must produce at least one split where the rare level is
    absent from a side — if it ever stopped doing so, the collapse in
    ``stratification_key`` would be unnecessary complexity and should be
    deleted. The assertion is on the *existence of a failure*, not on which
    seed fails, so it does not encode a scipy-version-specific draw.
    """
    frame = make_housing(n=600, n_island=5, seed=3)
    band = income_band(frame)
    failures = 0
    for seed in range(30):
        train, test = train_test_split(
            frame, test_size=0.2, random_state=seed, stratify=band
        )
        if (train["ocean_proximity"] == "ISLAND").sum() == 0:
            failures += 1
        if (test["ocean_proximity"] == "ISLAND").sum() == 0:
            failures += 1
    assert failures > 0, (
        "band-only stratification covered the rare level at all 30 seeds; "
        "if that is genuinely reliable, stratification_key's collapse is dead code"
    )


def test_composite_key_without_collapse_is_unsplittable() -> None:
    """The other rejected design: sklearn refuses a stratum of one."""
    frame = make_housing(n=600, n_island=5, seed=3)
    naive = income_band(frame).astype(str) + "|" + frame["ocean_proximity"]
    counts = naive.value_counts()
    if counts.min() >= 2:
        pytest.skip("this synthetic draw happens not to produce a singleton group")
    with pytest.raises(ValueError, match="least populated class"):
        train_test_split(frame, test_size=0.2, random_state=0, stratify=naive)


def test_collapse_is_per_level_not_per_group() -> None:
    """Collapsing only the offending group would leave a stratum of one.

    The key for every row of a collapsed level must be the bare level name —
    that is what pulls the level's rows together into one splittable stratum.
    """
    frame = make_housing(n=600, n_island=5, seed=3)
    key = stratification_key(frame)
    island_keys = set(key[frame["ocean_proximity"] == "ISLAND"])
    assert island_keys == {"ISLAND"}, island_keys
    assert (key.value_counts() >= 2).all()


def test_common_levels_keep_their_band_component() -> None:
    """The collapse must not quietly coarsen the whole key.

    If every level collapsed, income stratification would silently stop
    happening and nothing would fail loudly.
    """
    frame = make_housing(n=2000, seed=5)
    key = stratification_key(frame)
    common = frame["ocean_proximity"] == "<1H OCEAN"
    assert all("|" in k for k in key[common].unique())


def test_income_distribution_survives_the_split(housing: pd.DataFrame) -> None:
    """What stratification is *for*: the band mix is preserved on both sides."""
    train, test = split_train_test(housing)
    whole = income_band(housing).value_counts(normalize=True).sort_index()
    for side in (train, test):
        side_mix = income_band(side).value_counts(normalize=True).sort_index()
        np.testing.assert_allclose(side_mix.values, whole.values, atol=0.02)


def test_index_is_preserved(housing: pd.DataFrame) -> None:
    """A row must be traceable back to the raw file when a prediction is odd."""
    train, test = split_train_test(housing)
    assert train.index.isin(housing.index).all()
    assert test.index.isin(housing.index).all()


def test_impossible_stratum_is_refused_by_name() -> None:
    """A level with a single row cannot be split by any grouping.

    The refusal must name the level rather than letting scikit-learn raise
    three frames deeper with no mention of ``ocean_proximity``.
    """
    frame = make_housing(n=300, n_island=1, seed=11)
    with pytest.raises(ValueError, match="Cannot stratify"):
        stratification_key(frame)


def test_real_file_coverage_is_deterministic(real_housing: pd.DataFrame) -> None:
    """Skipped on CI. Locally: the same coverage at every seed, on real data."""
    reports = [
        coverage_report(*split_train_test(real_housing, seed=seed)) for seed in SEEDS
    ]
    for report in reports:
        assert (report["train"] > 0).all() and (report["test"] > 0).all()
    first = reports[0].loc["ISLAND"]
    for report in reports[1:]:
        assert report.loc["ISLAND"].equals(first)
