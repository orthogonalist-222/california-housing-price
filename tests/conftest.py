"""Shared fixtures.

Everything here is **synthetic**. CI has no Kaggle credentials and no
``data/raw/housing.csv``, so a test suite that needs the real file is a test
suite that only runs on one machine. The synthetic frame reproduces the raw
file's *structure* and its three awkward properties - a rare category, nulls in
``total_bedrooms``, a censored target - without reproducing its size.

Tests that genuinely need the real file use ``real_housing``, which **skips**
when it is absent rather than failing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calhousing import config


def make_housing(
    n: int = 600,
    *,
    seed: int = 0,
    n_island: int = 5,
    n_null_bedrooms: int = 7,
    n_capped: int = 20,
) -> pd.DataFrame:
    """A synthetic frame shaped exactly like the raw CSV.

    The defaults mirror the real file's awkward parts at small scale: a
    five-row ``ISLAND`` level, nulls confined to ``total_bedrooms``, and a
    cluster of rows pinned to the target cap.
    """
    rng = np.random.default_rng(seed)

    # Log-normal, not uniform. The real file's count columns have skew 3.4-4.9
    # (measured in M1-S3), and a uniform draw reproduces the column NAMES while
    # losing the property those columns exist to exercise - which is how a
    # de-skew test came to fail against a fixture that was never skewed.
    households = np.clip(rng.lognormal(5.6, 0.85, n), 5.0, 6000.0)
    rooms = households * rng.uniform(3.0, 9.0, n)
    frame = pd.DataFrame(
        {
            "longitude": rng.uniform(-124.0, -115.0, n),
            "latitude": rng.uniform(32.6, 41.9, n),
            "housing_median_age": rng.integers(1, 52, n).astype(float),
            "total_rooms": rooms.round(),
            "total_bedrooms": (rooms * rng.uniform(0.15, 0.30, n)).round(),
            "population": households * rng.uniform(1.5, 4.0, n),
            "households": households,
            "median_income": rng.uniform(0.5, 15.0, n),
            # A target with REAL SIGNAL, not a uniform draw - filled in below,
            # once every predictor column exists. Kept in place here so the
            # column ORDER still matches the raw CSV.
            #
            # Found in M3-S1: with a random target no model can beat a median
            # predictor, so `test_a_real_model_beats_the_dummy` failed against
            # correct code. A fixture that reproduces a dataset's shape but not
            # its LEARNABILITY cannot test anything above the transformer level.
            config.TARGET: np.nan,
            "ocean_proximity": rng.choice(
                ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY"], n
            ),
        }
    )

    # Deliberately crude: income drives price, the southern coast is dearer,
    # crowding is cheap. The tests assert RELATIONSHIPS (a model beats the
    # median, the inland segment can look worse), never coefficients.
    frame[config.TARGET] = (
        30_000.0
        + 28_000.0 * frame["median_income"]
        + 1_200.0 * (42.0 - frame["latitude"])
        - 2_500.0 * (frame["population"] / frame["households"])
        + rng.normal(0.0, 35_000.0, n)
    ).clip(20_000.0, 480_000.0).round()

    # The rare level, planted deliberately: this is the property the split and
    # the encoders are built around.
    island_rows = rng.choice(n, size=n_island, replace=False)
    frame.loc[frame.index[island_rows], "ocean_proximity"] = "ISLAND"

    null_rows = rng.choice(n, size=n_null_bedrooms, replace=False)
    frame.loc[frame.index[null_rows], "total_bedrooms"] = np.nan

    capped_rows = rng.choice(n, size=n_capped, replace=False)
    frame.loc[frame.index[capped_rows], config.TARGET] = config.TARGET_CAP

    return frame


@pytest.fixture
def housing() -> pd.DataFrame:
    """A synthetic frame with the real file's structure and defects."""
    return make_housing()


@pytest.fixture
def real_housing() -> pd.DataFrame:
    """The actual raw CSV, or a skip when it has not been downloaded."""
    from calhousing import config as cfg
    from calhousing.data import load_raw

    try:
        cfg.resolve_csv()
    except cfg.DataNotFoundError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"raw CSV not available: {exc}")
    return load_raw(verbose=False)
