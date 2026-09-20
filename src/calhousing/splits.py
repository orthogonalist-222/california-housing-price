"""The train/test split — where leakage is decided, before any model exists.

Two things are non-negotiable here and both were measured, not assumed.

**1. Stratify on income band.** ``median_income`` is the strongest single
predictor in this dataset. A purely random 20% holdout distorts its
distribution, and every number measured on top of that split inherits the
distortion.

**2. Guarantee rare-category coverage, deterministically.** ``ocean_proximity``
has a level with **five rows** (``ISLAND``). Stratifying on income band alone
leaves its coverage to the seed - measured on the real file:

===========  =====================  ====================
seed         ISLAND rows in train   ISLAND rows in test
===========  =====================  ====================
0            4                      1
**42**       **2**                  **3**
7            5                      **0**
2024         3                      2
===========  =====================  ====================

Seed 7 hands the test set *no* island at all; seed 42 keeps only two in train.
Either way the coverage is a lottery, and a lottery is not a protocol.

The obvious fix - stratify on ``income_band × ocean_proximity`` - does not work
either. ``band 3 × ISLAND`` has exactly **one** row, and scikit-learn refuses:
``The least populated class in y has only 1 member``.

So the key is built and then **collapsed per level**: if any composite group
within an ``ocean_proximity`` level is too small to split, every row of that
level is stratified by the level alone. ISLAND's five rows become one stratum
and split 4/1 whatever the seed is. Nothing else in the frame is affected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

__all__ = ["income_band", "stratification_key", "split_train_test"]


def income_band(frame: pd.DataFrame) -> pd.Series:
    """Bin ``median_income`` into the five bands used for stratification.

    Returned as a plain integer Series rather than a Categorical: it is a
    grouping key, and a Categorical's unused-category bookkeeping only causes
    surprises downstream.
    """
    band = pd.cut(
        frame["median_income"],
        bins=config.INCOME_BAND_EDGES,
        labels=config.INCOME_BAND_LABELS,
    )
    return band.astype("Int64").rename("income_band")


def stratification_key(frame: pd.DataFrame, *, min_per_stratum: int = 2) -> pd.Series:
    """Build ``income_band × ocean_proximity``, collapsing levels that cannot split.

    Parameters
    ----------
    min_per_stratum:
        The smallest group scikit-learn's stratified split can handle is 2. A
        larger value collapses more eagerly.

    Notes
    -----
    The collapse is **per level, not per group**. Collapsing only the offending
    group would leave ``ISLAND``'s single band-3 row in a stratum of one - the
    same refusal, one step later. Pulling the whole level together is what makes
    the stratum large enough to split.
    """
    band = income_band(frame)
    level = frame["ocean_proximity"].astype(str)
    key = band.astype(str).str.cat(level, sep="|")

    counts = key.value_counts()
    for name in level.unique():
        mask = level == name
        smallest = counts[key[mask].unique()].min()
        if smallest < min_per_stratum:
            key = key.mask(mask, name)

    # A level small enough to be its own stratum and STILL too small to split
    # cannot be handled by any grouping - say so here rather than letting
    # scikit-learn raise three frames deeper with no mention of the level.
    final = key.value_counts()
    impossible = final[final < min_per_stratum]
    if not impossible.empty:
        raise ValueError(
            "Cannot stratify: these strata have fewer than "
            f"{min_per_stratum} rows even after collapsing: {dict(impossible)}"
        )
    return key.rename("stratum")


def split_train_test(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train and test, stratified and reproducible.

    Returns frames with their original index preserved, so a row can be traced
    back to the raw file when a prediction looks wrong.
    """
    key = stratification_key(frame)
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=key,
        shuffle=True,
    )
    return train, test


def coverage_report(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Rows of each ``ocean_proximity`` level on each side of the split.

    Used by the tests and printed in the notebook. The point it exists to make
    is visible in one table: the rare level is present on both sides.
    """
    return pd.DataFrame(
        {
            "train": train["ocean_proximity"].value_counts(),
            "test": test["ocean_proximity"].value_counts(),
        }
    ).fillna(0).astype(int).sort_values("train", ascending=False)
