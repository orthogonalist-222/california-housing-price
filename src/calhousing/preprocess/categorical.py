"""The categorical block: label encoding, one-hot encoding, and binning.

The three Spark ML stages this project set out to translate:

======================  ==========================================
Spark ML                scikit-learn
======================  ==========================================
``StringIndexer``       ``OrdinalEncoder``
``OneHotEncoder``       ``OneHotEncoder``
``Bucketizer``          ``KBinsDiscretizer``
======================  ==========================================

The dataset has exactly one string column, and its rarest level has **five
rows** out of 20 640. Any split can leave that level out of a fold, so how an
encoder handles a category it never saw is not a footnote here — it is the
whole design question. Three things were measured before this module was
written, and each one changed it.

**1. ``infrequent_if_exist`` does nothing on its own — and the fix has a
precondition that this project's split is what supplies.**
With no ``min_frequency`` set it behaves *identically* to
``handle_unknown="ignore"``: an unknown row becomes all zeros. Setting
``min_frequency`` gives it a bucket to map into — but **only if a category in
the TRAINING data is actually below the threshold**. Measured::

    train CONTAINS the 4-row island, min_frequency=10
      names  = [... 'ocean_proximity_infrequent_sklearn']
      ISLAND -> [0, 0, 0, 0, 1]   sum=1.0
      novel  -> [0, 0, 0, 0, 1]   sum=1.0

    train has NO island, min_frequency=10
      names  = [...]                      <- no infrequent column at all
      ISLAND -> [0, 0, 0, 0]      sum=0.0
      novel  -> [0, 0, 0, 0]      sum=0.0

So the encoder gives an unknown a **positive** signal only when the rare level
was present at fit time. That is not a hope: ``splits.stratification_key``
guarantees ``ISLAND`` reaches the training side at every seed, precisely so
this column exists. The split design and the encoder design hold each other up,
and neither is sufficient alone.

**2. ``drop="first"`` destroys the distinction it looks like it keeps.**
Measured on scikit-learn 1.7.2::

    drop='first', handle_unknown='ignore'
    dropped reference level = '<1H OCEAN'
    unknown 'ISLAND'   -> [[0.0, 0.0]]
    known   '<1H OCEAN'-> [[0.0, 0.0]]
    IDENTICAL: True

An unseen category and the dropped baseline become the **same vector**. Without
``drop`` they are still distinguishable — a valid row sums to 1, an unknown row
sums to 0. This combination is therefore refused outright by
``make_onehot_encoder`` rather than merely discouraged in a docstring.

Worth knowing about the warning: scikit-learn emits its
``Found unknown categories ... will be encoded as all zeros`` ``UserWarning``
**only when ``drop`` is set**. Without ``drop``, an unknown row is silently
encoded as zeros and nothing is printed at all. The loud case is the safe one;
the quiet case is the one that needs the row-sum check.

**3. Declaring the categories up front removes the problem at the source.**
``OrdinalEncoder(categories=[OCEAN_PROXIMITY_ORDER])`` encodes ``ISLAND``
correctly even when the training fold contained none, because the *schema* —
not the sample — defines the levels. A genuinely novel level (something not in
the declared list at all) still needs ``unknown_value``.

That is also what makes an ordinal encoding defensible here rather than merely
convenient: the declared order is increasing distance from water, so
``0 < 1 < 2 < 3 < 4`` asserts something true. A ``StringIndexer`` ordering
levels by frequency asserts nothing, and any model that can use the ordering
will happily use the noise.
"""

from __future__ import annotations

import inspect

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, OrdinalEncoder

from .. import config

__all__ = [
    "ENCODERS",
    "BIN_STRATEGIES",
    "UnsafeEncoderError",
    "make_ordinal_encoder",
    "make_onehot_encoder",
    "build_categorical_block",
    "build_binned_block",
]

ENCODERS = ("onehot", "ordinal")
BIN_STRATEGIES = ("quantile", "uniform", "kmeans")

#: Encoded value for a level that is not in the declared schema at all.
#: Negative and out of range on purpose: a tree can split it off, and it is
#: visibly not a real category if it ever reaches a plot.
UNKNOWN_VALUE = -1


class UnsafeEncoderError(ValueError):
    """A requested encoder configuration is silently lossy.

    Raised rather than warned, because the loss it describes is invisible in
    the output: the numbers look fine, the shapes are right, and two different
    categories have simply become one.
    """


def make_ordinal_encoder(
    *,
    categories: list[str] | None = None,
    declared: bool = True,
) -> OrdinalEncoder:
    """Spark's ``StringIndexer``, with the ordering asserted rather than learned.

    Parameters
    ----------
    categories:
        The level order. Defaults to ``config.OCEAN_PROXIMITY_ORDER``
        (increasing distance from water).
    declared:
        When True the categories come from the schema, so a level missing from
        a training fold still encodes correctly. When False the encoder learns
        them from the data — which is what ``StringIndexer`` does, and what
        makes a five-row level a liability.
    """
    if not declared:
        return OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=UNKNOWN_VALUE
        )
    levels = list(categories or config.OCEAN_PROXIMITY_ORDER)
    return OrdinalEncoder(
        categories=[levels],
        handle_unknown="use_encoded_value",
        unknown_value=UNKNOWN_VALUE,
    )


def make_onehot_encoder(
    *,
    handle_unknown: str = "infrequent_if_exist",
    min_frequency: int | float | None = 10,
    drop: str | None = None,
) -> OneHotEncoder:
    """Spark's ``OneHotEncoder``, configured so an unknown level stays visible.

    ``min_frequency`` defaults to 10 so that ``infrequent_if_exist`` has a
    bucket to map into. Without it the option is a no-op and an unknown row is
    an all-zero row — indistinguishable from "every indicator happened to be
    off".

    The bucket is only created when some **training** category falls below the
    threshold. On this dataset that is ``ISLAND`` (4 rows on the training side),
    which is present in every fold only because ``splits.stratification_key``
    makes it so.

    Raises
    ------
    UnsafeEncoderError
        If ``drop`` is combined with anything but ``handle_unknown="error"``.
        scikit-learn permits it; the result is that an unseen category and the
        dropped reference level encode to the identical vector, with no error
        and no way to tell them apart downstream.
    """
    if drop is not None and handle_unknown != "error":
        raise UnsafeEncoderError(
            f"drop={drop!r} with handle_unknown={handle_unknown!r} encodes an "
            "unknown category identically to the dropped reference level - the "
            "two become the same vector and nothing downstream can separate "
            "them. Use drop=None (an unknown row then sums to 0 while a valid "
            "row sums to 1), or handle_unknown='error' if unknowns are "
            "genuinely impossible."
        )
    return OneHotEncoder(
        handle_unknown=handle_unknown,
        min_frequency=min_frequency,
        drop=drop,
        sparse_output=False,  # pandas output downstream; the matrix is tiny
    )


def _quantile_method_kwarg() -> dict[str, str]:
    """``quantile_method="averaged_inverted_cdf"`` — only where it exists.

    Pinning it is deliberate: scikit-learn 1.9 changes the default, and a
    silent change to the bin edges would move every downstream number with no
    diff to point at.

    But the parameter arrived in **1.7**, and this project declares
    ``scikit-learn>=1.5``. Kaggle's image sits below 1.7, so `pip` was already
    satisfied, installed nothing newer, and the published kernel died with::

        TypeError: KBinsDiscretizer.__init__() got an unexpected keyword
        argument 'quantile_method'

    Feature-detection rather than a version comparison: the question is whether
    this build accepts the argument, and the signature answers it directly.

    Behaviour is unchanged on older versions - below 1.7 the only behaviour is
    the one this value names.
    """
    if "quantile_method" in inspect.signature(KBinsDiscretizer.__init__).parameters:
        return {"quantile_method": "averaged_inverted_cdf"}
    return {}


def build_categorical_block(
    *,
    encoder: str = "onehot",
    impute: str = "most_frequent",
    **encoder_kwargs,
) -> Pipeline:
    """Assemble the block for genuine string columns.

    Step names ``impute`` and ``encode`` are public: M3-S2 addresses them by
    path, so renaming one silently empties a search space.
    """
    if encoder == "onehot":
        step: BaseEstimator = make_onehot_encoder(**encoder_kwargs)
    elif encoder == "ordinal":
        step = make_ordinal_encoder(**encoder_kwargs)
    else:
        raise ValueError(f"Unknown encoder {encoder!r}; expected one of {ENCODERS}")

    block = Pipeline(
        steps=[
            # The string column has no nulls in this dataset. The imputer is
            # here anyway: a block that only works on the file it was written
            # against breaks the first time it is pointed at a new extract, and
            # `most_frequent` is the one strategy that is defined for strings.
            ("impute", SimpleImputer(strategy=impute, missing_values=np.nan)),
            ("encode", step),
        ]
    )
    return block.set_output(transform="pandas")


def build_binned_block(
    *,
    n_bins: int = 5,
    strategy: str = "quantile",
    encoder: str = "onehot",
    seed: int = config.RANDOM_SEED,
) -> Pipeline:
    """Spark's ``Bucketizer`` / ``QuantileDiscretizer``: numeric → categorical.

    This is what turns a continuous column into something the two encoders
    above can act on, and it is why this project has more than one categorical
    column to talk about. ``median_income`` binned into five quantiles is the
    same construction the split stratifies on, and ``housing_median_age`` binned
    lets a linear model express "1960s housing stock" without a spline.

    ``subsample=None`` is deliberate: ``KBinsDiscretizer`` otherwise subsamples
    before computing quantiles, which makes the learned edges depend on a
    random draw and the whole pipeline non-reproducible for a reason nobody
    would think to look for.
    """
    if encoder not in ENCODERS:
        raise ValueError(f"Unknown encoder {encoder!r}; expected one of {ENCODERS}")
    if strategy not in BIN_STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; expected one of {BIN_STRATEGIES}")

    block = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            (
                "bin",
                KBinsDiscretizer(
                    n_bins=n_bins,
                    encode="onehot-dense" if encoder == "onehot" else "ordinal",
                    strategy=strategy,
                    subsample=None,
                    random_state=seed,
                    **_quantile_method_kwarg(),
                ),
            ),
        ]
    )
    return block.set_output(transform="pandas")
