"""The assembler — Spark's ``VectorAssembler``, which is a ``ColumnTransformer``.

This is where the three previous stories become one object. Six branches, each
taking the columns it is defined for, all of them fitted inside whatever fold
the caller is in:

===============  ==========================================  =================
branch           columns                                     what it does
===============  ==========================================  =================
``geo``          latitude, longitude, population             KMeans + RBF
``ratios``       the four counts                             impute -> ratios -> scale
``heavy``        the four counts                             impute -> de-skew -> scale
``plain``        longitude, latitude, age, income            impute -> scale
``categorical``  ocean_proximity                             impute -> encode
``binned``       median_income, housing_median_age           impute -> KBins -> encode
===============  ==========================================  =================

Columns appear in more than one branch on purpose. ``total_rooms`` is consumed
by ``ratios`` (as a denominator) and by ``heavy`` (as a de-skewed magnitude);
``median_income`` is both a scaled number and a set of bands. A
``ColumnTransformer`` is a fan-out, not a partition, and that is exactly what
makes it the right analogue for ``VectorAssembler``.

The one thing that matters
--------------------------
``build_pipeline`` returns a **single estimator** whose ``fit`` does all of the
above and then trains a model. Hand that to ``cross_val_score`` and every
learned statistic — medians, scaler means, bin edges, KMeans centroids, the
encoder's category list — is relearned from each training fold. Hand it a
preprocessed matrix instead and all of those were learned once, from
everything, including the rows you are about to score.

That is the difference this project exists to make visible, and
``tests/test_assemble.py`` measures it rather than asserting it.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .. import config
from .categorical import build_binned_block, build_categorical_block
from .features import ClusterSimilarity, QuantileClipper, RatioFeatures
from .numeric import build_numeric_block, make_imputer, make_scaler

__all__ = ["BRANCHES", "build_preprocessor", "build_pipeline", "feature_names"]

#: Branch names, in assembly order. Public: M3-S2 addresses them by path
#: (``preprocess__heavy__deskew``), so a rename silently empties a search space.
BRANCHES = ("geo", "ratios", "heavy", "plain", "categorical", "binned")

#: Columns that are numeric but must not be de-skewed. ``longitude`` is around
#: -124, where ``log1p`` is undefined (M2-S1, F-001); ``median_income`` and
#: ``housing_median_age`` are already well behaved.
_PLAIN_NUMERIC = ["longitude", "latitude", "housing_median_age", "median_income"]

_COUNT_COLUMNS = ["total_rooms", "total_bedrooms", "population", "households"]
_GEO_COLUMNS = ["latitude", "longitude", "population"]
_BINNED_COLUMNS = ["median_income", "housing_median_age"]


def _ratio_branch(imputer: str, scaler: str, seed: int, clip: float) -> Pipeline:
    """Impute, derive the ratios, clip the tails, then scale.

    Two orderings here are load-bearing.

    **Impute before deriving.** A ratio computed from a column with a null is a
    null, and nothing downstream would fill it — ``bedrooms_per_room`` would
    carry the 207 missing rows straight into the model.

    **Clip before scaling.** The ratios are unbounded and this dataset contains
    institutions: one block group has 6 households and 7 460 people. Scaled
    without clipping, that row is a z-score in the hundreds and a linear model
    extrapolates from it catastrophically - measured as a single fold at
    RMSE 1 805 055 against neighbours around 65 000. See ``QuantileClipper``.
    """
    block = Pipeline(
        steps=[
            ("impute", make_imputer(imputer, seed=seed)),
            ("ratios", RatioFeatures(keep_original=False)),
            ("clip", QuantileClipper(lower=clip, upper=1.0 - clip)),
            ("scale", make_scaler(scaler)),
        ]
    )
    return block.set_output(transform="pandas")


def build_preprocessor(
    *,
    imputer: str = "median",
    scaler: str = "standard",
    deskew: str = "log",
    encoder: str = "onehot",
    n_clusters: int = 10,
    gamma: float = 1.0,
    n_bins: int = 5,
    clip: float = 0.001,
    seed: int = config.RANDOM_SEED,
) -> ColumnTransformer:
    """Assemble every branch into one ``ColumnTransformer``.

    Every argument here is a hyperparameter of the *preprocessing*, and M3-S2
    searches them jointly with the model's own — which is what Spark's
    ``ParamGridBuilder`` does over a ``Pipeline``, and what a pipeline built by
    hand out of ``fit_transform`` calls can never do.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "geo",
                ClusterSimilarity(
                    n_clusters=n_clusters, gamma=gamma, random_state=seed
                ),
                _GEO_COLUMNS,
            ),
            ("ratios", _ratio_branch(imputer, scaler, seed, clip), _COUNT_COLUMNS),
            (
                "heavy",
                build_numeric_block(
                    imputer=imputer, deskew=deskew, scaler=scaler, seed=seed
                ),
                config.HEAVY_TAILED_COLUMNS,
            ),
            (
                "plain",
                build_numeric_block(
                    imputer=imputer, deskew="none", scaler=scaler, seed=seed
                ),
                _PLAIN_NUMERIC,
            ),
            (
                "categorical",
                build_categorical_block(encoder=encoder),
                config.CATEGORICAL_COLUMNS,
            ),
            (
                "binned",
                build_binned_block(n_bins=n_bins, encoder=encoder, seed=seed),
                _BINNED_COLUMNS,
            ),
        ],
        # Anything not named above is dropped rather than passed through. A
        # passthrough default is how a leaked column reaches a model without
        # anybody choosing to put it there.
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor.set_output(transform="pandas")


def build_pipeline(estimator: BaseEstimator, **preprocessor_kwargs: Any) -> Pipeline:
    """Preprocessing and model as one estimator.

    The two step names, ``preprocess`` and ``model``, are the public handles
    every search and every evaluation uses.
    """
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(**preprocessor_kwargs)),
            ("model", estimator),
        ]
    )


def feature_names(fitted: Pipeline | ColumnTransformer) -> list[str]:
    """The output column names of a fitted pipeline or preprocessor.

    Names are prefixed by branch (``geo__geo_cluster_3_similarity``), so a
    feature-importance plot says both *what* a feature is and *which stage
    produced it* — which is the difference between "the model likes
    ``total_rooms``" and "the model likes the de-skewed ``total_rooms``, not
    the one inside the ratios".
    """
    transformer = (
        fitted.named_steps["preprocess"] if isinstance(fitted, Pipeline) else fitted
    )
    return list(transformer.get_feature_names_out())
