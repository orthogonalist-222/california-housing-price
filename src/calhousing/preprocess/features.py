"""Engineered features — the stage Spark would have written as a UDF.

In a Spark pipeline these would be a `withColumn` chain or a UDF, applied
before the assembler. Here they are ``TransformerMixin`` subclasses, and the
difference is not stylistic: a transformer is **fitted**, so it composes into a
``Pipeline`` and is refitted inside every cross-validation fold.

For the ratios that distinction buys nothing today — they are stateless. For
``ClusterSimilarity`` it is the whole ballgame: its centroids are learned, and
learning them on the full dataset before splitting is a textbook leak that
improves every score and invalidates all of them.

Keeping both in the same shape means the habit survives the next person who
adds a stateful feature.

Why these features
------------------
M1-S3 measured the four raw count columns correlating strongly with **each
other** (a big block group has more of everything) and weakly with price. The
information is in the *ratios*: rooms per household is a proxy for dwelling
size, bedrooms per room for how much of that is sleeping space, population per
household for crowding. None of that is visible in the totals.

The geography figure showed the other half: price is spatial, and neither
``longitude`` nor ``latitude`` is monotone in it. A linear model cannot express
"near the Bay Area" from two coordinates. ``ClusterSimilarity`` gives it a
basis of smooth bumps over the map, which it can.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.validation import check_is_fitted

from .. import config

__all__ = ["RATIO_COLUMNS", "RatioFeatures", "ClusterSimilarity", "safe_divide"]

#: The engineered ratio names, in output order.
RATIO_COLUMNS = [
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
]

#: What a ratio becomes when its denominator is zero. Not NaN: a NaN leaving a
#: transformer that sits DOWNSTREAM of the imputer is a NaN nothing will ever
#: fill, and it surfaces as an error in the estimator.
ZERO_DENOMINATOR_FILL = 0.0


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    fill: float = ZERO_DENOMINATOR_FILL,
) -> pd.Series:
    """Element-wise division that yields ``fill`` instead of ``inf`` or NaN.

    **No row in this dataset has a zero denominator** — `households` and
    `total_rooms` are never 0, measured on the raw file. So this guard is not
    fixing an observed defect; it is refusing to depend on a property of one
    CSV. A block group with no households is not impossible, only absent here,
    and an ``inf`` reaching ``StandardScaler`` poisons a whole column with a
    message that points at the scaler rather than at the division.

    The tests for this use synthetic rows, and say so. Claiming the dataset
    forced this would be a nicer story and a false one.
    """
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill)


class RatioFeatures(BaseEstimator, TransformerMixin):
    """Append the three ratio features.

    Parameters
    ----------
    keep_original:
        When True (default) the input columns are retained alongside the
        ratios. Set False to hand a model the ratios *instead of* the totals,
        which M3-S2 searches as an option — the totals carry block-group size,
        which is either a useful feature or a nuisance depending on the model.
    """

    def __init__(self, keep_original: bool = True) -> None:
        self.keep_original = keep_original

    def fit(self, X, y=None):  # noqa: D102 - stateless, but sklearn requires it
        frame = self._as_frame(X)
        missing = [
            column
            for column in ("total_rooms", "total_bedrooms", "population", "households")
            if column not in frame.columns
        ]
        if missing:
            raise ValueError(
                f"RatioFeatures needs {missing} to compute {RATIO_COLUMNS}; "
                f"got columns {list(frame.columns)}. This usually means the "
                "ColumnTransformer routed the wrong columns here."
            )
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        frame = self._as_frame(X)
        ratios = pd.DataFrame(index=frame.index)
        ratios["rooms_per_household"] = safe_divide(
            frame["total_rooms"], frame["households"]
        )
        ratios["bedrooms_per_room"] = safe_divide(
            frame["total_bedrooms"], frame["total_rooms"]
        )
        ratios["population_per_household"] = safe_divide(
            frame["population"], frame["households"]
        )
        if self.keep_original:
            return pd.concat([frame, ratios], axis=1)
        return ratios

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        if self.keep_original:
            return np.asarray(
                list(self.feature_names_in_) + RATIO_COLUMNS, dtype=object
            )
        return np.asarray(RATIO_COLUMNS, dtype=object)

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """Similarity to ``n_clusters`` learned geographic centres.

    ``KMeans`` over (latitude, longitude), then an RBF kernel from each row to
    each centre. The output is ``n_clusters`` smooth columns in [0, 1], one per
    region, peaking where that region is.

    This is the transformer that makes the fit/transform discipline matter. The
    centroids are **learned parameters**. Fitting them on the whole dataset
    before splitting — which is what happens the moment this is written as a
    preprocessing function instead of a transformer — lets every test row
    contribute to the definition of the regions it is later scored against.
    The score improves and means nothing.

    Parameters
    ----------
    n_clusters:
        How many regions. Searched in M3-S2; the right number is not knowable
        from first principles.
    gamma:
        RBF width. Large gamma gives narrow, local bumps; small gamma gives
        broad overlapping ones that approach a constant and carry no signal.
    weight_by:
        Column used as ``sample_weight`` for KMeans, or None. ``population``
        places the centres where people are rather than where empty land is.
    """

    def __init__(
        self,
        n_clusters: int = 10,
        gamma: float = 1.0,
        weight_by: str | None = "population",
        random_state: int = config.RANDOM_SEED,
    ) -> None:
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.weight_by = weight_by
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        frame = self._as_frame(X)
        coords = self._coordinates(frame)

        if sample_weight is None and self.weight_by and self.weight_by in frame:
            sample_weight = frame[self.weight_by].to_numpy()
        if sample_weight is not None:
            # A zero or negative weight is rejected by KMeans with a message
            # that does not name the column it came from.
            sample_weight = np.clip(np.asarray(sample_weight, dtype=float), 1e-9, None)

        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=self.random_state,
        ).fit(coords, sample_weight=sample_weight)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X) -> pd.DataFrame:
        check_is_fitted(self, "kmeans_")
        frame = self._as_frame(X)
        similarity = rbf_kernel(
            self._coordinates(frame), self.kmeans_.cluster_centers_, gamma=self.gamma
        )
        return pd.DataFrame(
            similarity, columns=self.get_feature_names_out(), index=frame.index
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "kmeans_")
        return np.asarray(
            [f"geo_cluster_{index}_similarity" for index in range(self.n_clusters)],
            dtype=object,
        )

    @property
    def centroids_(self) -> np.ndarray:
        """The learned centres, as (latitude, longitude) pairs."""
        check_is_fitted(self, "kmeans_")
        return self.kmeans_.cluster_centers_

    @staticmethod
    def _coordinates(frame: pd.DataFrame) -> np.ndarray:
        missing = [c for c in ("latitude", "longitude") if c not in frame.columns]
        if missing:
            raise ValueError(
                f"ClusterSimilarity needs {missing}; got columns "
                f"{list(frame.columns)}. This usually means the "
                "ColumnTransformer routed the wrong columns here."
            )
        return frame[["latitude", "longitude"]].to_numpy(dtype=float)

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
