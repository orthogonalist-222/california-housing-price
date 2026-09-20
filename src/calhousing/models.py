"""The estimator registry and the cross-validation harness.

One place that knows which models exist, what each one's search space is, and
how any of them is scored. Scattering that across a notebook is how two arms of
a comparison end up cross-validated differently and the difference gets
attributed to the model.

The harness scores **the whole pipeline**, never a pre-transformed matrix. That
is the one rule here, and ``build_cv`` exists so no caller has to remember it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_validate

from . import config
from .preprocess.assemble import build_pipeline
from .preprocess.numeric import make_deskew, make_imputer

__all__ = [
    "ModelSpec",
    "REGISTRY",
    "get_model",
    "build_cv",
    "cross_validate_pipeline",
    "PREPROCESS_SEARCH_SPACE",
]

#: Scoring used everywhere. Negated because scikit-learn maximises; the harness
#: flips the sign back so every number a human reads is a positive error.
SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


@dataclass(frozen=True)
class ModelSpec:
    """An estimator, its search space, and what it is for.

    ``search_space`` keys are **pipeline paths** (``model__alpha``), not bare
    parameter names, because that is what ``RandomizedSearchCV`` receives once
    the estimator is wrapped. Keeping them in that form means a spec can be
    handed straight to a search with no translation step to get wrong.
    """

    name: str
    factory: Callable[[], BaseEstimator]
    why: str
    search_space: dict[str, Any] = field(default_factory=dict)
    #: Rough fit cost, used to keep a Kaggle kernel inside its budget.
    cost: str = "fast"


def _dummy() -> BaseEstimator:
    return DummyRegressor(strategy="median")


def _linear() -> BaseEstimator:
    return LinearRegression()


def _ridge() -> BaseEstimator:
    return Ridge(alpha=1.0)


def _random_forest() -> BaseEstimator:
    # n_jobs=1 on the estimator: the SEARCH parallelises across draws, and
    # nesting a thread pool inside a process pool oversubscribes the machine
    # and is reliably slower than either alone.
    return RandomForestRegressor(random_state=config.RANDOM_SEED, n_jobs=1)


def _hist_gradient_boosting() -> BaseEstimator:
    return HistGradientBoostingRegressor(random_state=config.RANDOM_SEED)


#: Every model this project knows about. M3-S2 and M3-S3 add to it; nothing
#: else creates estimators.
REGISTRY: dict[str, ModelSpec] = {
    "dummy": ModelSpec(
        name="dummy",
        factory=_dummy,
        why=(
            "Predicts the training median. The floor every other number is read "
            "against - an R2 of 0.7 means nothing until you know what 0.0 costs."
        ),
        cost="fast",
    ),
    "linear": ModelSpec(
        name="linear",
        factory=_linear,
        why=(
            "Unregularised least squares. Included to show what the engineered "
            "features buy a model that cannot invent interactions."
        ),
        cost="fast",
    ),
    "ridge": ModelSpec(
        name="ridge",
        factory=_ridge,
        why=(
            "The linear baseline that survives the one-hot columns. With 36 "
            "features, several of them collinear by construction, a penalty is "
            "not optional."
        ),
        search_space={"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        cost="fast",
    ),
    "rf": ModelSpec(
        name="rf",
        factory=_random_forest,
        why=(
            "Bagged trees. Invents the interactions a linear model cannot, and "
            "is indifferent to the scaling and de-skewing - which makes it the "
            "arm that shows how much of the pipeline the LINEAR model needed."
        ),
        search_space={
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": [0.3, 0.5, 0.7, 1.0],
        },
        cost="slow",
    ),
    "hgb": ModelSpec(
        name="hgb",
        factory=_hist_gradient_boosting,
        why=(
            "Histogram gradient boosting - the strongest thing in scikit-learn "
            "itself on tabular data of this size, and fast enough to search "
            "properly inside a Kaggle kernel budget."
        ),
        search_space={
            "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "model__max_iter": [200, 400, 600],
            "model__max_leaf_nodes": [15, 31, 63, 127],
            "model__min_samples_leaf": [5, 10, 20, 40],
            "model__l2_regularization": [0.0, 0.1, 1.0],
        },
        cost="medium",
    ),
}

#: Preprocessing hyperparameters, shared by every model's search.
#:
#: This is the part a hand-rolled pipeline cannot do. These are not model
#: settings - they are choices about imputation, encoding and feature geometry,
#: searched jointly with the model exactly as Spark's ``ParamGridBuilder``
#: searches a ``Pipeline``'s stages.
PREPROCESS_SEARCH_SPACE: dict[str, Any] = {
    # Imputers are passed as OBJECTS, not names: a search sets pipeline steps,
    # and "median" is a string the step would reject. `"passthrough"` would be
    # accepted and is a trap - it disables the step and lets the 207 nulls
    # through to an estimator that cannot take them.
    "preprocess__heavy__impute": [
        make_imputer("median"),
        make_imputer("mean"),
        make_imputer("knn"),
    ],
    "preprocess__heavy__deskew": [
        make_deskew("none"),
        make_deskew("log"),
        make_deskew("yeo-johnson"),
    ],
    "preprocess__geo__n_clusters": [5, 10, 15, 20],
    "preprocess__geo__gamma": [0.1, 0.3, 1.0, 3.0],
    "preprocess__binned__bin__n_bins": [3, 5, 8],
}


def get_model(name: str) -> ModelSpec:
    """Look up a spec, refusing an unknown name with the list of known ones."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown model {name!r}. Known models: {sorted(REGISTRY)}"
        ) from None


def build_cv(n_splits: int = 5, seed: int = config.RANDOM_SEED) -> KFold:
    """The one cross-validation object every comparison uses.

    Shuffled, seeded, and identical across arms. Two models scored under
    different folds are not comparable, and the difference gets attributed to
    the model.

    Not stratified: the target is continuous. The *train/test* split is
    stratified (M1-S2) because that boundary is drawn once and matters; within
    the training data, five shuffled folds of 16 512 rows reproduce the
    distribution closely enough that stratifying adds complexity and no signal.
    """
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def cross_validate_pipeline(
    model: str | BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
    seed: int = config.RANDOM_SEED,
    preprocessor_kwargs: dict[str, Any] | None = None,
    return_estimator: bool = False,
) -> dict[str, Any]:
    """Cross-validate a model **as a full pipeline**.

    ``model`` may be a registry name or an estimator. Either way it is wrapped
    in ``build_pipeline`` first, so every fold refits the imputers, scalers,
    encoders, bin edges and KMeans centroids from its own training rows.

    Returns mean and standard deviation for each metric, with the signs already
    flipped so the caller reads positive errors.
    """
    estimator = (
        clone(get_model(model).factory()) if isinstance(model, str) else clone(model)
    )
    pipeline = build_pipeline(estimator, **(preprocessor_kwargs or {}))

    raw = cross_validate(
        pipeline,
        X,
        y,
        cv=build_cv(n_splits, seed),
        scoring=SCORING,
        return_train_score=True,
        return_estimator=return_estimator,
        n_jobs=1,  # deterministic ordering; these fits are seconds, not hours
    )

    out: dict[str, Any] = {}
    for metric in SCORING:
        test = np.asarray(raw[f"test_{metric}"], dtype=float)
        train = np.asarray(raw[f"train_{metric}"], dtype=float)
        # r2 is already oriented "higher is better"; the error metrics arrive
        # negated and are flipped so every printed number is a positive error.
        sign = 1.0 if metric == "r2" else -1.0
        out[metric] = float((sign * test).mean())
        out[f"{metric}_std"] = float(test.std())
        out[f"{metric}_train"] = float((sign * train).mean())
    out["fit_seconds"] = float(np.asarray(raw["fit_time"], dtype=float).mean())
    if return_estimator:
        out["estimators"] = raw["estimator"]
    return out
