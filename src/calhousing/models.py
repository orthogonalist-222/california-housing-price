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
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_validate
from xgboost import XGBRegressor

from . import config
from .preprocess.assemble import build_pipeline
from .preprocess.numeric import make_deskew, make_imputer

__all__ = [
    "ModelSpec",
    "TUNED",
    "tuned_pipeline",
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


def _xgboost() -> BaseEstimator:
    # `n_jobs=1` for the same reason as the forest: the SEARCH parallelises
    # across draws, and a thread pool inside a process pool oversubscribes.
    return XGBRegressor(
        random_state=config.RANDOM_SEED,
        n_jobs=1,
        tree_method="hist",
        # XGBoost 2.x renamed several parameters. Setting the ones this project
        # uses explicitly means a version bump inside the declared range cannot
        # silently change a default underneath the recorded results.
        objective="reg:squarederror",
    )


def _lightgbm() -> BaseEstimator:
    return LGBMRegressor(
        random_state=config.RANDOM_SEED,
        n_jobs=1,
        # LightGBM prints a per-fit banner otherwise; across 25 draws x 5 folds
        # that is 125 banners burying the only line anybody reads.
        verbosity=-1,
    )


#: The winning configurations from the M3-S2 and M3-S3 searches.
#:
#: Recorded in CODE, not read from ``artifacts/search/*.joblib``. Artifacts are
#: cattle - the directory is gitignored and may be empty on any checkout - so a
#: stack that loaded them would be a recipe that cannot rebuild itself. These
#: numbers are transcribed from the search records committed in each story's PR.
#:
#: Each entry separates the estimator's own parameters from the preprocessing
#: the search chose for it, because the two are set through different doors.
TUNED: dict[str, dict[str, Any]] = {
    "ridge": {
        "model": {"alpha": 10.0},
        "preprocess": {},
    },
    "rf": {  # CV RMSE 42,707 - M3-S3 re-run
        "model": {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": 0.3,
        },
        "preprocess": {"n_clusters": 45, "gamma": 3.0, "n_bins": 8, "deskew": "none"},
    },
    "hgb": {  # CV RMSE 43,571
        "model": {
            "learning_rate": 0.05,
            "max_iter": 600,
            "max_leaf_nodes": 127,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        },
        "preprocess": {"n_clusters": 30, "gamma": 0.1, "n_bins": 5, "deskew": "log"},
    },
    "xgb": {  # CV RMSE 42,401
        "model": {
            "n_estimators": 600,
            "learning_rate": 0.05,
            "max_depth": 8,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "min_child_weight": 5,
        },
        "preprocess": {
            "n_clusters": 30,
            "gamma": 1.0,
            "n_bins": 3,
            "deskew": "yeo-johnson",
        },
    },
    "lgbm": {  # CV RMSE 42,166 - the best score AND the cheapest artefact
        "model": {
            "n_estimators": 600,
            "learning_rate": 0.05,
            "num_leaves": 127,
            "min_child_samples": 20,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        },
        "preprocess": {
            "n_clusters": 20,
            "gamma": 0.3,
            "n_bins": 12,
            "deskew": "yeo-johnson",
        },
    },
}

#: Base learners for the stack, and the one preprocessing they share.
#:
#: ``rf`` is deliberately absent: 289 MB and 115 s per fit for a score
#: statistically indistinguishable from LightGBM's 6.8 MB and 5.1 s (RT-018).
#: Inside a stack that cost is multiplied by the internal CV.
#:
#: The three tree learners have genuinely different inductive biases -
#: level-wise (xgb), histogram (hgb), leaf-wise (lgbm) - plus a linear arm, so
#: the meta-learner has real disagreement to arbitrate rather than four copies
#: of one opinion.
STACK_MEMBERS = ("lgbm", "xgb", "hgb", "ridge")

#: The stack shares ONE preprocessing configuration - LightGBM's winner - rather
#: than nesting each arm's own. Nesting would mean six full preprocessing fits
#: per stack fit, and the arms disagreed about preprocessing by less than their
#: fold noise anyway (RT-018). Stated because it is a real simplification, not
#: a detail: the base learners are NOT exactly the models that were tuned.
STACK_PREPROCESS = TUNED["lgbm"]["preprocess"]


def tuned_estimator(name: str) -> BaseEstimator:
    """The registry's estimator, configured with its recorded winning params."""
    estimator = get_model(name).factory()
    params = TUNED.get(name, {}).get("model", {})
    if params:
        estimator.set_params(**params)
    return estimator


def tuned_pipeline(name: str):
    """A full pipeline at the configuration its search actually chose."""
    return build_pipeline(tuned_estimator(name), **TUNED.get(name, {}).get("preprocess", {}))


def _stack() -> BaseEstimator:
    return StackingRegressor(
        estimators=[(name, tuned_estimator(name)) for name in STACK_MEMBERS],
        # RidgeCV rather than a plain average: the meta-learner has to be able
        # to say that one base learner adds nothing, and a fixed average cannot.
        final_estimator=RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)),
        cv=build_cv(5),
        passthrough=False,
        n_jobs=1,
    )


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
    "xgb": ModelSpec(
        name="xgb",
        factory=_xgboost,
        why=(
            "The library most Kaggle tabular baselines are written in. Included "
            "so the comparison is against what a reader would actually reach "
            "for, not only against what ships with scikit-learn."
        ),
        search_space={
            "model__n_estimators": [300, 600, 1000],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [4, 6, 8, 10],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__reg_lambda": [0.5, 1.0, 5.0],
            "model__min_child_weight": [1, 5, 20],
        },
        cost="medium",
    ),
    "lgbm": ModelSpec(
        name="lgbm",
        factory=_lightgbm,
        why=(
            "Leaf-wise growth, which is a genuinely different inductive bias "
            "from the level-wise trees above - so it is a real second opinion "
            "for the stack in M3-S4, not a fourth copy of the same one."
        ),
        search_space={
            "model__n_estimators": [300, 600, 1000],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__num_leaves": [31, 63, 127, 255],
            "model__min_child_samples": [5, 20, 50],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__subsample_freq": [1],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__reg_lambda": [0.0, 1.0, 5.0],
        },
        cost="medium",
    ),
    "stack": ModelSpec(
        name="stack",
        factory=_stack,
        why=(
            "Four base learners with genuinely different inductive biases, "
            "arbitrated by a RidgeCV meta-learner. The arm that answers whether "
            "combining them buys anything over the best single model."
        ),
        cost="slow",
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
    # Widened in M3-S3: both M3-S2 winners chose the TOP of the old ranges
    # (n_clusters=20, n_bins=8), which means the search was cut off rather
    # than converged. A boundary winner is a range that has not been tested.
    "preprocess__geo__n_clusters": [5, 10, 20, 30, 45],
    "preprocess__geo__gamma": [0.1, 0.3, 1.0, 3.0],
    "preprocess__binned__bin__n_bins": [3, 5, 8, 12, 16],
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
