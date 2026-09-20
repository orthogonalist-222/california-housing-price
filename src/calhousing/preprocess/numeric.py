"""The numeric block: impute → de-skew → scale.

Spark ML's ``Imputer`` + ``StandardScaler``, rebuilt as a scikit-learn
``Pipeline`` so that every stateful step is fitted on training rows only.

Three choices are deliberately left open rather than hard-coded, because the
right answer is an empirical question this project is equipped to ask —
M3-S2 searches over them **jointly with the model**, which is what Spark's
``ParamGridBuilder`` does and what most tutorial pipelines never do:

``imputer``
    ``median`` (robust to the 3.4–4.9 skew measured in M1-S3), ``knn``, or
    ``iterative``. Only ``total_bedrooms`` is ever missing — 207 rows, 1% —
    so this is a small effect by construction, and the point of offering three
    is to be able to *say* it is small rather than assume it.

``deskew``
    ``none``, ``log`` (``log1p``), or ``yeo-johnson``. Applied only to the
    columns that are actually skewed; ``assemble`` routes them here separately
    from ``longitude``/``latitude``/``median_income``, which a log would only
    damage.

``scaler``
    ``standard``, ``robust``, ``minmax`` or ``none``. Tree ensembles do not
    care; the linear baseline and any distance-based imputer do.

The block always emits a pandas frame (``set_output(transform="pandas")``) so
feature names survive to the far end of the pipeline. A model whose importances
are labelled ``x17`` cannot be interrogated.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (registers IterativeImputer)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from .. import config

__all__ = [
    "DomainError",
    "IMPUTERS",
    "DESKEWERS",
    "SCALERS",
    "make_imputer",
    "make_deskew",
    "make_scaler",
    "build_numeric_block",
]

def _pandas(transformer):
    """Configure a transformer to emit pandas, and return it.

    Every factory below routes through this, and it is not decoration.
    ``build_numeric_block`` calls ``set_output`` on the *Pipeline*, which
    configures the steps that exist **at that moment**. A search that later
    replaces a step - ``preprocess__heavy__impute=SimpleImputer()`` - installs
    an estimator nobody configured, it returns a bare ndarray, and the branch
    loses its index.

    What that looks like downstream (F-006)::

        ValueError: Concatenating DataFrames from the transformer's output lead
        to an inconsistent number of samples. The output may have Pandas
        Indexes that do not match...

    Raised by ``ColumnTransformer``, naming neither the branch nor the step,
    minutes into a search. Configuring the object at construction means any
    factory output is safe to drop into a pipeline.
    """
    return transformer.set_output(transform="pandas")


IMPUTERS = ("median", "mean", "knn", "iterative")
DESKEWERS = ("none", "log", "yeo-johnson")
SCALERS = ("standard", "robust", "minmax", "none")


class DomainError(ValueError):
    """A transform was handed values it is not defined on.

    Its own class because the alternative is far worse than an exception:
    ``np.log1p(-124.3)`` returns **NaN**, silently, and the failure surfaces
    three steps later as a scaler producing all-NaN statistics with no mention
    of the column that caused it.
    """


def _checked_log1p(X):
    """``log1p`` that refuses its own domain error instead of emitting NaN.

    Found by a test: routing the whole numeric frame through ``deskew="log"``
    produced silent NaNs, because ``longitude`` is about −124 and ``log1p`` is
    undefined at or below −1. The pipeline did not fail — it produced a frame
    of NaNs and handed it to ``StandardScaler``, which then warned about
    dividing by an invalid value from inside scikit-learn's extmath. Nothing in
    that chain names ``longitude``.

    The cure is not to make the log cleverer. It is to say which column is
    wrong, and let the caller route it somewhere else — which is exactly what
    ``assemble`` does.
    """
    values = np.asarray(X, dtype=float)
    bad_mask = np.nan_to_num(values, nan=0.0) <= -1.0
    if bad_mask.any():
        if hasattr(X, "columns"):
            offenders = [
                f"{name} (min {float(np.nanmin(values[:, index])):.4g})"
                for index, name in enumerate(X.columns)
                if bad_mask[:, index].any()
            ]
        else:
            offenders = [f"column {index}" for index in np.where(bad_mask.any(axis=0))[0]]
        raise DomainError(
            "log1p is undefined at or below -1, and would return NaN for: "
            + "; ".join(offenders)
            + ". Route these columns through a block with deskew='none' or "
            "'yeo-johnson' instead."
        )
    return np.log1p(X)


def _log1p_transformer() -> FunctionTransformer:
    """``log1p`` as a transformer that keeps its column names.

    ``feature_names_out="one-to-one"`` is not optional: without it the
    transformer refuses to participate in ``get_feature_names_out``, and the
    whole pipeline loses the ability to name its own columns — which only
    surfaces much later, in M3-S4's importance plot.

    ``inverse_func`` is supplied so the step is genuinely invertible; nothing
    in this project inverts it today, but a transformer that silently cannot be
    undone is a trap for whoever first wants to.
    """
    return FunctionTransformer(
        func=_checked_log1p,
        inverse_func=np.expm1,
        feature_names_out="one-to-one",
        check_inverse=False,
        validate=False,
    )


def make_imputer(kind: str = "median", *, seed: int = config.RANDOM_SEED) -> BaseEstimator:
    """Return the named imputer.

    ``median`` is the default because the columns carrying the nulls are the
    skewed ones, and a mean pulled by a long right tail imputes a value the
    distribution barely contains.
    """
    if kind == "median":
        return _pandas(SimpleImputer(strategy="median"))
    if kind == "mean":
        return _pandas(SimpleImputer(strategy="mean"))
    if kind == "knn":
        # Distance-based, so it is only sane downstream of a scaler. In this
        # pipeline it sits BEFORE scaling, which means the largest-magnitude
        # column dominates the neighbour search. That is a real caveat and the
        # reason `knn` is not the default; see the note in the class docstring
        # of tests/test_numeric.py::test_knn_imputer_is_scale_sensitive.
        return _pandas(KNNImputer(n_neighbors=5, weights="distance"))
    if kind == "iterative":
        return _pandas(
            IterativeImputer(random_state=seed, max_iter=10, sample_posterior=False)
        )
    raise ValueError(f"Unknown imputer {kind!r}; expected one of {IMPUTERS}")


def make_deskew(kind: str = "none") -> BaseEstimator | str:
    """Return the named de-skew step, or ``"passthrough"``.

    ``"passthrough"`` rather than ``None`` so the step KEEPS ITS NAME in the
    pipeline. A named-but-inert step can be switched on by a search
    (``...__deskew`` stays addressable); a removed one cannot.
    """
    if kind == "none":
        return "passthrough"
    if kind == "log":
        return _pandas(_log1p_transformer())
    if kind == "yeo-johnson":
        # Yeo-Johnson rather than Box-Cox: Box-Cox requires strictly positive
        # input, and nothing here guarantees that for a column the caller might
        # route through later.
        return _pandas(PowerTransformer(method="yeo-johnson", standardize=False))
    raise ValueError(f"Unknown deskew {kind!r}; expected one of {DESKEWERS}")


def make_scaler(kind: str = "standard") -> BaseEstimator | str:
    """Return the named scaler, or ``"passthrough"``."""
    if kind == "standard":
        return _pandas(StandardScaler())
    if kind == "robust":
        return _pandas(RobustScaler())
    if kind == "minmax":
        return _pandas(MinMaxScaler())
    if kind == "none":
        return "passthrough"
    raise ValueError(f"Unknown scaler {kind!r}; expected one of {SCALERS}")


def build_numeric_block(
    *,
    imputer: str = "median",
    deskew: str = "none",
    scaler: str = "standard",
    seed: int = config.RANDOM_SEED,
) -> Pipeline:
    """Assemble the numeric block.

    The three step names — ``impute``, ``deskew``, ``scale`` — are part of the
    public interface: M3-S2's search addresses them by name
    (``preprocess__numeric__impute``), so renaming one silently empties a
    search space rather than raising.
    """
    block = Pipeline(
        steps=[
            ("impute", make_imputer(imputer, seed=seed)),
            ("deskew", make_deskew(deskew)),
            ("scale", make_scaler(scaler)),
        ]
    )
    # pandas in, pandas out - all the way through. Without this the block
    # returns a bare ndarray and every downstream feature name is positional.
    return block.set_output(transform="pandas")
