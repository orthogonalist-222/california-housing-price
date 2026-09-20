"""Numeric block invariants.

What is asserted here is what must stay true when the block is reconfigured:
nulls leave, shape is preserved, names survive, and — the one that matters —
**the transform of a test row does not depend on the other test rows.** That
last property is leakage, stated as something a test can check on a single
block, long before there is a pipeline or a model to hide it in.

No test pins a mean, a median or a scaled value. Those are era-facts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calhousing import config
from calhousing.preprocess.numeric import (
    DESKEWERS,
    IMPUTERS,
    SCALERS,
    DomainError,
    build_numeric_block,
    make_deskew,
    make_imputer,
    make_scaler,
)

NUMERIC = config.NUMERIC_COLUMNS


@pytest.fixture
def numeric_frame(housing: pd.DataFrame) -> pd.DataFrame:
    return housing[NUMERIC]


# --- the factories -----------------------------------------------------------


@pytest.mark.parametrize("kind", IMPUTERS)
def test_every_named_imputer_builds(kind: str) -> None:
    assert make_imputer(kind) is not None


@pytest.mark.parametrize("kind", DESKEWERS)
def test_every_named_deskew_builds(kind: str) -> None:
    assert make_deskew(kind) is not None


@pytest.mark.parametrize("kind", SCALERS)
def test_every_named_scaler_builds(kind: str) -> None:
    assert make_scaler(kind) is not None


@pytest.mark.parametrize(
    "factory, bad",
    [(make_imputer, "bayesian"), (make_deskew, "boxcox"), (make_scaler, "quantile")],
)
def test_unknown_name_is_refused_with_the_valid_options(factory, bad: str) -> None:
    """A typo must name the alternatives, not raise a bare KeyError.

    ``boxcox`` and ``quantile`` are real scikit-learn things that this project
    happens not to offer — exactly the kind of near-miss that deserves a list.
    """
    with pytest.raises(ValueError, match="expected one of"):
        factory(bad)


def test_disabled_steps_keep_their_names() -> None:
    """``passthrough``, not ``None``.

    A named-but-inert step stays addressable by a search
    (``...__deskew``); a removed one does not, and the search space would
    silently target nothing.
    """
    block = build_numeric_block(deskew="none", scaler="none")
    assert [name for name, _ in block.steps] == ["impute", "deskew", "scale"]
    assert block.named_steps["deskew"] == "passthrough"
    assert block.named_steps["scale"] == "passthrough"


# --- the invariants ----------------------------------------------------------


@pytest.mark.parametrize("imputer", IMPUTERS)
def test_no_nulls_survive_any_imputer(imputer: str, numeric_frame: pd.DataFrame) -> None:
    assert numeric_frame.isna().any().any(), "fixture must actually contain nulls"
    out = build_numeric_block(imputer=imputer).fit_transform(numeric_frame)
    assert not np.isnan(np.asarray(out, dtype=float)).any()


@pytest.mark.parametrize("deskew", DESKEWERS)
@pytest.mark.parametrize("scaler", SCALERS)
def test_shape_and_names_are_preserved(
    deskew: str, scaler: str, numeric_frame: pd.DataFrame
) -> None:
    """The block is one-to-one: it never adds or drops a column."""
    # `log` is only defined above -1, so it sees the positive count columns -
    # which is exactly how `assemble` routes them. Every other configuration
    # takes the whole numeric frame, longitude included.
    frame = (
        numeric_frame[config.HEAVY_TAILED_COLUMNS]
        if deskew == "log"
        else numeric_frame
    )
    block = build_numeric_block(deskew=deskew, scaler=scaler)
    out = block.fit_transform(frame)
    assert isinstance(out, pd.DataFrame), "set_output(pandas) must survive every config"
    assert out.shape == frame.shape
    assert list(out.columns) == list(frame.columns)


def test_feature_names_out_is_available(numeric_frame: pd.DataFrame) -> None:
    """A model whose importances read ``x17`` cannot be interrogated."""
    heavy = numeric_frame[config.HEAVY_TAILED_COLUMNS]
    block = build_numeric_block(deskew="log").fit(heavy)
    assert list(block.get_feature_names_out()) == list(heavy.columns)


def test_transform_of_a_row_does_not_depend_on_the_other_rows(
    numeric_frame: pd.DataFrame,
) -> None:
    """**Leakage, as a property of one block.**

    Fit once. Then transform the whole test frame, and transform each of a few
    rows *alone*. If the two disagree, the transform is reading statistics from
    the data it is scoring — which is exactly what happens when an imputer or a
    scaler is fitted outside the fold.
    """
    train = numeric_frame.iloc[:400]
    test = numeric_frame.iloc[400:]
    block = build_numeric_block(imputer="median", scaler="standard").fit(train)

    together = block.transform(test)
    for position in (0, 5, len(test) - 1):
        alone = block.transform(test.iloc[[position]])
        np.testing.assert_allclose(
            alone.to_numpy(), together.iloc[[position]].to_numpy(), rtol=1e-12, atol=1e-12
        )


def test_fitted_statistics_come_from_train_only(numeric_frame: pd.DataFrame) -> None:
    """Red-team the same property from the other side.

    Fit on train, then fit on train with wildly different test rows appended.
    The imputer's learned fill values must be identical in the first case and
    must move in the second — if appending unseen rows changed nothing, the
    test would be vacuous.
    """
    train = numeric_frame.iloc[:400]
    poisoned = pd.concat([train, numeric_frame.iloc[400:] * 1000.0])

    honest = build_numeric_block().fit(train).named_steps["impute"].statistics_
    again = build_numeric_block().fit(train).named_steps["impute"].statistics_
    leaked = build_numeric_block().fit(poisoned).named_steps["impute"].statistics_

    np.testing.assert_allclose(honest, again)
    assert not np.allclose(honest, leaked), (
        "appending very different rows did not move the imputer's statistics; "
        "this test cannot detect leakage as written"
    )


def test_log_on_an_out_of_domain_column_is_refused_by_name(
    numeric_frame: pd.DataFrame,
) -> None:
    """Red-team: the defect this story actually found.

    ``longitude`` is about -124, and ``np.log1p`` returns **NaN** there without
    complaint. Routed through the whole numeric frame, the block produced a
    frame of NaNs and handed it to ``StandardScaler``, which warned about an
    invalid divide from inside scikit-learn's extmath - a message that never
    mentions ``longitude``.

    The refusal must name the column and say where to send it instead.
    """
    with pytest.raises(DomainError) as exc:
        build_numeric_block(deskew="log").fit_transform(numeric_frame)
    message = str(exc.value)
    assert "longitude" in message
    assert "yeo-johnson" in message


def test_the_guard_does_not_fire_on_valid_input(numeric_frame: pd.DataFrame) -> None:
    """The other half: a guard that refuses correct input is worse than none."""
    out = build_numeric_block(deskew="log", scaler="none").fit_transform(
        numeric_frame[config.HEAVY_TAILED_COLUMNS]
    )
    assert not out.isna().any().any()


def test_nulls_do_not_trip_the_domain_guard(numeric_frame: pd.DataFrame) -> None:
    """NaN must not be read as 'below -1'.

    The guard runs downstream of the imputer, so in practice it never sees a
    null - but a future reordering would silently turn every missing value into
    a domain error, and that is a failure mode worth pinning now.
    """
    from calhousing.preprocess.numeric import _checked_log1p

    frame = numeric_frame[config.HEAVY_TAILED_COLUMNS].copy()
    assert frame.isna().any().any()
    _checked_log1p(frame)


def test_log_deskew_actually_reduces_skew(numeric_frame: pd.DataFrame) -> None:
    """The option exists because M1-S3 measured skew of 3.4–4.9.

    Asserted as a direction, not a value: the transformed columns must be less
    skewed than the raw ones. A specific number here would be an era-fact.
    """
    heavy = numeric_frame[config.HEAVY_TAILED_COLUMNS]
    raw_skew = heavy.skew().abs()
    out = build_numeric_block(deskew="log", scaler="none").fit_transform(heavy)
    assert (out.skew().abs() < raw_skew).all()


def test_yeo_johnson_also_reduces_skew(numeric_frame: pd.DataFrame) -> None:
    heavy = numeric_frame[config.HEAVY_TAILED_COLUMNS]
    raw_skew = heavy.skew().abs()
    out = build_numeric_block(deskew="yeo-johnson", scaler="none").fit_transform(heavy)
    assert (out.skew().abs() < raw_skew).all()


def test_standard_scaler_centres_the_training_data(numeric_frame: pd.DataFrame) -> None:
    """A property of the scaler, not a stored constant."""
    out = build_numeric_block(scaler="standard").fit_transform(numeric_frame)
    np.testing.assert_allclose(out.mean().to_numpy(), 0.0, atol=1e-10)
    np.testing.assert_allclose(out.std(ddof=0).to_numpy(), 1.0, atol=1e-10)


def test_minmax_bounds_the_training_data(numeric_frame: pd.DataFrame) -> None:
    out = build_numeric_block(scaler="minmax").fit_transform(numeric_frame)
    assert out.to_numpy().min() >= -1e-12
    assert out.to_numpy().max() <= 1 + 1e-12


def test_knn_imputer_is_scale_sensitive(numeric_frame: pd.DataFrame) -> None:
    """Documents a real caveat rather than hiding it.

    ``KNNImputer`` sits *before* the scaler in this block, so its neighbour
    search is dominated by whichever column has the largest magnitude. Blowing
    up one column's scale must therefore change what it imputes. This is why
    ``knn`` is not the default — and asserting it here means the caveat cannot
    quietly stop being true without someone noticing.
    """
    frame = numeric_frame.copy()
    baseline = build_numeric_block(imputer="knn", scaler="none").fit_transform(frame)

    rescaled = frame.copy()
    rescaled["population"] = rescaled["population"] * 1e6
    shifted = build_numeric_block(imputer="knn", scaler="none").fit_transform(rescaled)

    missing = frame["total_bedrooms"].isna().to_numpy()
    assert missing.any()
    assert not np.allclose(
        baseline.loc[missing, "total_bedrooms"].to_numpy(),
        shifted.loc[missing, "total_bedrooms"].to_numpy(),
    )


def test_iterative_imputer_is_reproducible(numeric_frame: pd.DataFrame) -> None:
    """It draws randomly; the seed must actually pin it."""
    a = build_numeric_block(imputer="iterative").fit_transform(numeric_frame)
    b = build_numeric_block(imputer="iterative").fit_transform(numeric_frame)
    pd.testing.assert_frame_equal(a, b)


def test_block_is_clonable_and_searchable() -> None:
    """M3-S2 sets these by path; a rename would silently empty the search."""
    from sklearn.base import clone

    block = build_numeric_block()
    clone(block)
    block.set_params(impute=make_imputer("knn"), deskew=make_deskew("log"))
    assert block.named_steps["impute"].__class__.__name__ == "KNNImputer"
