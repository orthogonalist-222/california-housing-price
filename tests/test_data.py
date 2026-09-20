"""The schema contract, and the red-team that proves it refuses.

Each refusal gets its own test with a planted defect, because the contract's
value is entirely in saying *which* thing is wrong. A contract that raised one
generic error for all three cases would pass a test that only checked "it
raised".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calhousing import config
from calhousing.data import (
    ColumnTypeError,
    MissingColumnsError,
    SchemaError,
    ValueRangeError,
    expected_columns,
    load_raw,
    validate,
)


def test_valid_frame_passes_unchanged(housing: pd.DataFrame) -> None:
    out = validate(housing)
    assert out is housing


def test_nulls_are_not_a_schema_violation(housing: pd.DataFrame) -> None:
    """Missing ``total_bedrooms`` is the imputer's job, not the contract's.

    This is the invariant that keeps the two concerns apart: if the range check
    ever started treating NaN as out-of-range, every real file would be refused.
    """
    assert housing["total_bedrooms"].isna().any()
    validate(housing)


# --- red-team: three planted defects, three distinguishable refusals ---------


def test_missing_column_is_refused(housing: pd.DataFrame) -> None:
    planted = housing.drop(columns=["median_income"])
    with pytest.raises(MissingColumnsError) as exc:
        validate(planted)
    assert "median_income" in str(exc.value)


def test_wrong_dtype_is_refused(housing: pd.DataFrame) -> None:
    """A numeric column arriving as text - a stray header row, a thousands
    separator - is the classic silent parse failure."""
    planted = housing.copy()
    planted["total_rooms"] = planted["total_rooms"].astype(str)
    with pytest.raises(ColumnTypeError) as exc:
        validate(planted)
    message = str(exc.value)
    assert "total_rooms" in message and "expected numeric" in message


def test_categorical_arriving_as_numeric_is_refused(housing: pd.DataFrame) -> None:
    planted = housing.copy()
    planted["ocean_proximity"] = 1
    with pytest.raises(ColumnTypeError):
        validate(planted)


def test_out_of_range_value_is_refused(housing: pd.DataFrame) -> None:
    """Latitude in Oregon: the frame has this dataset's shape, not its content."""
    planted = housing.copy()
    planted.loc[planted.index[0], "latitude"] = 45.6
    with pytest.raises(ValueRangeError) as exc:
        validate(planted)
    message = str(exc.value)
    assert "latitude" in message and "45.6" in message


def test_the_three_refusals_are_distinguishable(housing: pd.DataFrame) -> None:
    """All three are ``SchemaError``, none is another's class.

    Catching the base class must work (callers want one ``except``), while
    ``except MissingColumnsError`` must not swallow a range problem.
    """
    missing = housing.drop(columns=["households"])
    typed = housing.assign(households=housing["households"].astype(str))
    ranged = housing.copy()
    ranged.loc[ranged.index[0], "median_income"] = 99.0

    for planted, expected in [
        (missing, MissingColumnsError),
        (typed, ColumnTypeError),
        (ranged, ValueRangeError),
    ]:
        with pytest.raises(SchemaError):
            validate(planted)
        with pytest.raises(expected):
            validate(planted)

    # ...and the classes do not overlap: a range problem must not be caught by
    # `except MissingColumnsError`.
    with pytest.raises(SchemaError) as exc:
        validate(ranged)
    assert not isinstance(exc.value, (MissingColumnsError, ColumnTypeError))


def test_checks_run_most_fundamental_first(housing: pd.DataFrame) -> None:
    """A frame broken three ways reports the *missing column*, not the range.

    Ordering matters: telling someone their latitudes are odd when the real
    problem is that they loaded the wrong file sends them down the wrong path.
    """
    planted = housing.drop(columns=["latitude"]).copy()
    planted["households"] = planted["households"].astype(str)
    with pytest.raises(MissingColumnsError):
        validate(planted)


def test_target_can_be_optional(housing: pd.DataFrame) -> None:
    features = housing.drop(columns=[config.TARGET])
    with pytest.raises(MissingColumnsError):
        validate(features)
    validate(features, require_target=False)


# --- path resolution ---------------------------------------------------------


def test_missing_file_raises_with_every_location_tried(tmp_path, monkeypatch) -> None:
    """The refusal must name what it tried and how to fix it.

    ``DataNotFoundError`` is deliberately not a ``SchemaError``: a file that is
    absent and a file that is malformed have nothing in common except that both
    stop the run.
    """
    monkeypatch.delenv(config.CSV_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "_CSV_CANDIDATES", [tmp_path / "nope.csv"])
    with pytest.raises(config.DataNotFoundError) as exc:
        config.resolve_csv()
    message = str(exc.value)
    assert "nope.csv" in message
    assert "kaggle datasets download" in message
    assert not isinstance(exc.value, SchemaError)


def test_a_kaggle_mount_is_searched_not_hardcoded(tmp_path, monkeypatch, housing) -> None:
    """Red-team F-010, which reached a published kernel.

    The loader used to hardcode
    ``/kaggle/input/california-housing-prices/housing.csv``. On the real kernel
    the file was not there, and the notebook died on its first data cell with a
    message that listed the paths it wanted and nothing about what was actually
    mounted.

    The mount directory is named for the dataset slug and depends on what is
    attached and how, so the file is found by NAME under whatever is there.
    """
    monkeypatch.delenv(config.CSV_ENV_VAR, raising=False)
    mount = tmp_path / "input"
    # A slug this project has never heard of - the point is that it still works.
    dataset = mount / "some-other-slug"
    dataset.mkdir(parents=True)
    housing.to_csv(dataset / config.CSV_NAME, index=False)

    monkeypatch.setattr(config, "KAGGLE_INPUT", mount)
    monkeypatch.setattr(config, "_CSV_CANDIDATES", [tmp_path / "nowhere.csv"])
    assert config.resolve_csv() == dataset / config.CSV_NAME


def test_a_nested_kaggle_mount_is_found(tmp_path, monkeypatch, housing) -> None:
    """Some datasets mount the file one directory deeper."""
    monkeypatch.delenv(config.CSV_ENV_VAR, raising=False)
    mount = tmp_path / "input"
    nested = mount / "a-dataset" / "sub"
    nested.mkdir(parents=True)
    housing.to_csv(nested / config.CSV_NAME, index=False)

    monkeypatch.setattr(config, "KAGGLE_INPUT", mount)
    monkeypatch.setattr(config, "_CSV_CANDIDATES", [tmp_path / "nowhere.csv"])
    assert config.resolve_csv() == nested / config.CSV_NAME


def test_the_kaggle_failure_says_what_is_mounted(tmp_path, monkeypatch) -> None:
    """The other half. A refusal that lists only what it WANTED leaves the
    reader guessing - which is what happened on the published kernel."""
    monkeypatch.delenv(config.CSV_ENV_VAR, raising=False)
    mount = tmp_path / "input"
    (mount / "something-else").mkdir(parents=True)

    monkeypatch.setattr(config, "KAGGLE_INPUT", mount)
    monkeypatch.setattr(config, "_CSV_CANDIDATES", [tmp_path / "nowhere.csv"])
    with pytest.raises(config.DataNotFoundError) as exc:
        config.resolve_csv()
    message = str(exc.value)
    assert "something-else" in message, "the refusal must name what IS mounted"
    assert "Attach the camnugent" in message


def test_no_kaggle_mount_gives_the_local_instructions(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(config.CSV_ENV_VAR, raising=False)
    monkeypatch.setattr(config, "KAGGLE_INPUT", tmp_path / "absent")
    monkeypatch.setattr(config, "_CSV_CANDIDATES", [tmp_path / "nowhere.csv"])
    with pytest.raises(config.DataNotFoundError, match="kaggle datasets download"):
        config.resolve_csv()


def test_explicit_path_wins(tmp_path, housing: pd.DataFrame) -> None:
    path = tmp_path / "elsewhere.csv"
    housing.to_csv(path, index=False)
    loaded = load_raw(path, verbose=False)
    assert list(loaded.columns) == expected_columns()


def test_env_var_is_honoured(tmp_path, housing: pd.DataFrame, monkeypatch) -> None:
    path = tmp_path / "from_env.csv"
    housing.to_csv(path, index=False)
    monkeypatch.setenv(config.CSV_ENV_VAR, str(path))
    assert config.resolve_csv() == path


# --- the real file, when it is here ------------------------------------------


def test_real_file_satisfies_the_contract(real_housing: pd.DataFrame) -> None:
    """Skipped on CI, where there is no data. Locally this is the check that
    matters: the contract must accept the actual dataset."""
    validate(real_housing)
    assert list(real_housing.columns) == expected_columns()
    # Structural invariants, not this month's numbers: nulls exist and are
    # confined to one column, and the rare level exists.
    nulls = real_housing.isna().sum()
    assert nulls[nulls > 0].index.tolist() == ["total_bedrooms"]
    assert "ISLAND" in set(real_housing["ocean_proximity"])
