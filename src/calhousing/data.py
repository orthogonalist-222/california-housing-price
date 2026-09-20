"""Load the raw CSV and hold it to a schema contract.

The contract exists because of a specific class of failure: a pipeline handed
the wrong file, or a file whose columns were renamed upstream, does not crash.
It trains, it scores, and the number it reports is meaningless. A loud refusal
at the door costs one function call and converts that into a build failure.

Three distinguishable refusals, because they have three different cures:

``MissingColumnsError``
    The file is not this dataset (or a rename happened upstream).
``ColumnTypeError``
    The file parsed, but a column came back as the wrong dtype - usually a
    stray header row or a thousands separator turning numbers into strings.
``ValueRangeError``
    The file is this dataset's *shape* but not its *content* - latitudes in
    Oregon, a negative population, a target outside any plausible band.

A failure that reads like every other failure is a debugging tax, so each
message names the columns at fault and what was expected.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from pandas.api import types as ptypes

from . import config

__all__ = [
    "SchemaError",
    "MissingColumnsError",
    "ColumnTypeError",
    "ValueRangeError",
    "load_raw",
    "validate",
    "expected_columns",
]


class SchemaError(ValueError):
    """Base class for every schema-contract refusal."""


class MissingColumnsError(SchemaError):
    """The frame does not carry the columns this dataset is defined by."""


class ColumnTypeError(SchemaError):
    """A column is present but holds the wrong kind of value."""


class ValueRangeError(SchemaError):
    """A column's values fall outside the range this dataset can contain."""


def expected_columns() -> list[str]:
    """The full column list, in the order the raw file supplies it."""
    return [*config.NUMERIC_COLUMNS, config.TARGET, *config.CATEGORICAL_COLUMNS]


def validate(frame: pd.DataFrame, *, require_target: bool = True) -> pd.DataFrame:
    """Return ``frame`` unchanged, or raise the most specific ``SchemaError``.

    Parameters
    ----------
    require_target:
        False when validating a feature-only frame (a prediction request has no
        ``median_house_value``). Everything else is checked identically.

    Checks run missing → dtype → range, most fundamental first: a range check on
    a column that is secretly a string produces a confusing TypeError instead of
    the clear message the caller needs.
    """
    expected = expected_columns()
    if not require_target:
        expected = [c for c in expected if c != config.TARGET]

    missing = [c for c in expected if c not in frame.columns]
    if missing:
        raise MissingColumnsError(
            f"Missing required column(s): {missing}. "
            f"Expected {expected}, got {list(frame.columns)}."
        )

    wrong_type: list[str] = []
    for column in config.NUMERIC_COLUMNS + ([config.TARGET] if require_target else []):
        if not ptypes.is_numeric_dtype(frame[column]):
            wrong_type.append(f"{column} (got {frame[column].dtype}, expected numeric)")
    for column in config.CATEGORICAL_COLUMNS:
        series = frame[column]
        if not (ptypes.is_string_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype)):
            wrong_type.append(f"{column} (got {series.dtype}, expected string/category)")
    if wrong_type:
        raise ColumnTypeError("Column(s) with the wrong dtype: " + "; ".join(wrong_type))

    out_of_range: list[str] = []
    for column, (low, high) in config.VALUE_RANGES.items():
        if column not in frame.columns:
            continue
        series = frame[column]
        # NaN compares False against every bound, so a null would read as
        # "in range". That is correct here: missing values are the imputer's
        # job, and this check must not double as a null check.
        bad = series.notna() & ((series < low) | (series > high))
        if bad.any():
            observed = (series[bad].min(), series[bad].max())
            out_of_range.append(
                f"{column}: {int(bad.sum())} row(s) outside [{low}, {high}] "
                f"(observed {observed[0]} .. {observed[1]})"
            )
    if out_of_range:
        raise ValueRangeError("Value(s) outside the expected range: " + "; ".join(out_of_range))

    return frame


def load_raw(
    path: str | os.PathLike[str] | None = None,
    *,
    validate_schema: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Read the raw housing CSV and validate it.

    The resolved path is printed by default. A tool that silently picks one of
    several candidate files is the one that reports a healthy run on the wrong
    data; saying which file was opened costs a line and removes the whole class
    of confusion.
    """
    resolved: Path = config.resolve_csv(path)
    frame = pd.read_csv(resolved)
    if verbose:
        print(f"[calhousing] read {resolved}  ->  {frame.shape[0]:,} rows x {frame.shape[1]} columns")
    if validate_schema:
        validate(frame)
    return frame
