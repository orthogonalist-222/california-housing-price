"""Paths, column groups and constants — the facts every other module agrees on.

One source of truth per fact. A column list duplicated in three modules is three
places to forget when a column is renamed, and the failure surfaces as a
``KeyError`` three call frames away from the edit that caused it.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Reproducibility seed. One number, used everywhere a split or a search draws.
RANDOM_SEED = 42

#: What we predict.
TARGET = "median_house_value"

#: The raw measurements, as they arrive in the CSV.
NUMERIC_COLUMNS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

#: The one genuine string column.
CATEGORICAL_COLUMNS = ["ocean_proximity"]

#: Columns whose right tail is long enough to matter (measured skew 3.4–4.9 on
#: the raw file). The numeric block offers a de-skew step for exactly these.
HEAVY_TAILED_COLUMNS = ["total_rooms", "total_bedrooms", "population", "households"]

#: Counts that appear as denominators in the engineered ratios.
DENOMINATOR_COLUMNS = ["households", "total_rooms"]

#: ``ocean_proximity`` in increasing distance from water.
#:
#: This ordering is what makes an OrdinalEncoder defensible here rather than
#: merely convenient: the levels are not arbitrary names, they are a crude
#: distance scale, and imposing 0 < 1 < 2 < 3 < 4 on them asserts something
#: true. ISLAND sits at 0 because an island is surrounded by water; INLAND at
#: the far end. See docs/adr/ADR-002-encoding-strategy.md.
OCEAN_PROXIMITY_ORDER = ["ISLAND", "NEAR BAY", "NEAR OCEAN", "<1H OCEAN", "INLAND"]

#: The target is censored at BOTH ends in the source data: 965 rows sit at the
#: cap and 4 at the floor. Recorded as numbers rather than discovered again by
#: every reader. See docs/adr/ADR-001-dataset-and-target-censoring.md.
TARGET_CAP = 500_001.0
TARGET_FLOOR = 14_999.0

#: Income bands used to stratify the split. Géron's cut points: median_income is
#: the single strongest predictor here, so a split that distorts its
#: distribution distorts everything measured on top of it.
INCOME_BAND_EDGES = [0.0, 1.5, 3.0, 4.5, 6.0, float("inf")]
INCOME_BAND_LABELS = [1, 2, 3, 4, 5]

#: Plausible ranges for the range check in ``data.validate``. Generous on
#: purpose: these are asserted to catch a wrong file or a mangled parse, not to
#: quietly filter real rows. California spans roughly 32.5–42.0 N and
#: 124.4–114.1 W.
VALUE_RANGES: dict[str, tuple[float, float]] = {
    "longitude": (-125.0, -113.0),
    "latitude": (32.0, 43.0),
    "housing_median_age": (1.0, 100.0),
    "total_rooms": (1.0, 100_000.0),
    "total_bedrooms": (1.0, 100_000.0),
    "population": (1.0, 100_000.0),
    "households": (1.0, 100_000.0),
    "median_income": (0.0, 20.0),
    TARGET: (0.0, 600_000.0),
}

#: Where the raw CSV is looked for, in order. The Kaggle path comes first
#: because the kernel is the environment we cannot edit; the local path is the
#: development default.
_CSV_CANDIDATES = [
    Path("/kaggle/input/california-housing-prices/housing.csv"),
    Path("data/raw/housing.csv"),
]

#: Override, for a checkout that keeps its data elsewhere.
CSV_ENV_VAR = "CALHOUSING_CSV"


class DataNotFoundError(FileNotFoundError):
    """Raised when no candidate path holds the raw CSV.

    Its own class so a missing *file* is never confused with a malformed one -
    the cures are completely different (download it vs. investigate it).
    """


def resolve_csv(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Return the raw CSV's path, refusing rather than guessing.

    Resolution order: ``explicit`` argument, then ``$CALHOUSING_CSV``, then the
    Kaggle input mount, then ``data/raw/housing.csv``.

    A tool that silently picks a file is the one that reports a healthy run on
    the wrong data, so the chosen path is always printed by the caller
    (``data.load_raw``) and a miss raises with every location that was tried.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise DataNotFoundError(f"No such file: {path}")
        return path

    tried: list[Path] = []
    env = os.environ.get(CSV_ENV_VAR)
    if env:
        path = Path(env)
        if path.is_file():
            return path
        tried.append(path)

    for candidate in _CSV_CANDIDATES:
        if candidate.is_file():
            return candidate
        tried.append(candidate)

    locations = "\n".join(f"    {p}" for p in tried)
    raise DataNotFoundError(
        "Raw CSV not found. Tried, in order:\n"
        f"{locations}\n"
        "Fetch it with:\n"
        "    kaggle datasets download -d camnugent/california-housing-prices "
        "-p data/raw --unzip\n"
        f"or point {CSV_ENV_VAR} at an existing copy."
    )
