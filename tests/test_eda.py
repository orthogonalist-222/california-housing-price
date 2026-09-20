"""The EDA build is a recipe, so it has to be reproducible.

Figures are cattle: deleted and rebuilt at will. That is only true if the
rebuild produces the same thing, which is not automatic — matplotlib writes its
own version into PNG metadata unless told not to, so two builds of an identical
figure differ in bytes. This module is the guard on that.

All of it runs on synthetic data. CI has no ``data/raw/housing.csv``, and a
determinism check that only runs on one machine is not a check.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from calhousing import eda


def _digests(paths: list[Path]) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_every_figure_is_produced(housing: pd.DataFrame, tmp_path: Path) -> None:
    paths = eda.build_all(housing, tmp_path, verbose=False)
    assert len(paths) == len(eda._FIGURES)
    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 0


def test_figure_names_are_ordered_and_unique(
    housing: pd.DataFrame, tmp_path: Path
) -> None:
    """The numeric prefix is what makes a directory listing match the notebook."""
    names = [p.name for p in eda.build_all(housing, tmp_path, verbose=False)]
    assert len(set(names)) == len(names)
    assert names == sorted(names)


def test_rebuild_is_byte_identical(housing: pd.DataFrame, tmp_path: Path) -> None:
    """THE invariant. A recipe that cannot reproduce its own output cannot be
    held to a staleness gate, and every later build step depends on that."""
    first = _digests(eda.build_all(housing, tmp_path / "a", verbose=False))
    second = _digests(eda.build_all(housing, tmp_path / "b", verbose=False))
    assert first == second


def test_matplotlib_version_is_not_baked_into_the_png(
    housing: pd.DataFrame, tmp_path: Path
) -> None:
    """Red-team the determinism, by showing what breaks it.

    Saving the same figure the default way embeds a ``Software`` chunk naming
    the matplotlib version. That is precisely the byte that would make a
    rebuild differ after an unrelated dependency upgrade, so the gate above
    would fail for a reason that has nothing to do with the data.
    """
    paths = eda.build_all(housing, tmp_path, verbose=False)
    blob = paths[0].read_bytes()
    assert b"matplotlib" not in blob.lower()


def test_build_all_accepts_a_string_path(housing: pd.DataFrame, tmp_path: Path) -> None:
    paths = eda.build_all(housing, str(tmp_path / "as-string"), verbose=False)
    assert all(p.is_file() for p in paths)


def test_cli_builds_into_the_requested_directory(
    housing: pd.DataFrame, tmp_path: Path
) -> None:
    csv = tmp_path / "housing.csv"
    housing.to_csv(csv, index=False)
    out = tmp_path / "cli-out"
    assert eda.main(["--csv", str(csv), "--out", str(out)]) == 0
    assert len(list(out.glob("*.png"))) == len(eda._FIGURES)


def test_figures_survive_a_frame_with_no_rare_level(tmp_path: Path) -> None:
    """A figure that only works on the real file is a figure that will break.

    ``fig_rare_category`` colours a level red below a threshold and
    ``fig_income_and_split`` builds a coverage table; neither may assume the
    island exists.
    """
    from conftest import make_housing

    frame = make_housing(n=400, n_island=0, seed=9)
    assert "ISLAND" not in set(frame["ocean_proximity"])
    paths = eda.build_all(frame, tmp_path, verbose=False)
    assert len(paths) == len(eda._FIGURES)


def test_real_figures_are_deterministic(real_housing: pd.DataFrame, tmp_path: Path) -> None:
    """Skipped on CI. Locally, the same check against the actual dataset."""
    first = _digests(eda.build_all(real_housing, tmp_path / "a", verbose=False))
    second = _digests(eda.build_all(real_housing, tmp_path / "b", verbose=False))
    assert first == second
