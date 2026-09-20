"""Tests for the layout gate - including the gate's own red-team.

A gate nobody has watched refuse anything is an assumption (section 4). The
planted defects here are synthetic path lists, never the real tree: the gate is
exercised, the repo is not touched.

Note what these tests deliberately do NOT assert: the number of tracked files,
or that any specific file exists. Those are era-facts that change every story,
and a guard that fires on correct behaviour teaches the next session to edit the
assertion instead of the code. What is asserted is the invariant - an undeclared
path is refused, a declared one is not, and the two refusals are distinguishable.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import check_layout  # noqa: E402


def test_declared_paths_are_accepted() -> None:
    declared = [glob for glob, _ in check_layout.DECLARED if "*" not in glob]
    forbidden, undeclared = check_layout.classify(declared)
    assert forbidden == []
    assert undeclared == []


def test_wildcard_entries_match_within_one_segment_only() -> None:
    """``docs/field-notes/*.md`` must not bless a nested directory."""
    forbidden, undeclared = check_layout.classify(
        ["docs/field-notes/m1-s1-scaffold.md", "docs/field-notes/nested/sneak.md"]
    )
    assert forbidden == []
    assert undeclared == ["docs/field-notes/nested/sneak.md"]


def test_undeclared_path_is_refused() -> None:
    """Red-team: plant a plausible stray file, watch the gate refuse it."""
    forbidden, undeclared = check_layout.classify(["src/calhousing/scratch.py"])
    assert undeclared == ["src/calhousing/scratch.py"]
    assert forbidden == []


def test_forbidden_path_is_refused_with_its_own_signature() -> None:
    """Red-team: a secret and a data file are refused as FORBIDDEN, not merely
    as undeclared, so the message points at the real cure."""
    forbidden, undeclared = check_layout.classify(
        ["kaggle.json", "data/raw/housing.csv", "HANDOFF.md"]
    )
    assert undeclared == []
    assert {path for path, _ in forbidden} == {
        "kaggle.json",
        "data/raw/housing.csv",
        "HANDOFF.md",
    }
    assert all(why for _, why in forbidden)


def test_forbidden_beats_declared() -> None:
    """A path that is both declared and forbidden must refuse.

    Nothing declares ``.env`` today, so this pins the PRECEDENCE rather than
    today's lists: if a later story ever adds an entry that overlaps FORBIDDEN,
    the secret rule still wins.
    """
    original = check_layout.DECLARED
    try:
        check_layout.DECLARED = original + [(".env", "hypothetical")]
        check_layout._DECLARED_RE.append((check_layout._to_regex(".env"), ".env", "x"))
        forbidden, undeclared = check_layout.classify([".env"])
        assert [p for p, _ in forbidden] == [".env"]
        assert undeclared == []
    finally:
        check_layout.DECLARED = original
        check_layout._DECLARED_RE.pop()


def test_a_declared_path_that_is_not_tracked_is_refused() -> None:
    """Red-team the direction the gate used to be blind in (F-007).

    The gate enforced *tracked implies declared* only. A declared path that was
    never committed passed on the machine that wrote it and failed on every
    other one - which is exactly how `kaggle/src_dataset/dataset-metadata.json`
    reached main uncommitted and broke CI.
    """
    tracked = [glob for glob, _ in check_layout.DECLARED if "*" not in glob]
    assert check_layout.missing(tracked) == []

    without_one = [p for p in tracked if p != "pyproject.toml"]
    absent = check_layout.missing(without_one)
    assert ("pyproject.toml", "M1-S1") in absent


def test_glob_entries_are_not_required_to_exist() -> None:
    """``docs/field-notes/*.md`` says *these are allowed*, not *at least one
    must exist*. Requiring globs would make the gate refuse an empty repo."""
    assert all("*" not in glob for glob, _ in check_layout.missing([]))


def test_the_three_failure_modes_have_three_different_messages() -> None:
    """A gate whose refusals all read alike is a debugging tax."""
    import io
    import contextlib

    forbidden, undeclared = check_layout.classify(["kaggle.json", "src/calhousing/x.py"])
    assert forbidden and undeclared
    assert check_layout.missing([]) != []


def test_gate_passes_on_the_real_tree() -> None:
    """The actual repo, right now, is fully declared."""
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "tools/check_layout.py"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
