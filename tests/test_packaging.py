"""Kaggle packaging — the metadata, the staging tool, and the fallback's wiring.

The two install paths themselves are verified by actually installing into clean
environments (ADR-004 records both runs). What is tested here is everything
that would make those verifications meaningless later: metadata that stops
naming the dataset the fallback reads, a staging tool that uploads on its own,
an install cell that lost its second branch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import stage_kaggle

REPO = Path(__file__).resolve().parents[1]
DATASET_META = REPO / "kaggle" / "src_dataset" / "dataset-metadata.json"
KERNEL_META = REPO / "kaggle" / "kernel" / "kernel-metadata.json"
NOTEBOOK = REPO / "notebooks" / "california-housing-sklearn-pipeline.ipynb"


def _install_cell() -> str:
    import build_notebook

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(
        c
        for c in notebook["cells"]
        if c["cell_type"] == "code" and build_notebook.INSTALL_TAG in "".join(c["source"])
    )
    return "".join(cell["source"])


# --- metadata ----------------------------------------------------------------


def test_metadata_files_are_valid_json() -> None:
    for path in (DATASET_META, KERNEL_META):
        json.loads(path.read_text(encoding="utf-8"))


def test_the_kernel_attaches_the_dataset_the_fallback_reads() -> None:
    """**The wiring that makes the fallback a mechanism rather than a comment.**

    If the kernel stops attaching `calhousing-src`, the offline branch searches
    a directory that does not exist, finds no wheel, and the notebook dies -
    with a message that at least says so, but only after failing.
    """
    dataset = json.loads(DATASET_META.read_text(encoding="utf-8"))
    kernel = json.loads(KERNEL_META.read_text(encoding="utf-8"))
    assert dataset["id"] in kernel["dataset_sources"]


def test_the_kernel_attaches_the_raw_data() -> None:
    kernel = json.loads(KERNEL_META.read_text(encoding="utf-8"))
    assert "camnugent/california-housing-prices" in kernel["dataset_sources"]


def test_internet_is_enabled_for_the_primary_path() -> None:
    """ADR-004: GitHub first. That needs the toggle on, and the toggle lives in
    this file - not in a setting somebody remembers to flick."""
    kernel = json.loads(KERNEL_META.read_text(encoding="utf-8"))
    assert kernel["enable_internet"] == "true"


def test_the_kernel_points_at_the_generated_notebook() -> None:
    kernel = json.loads(KERNEL_META.read_text(encoding="utf-8"))
    assert kernel["code_file"] == NOTEBOOK.name


def test_no_gpu_is_requested() -> None:
    """Nothing here needs one, and an idle GPU kernel is a slower queue for
    everybody."""
    kernel = json.loads(KERNEL_META.read_text(encoding="utf-8"))
    assert kernel["enable_gpu"] == "false"
    assert kernel["enable_tpu"] == "false"


# --- the install cell --------------------------------------------------------


def test_the_install_cell_has_both_paths() -> None:
    body = _install_cell()
    assert "git+https://github.com/orthogonalist-222/california-housing-price" in body
    assert "/kaggle/input/calhousing-src" in body
    assert "--no-index" in body


def test_the_fallback_directory_matches_the_dataset_id() -> None:
    """The mount path Kaggle creates is `/kaggle/input/<dataset-slug>`.

    A renamed dataset with an unchanged notebook is a fallback that silently
    looks in the wrong place.
    """
    dataset = json.loads(DATASET_META.read_text(encoding="utf-8"))
    slug = dataset["id"].split("/")[-1]
    assert f"/kaggle/input/{slug}" in _install_cell()


def test_the_notebook_says_which_path_it_used() -> None:
    """Two runs of the same notebook can be running different code. A reader
    debugging a stale result needs to know which."""
    body = _install_cell()
    assert "installed from" in body
    assert "source" in body


def test_the_install_cell_fails_loudly_when_neither_path_works() -> None:
    body = _install_cell()
    assert "raise SystemExit" in body
    assert "Internet" in body


def test_no_credentials_anywhere_in_the_notebook() -> None:
    """A token in a public notebook is a token that is now public."""
    body = NOTEBOOK.read_text(encoding="utf-8")
    for smell in ("kaggle.json", "KAGGLE_KEY", "ghp_", "github_pat_", "Authorization"):
        assert smell not in body, smell


# --- the staging tool --------------------------------------------------------


def test_staging_refuses_when_the_kernel_does_not_attach_the_dataset(
    monkeypatch, tmp_path: Path
) -> None:
    """Red-team: break the wiring, watch the tool refuse.

    Without this the tool happily builds a wheel for a dataset no kernel reads.
    """
    dataset_dir = tmp_path / "src_dataset"
    kernel_dir = tmp_path / "kernel"
    dataset_dir.mkdir()
    kernel_dir.mkdir()
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps({"title": "x", "id": "someone/calhousing-src"}), encoding="utf-8"
    )
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps({"id": "someone/k", "dataset_sources": []}), encoding="utf-8"
    )
    monkeypatch.setattr(stage_kaggle, "DATASET_DIR", dataset_dir)
    monkeypatch.setattr(stage_kaggle, "KERNEL_DIR", kernel_dir)
    monkeypatch.setattr(stage_kaggle, "NOTEBOOK", NOTEBOOK)

    with pytest.raises(SystemExit, match="does not attach"):
        stage_kaggle.main(["--skip-wheel"])


def test_staging_does_not_upload(monkeypatch, tmp_path: Path) -> None:
    """**The tool prints commands; it never runs them.**

    Publishing to a live Kaggle account is outward-facing. A tool that does it
    as a side effect of "staging" will one day publish something nobody meant
    to, so the absence of an upload is asserted rather than assumed.
    """
    calls: list[list[str]] = []

    def spy(args, *rest, **kwargs):  # pragma: no cover - the point is it is unused
        calls.append(list(args))
        raise AssertionError(f"stage_kaggle tried to run: {args}")

    monkeypatch.setattr(stage_kaggle.subprocess, "run", spy)
    monkeypatch.setattr(stage_kaggle, "KERNEL_DIR", tmp_path)
    (tmp_path / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "duonghongphu/k",
                "dataset_sources": ["duonghongphu/calhousing-src"],
            }
        ),
        encoding="utf-8",
    )
    assert stage_kaggle.main(["--skip-wheel"]) == 0
    assert calls == []


def test_staging_refuses_without_a_built_notebook(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(stage_kaggle, "NOTEBOOK", tmp_path / "absent.ipynb")
    with pytest.raises(SystemExit, match="does not exist"):
        stage_kaggle.stage_kernel()


def test_staged_copies_are_not_tracked() -> None:
    """The wheel is a build output and the notebook is committed once, at its
    source. A second tracked copy is a second thing to keep in sync."""
    ignored = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert "kaggle/src_dataset/*.whl" in ignored
    assert "kaggle/kernel/*.ipynb" in ignored
