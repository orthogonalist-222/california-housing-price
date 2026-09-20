"""The notebook is a build output. These tests are what make that true.

Committing a generated file is a deliberate trade — Kaggle needs something to
push and GitHub renders it — and the cost is that it can drift from its
generator the moment somebody edits the ``.ipynb`` directly. CI's staleness job
is the enforcement; this file tests the properties that job depends on.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "california-housing-sklearn-pipeline.ipynb"


def _build_module():
    """A fresh import each time — the module accumulates cells at import."""
    for name in ("build_notebook",):
        sys.modules.pop(name, None)
    return importlib.import_module("build_notebook")


def test_the_committed_notebook_matches_its_generator() -> None:
    """The same check CI runs. Failing here means someone edited the .ipynb."""
    before = NOTEBOOK.read_bytes()
    result = subprocess.run(
        [sys.executable, "tools/build_notebook.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    after = NOTEBOOK.read_bytes()
    assert before == after, (
        "notebooks/ is stale - rerun `uv run python tools/build_notebook.py`"
    )


def test_rebuild_is_byte_identical(tmp_path: Path) -> None:
    """Stable cell ids, not random ones.

    nbformat 4.5 requires an id on every cell. Generating them randomly would
    make each rebuild differ, and the staleness gate would fail on every run
    rather than only when the content changed - which trains everyone to ignore
    it.
    """
    first = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    subprocess.run([sys.executable, "tools/build_notebook.py"], cwd=REPO, check=True)
    second = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    assert first == second


def test_every_cell_has_a_stable_id() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ids = [cell["id"] for cell in notebook["cells"]]
    assert all(ids), "a cell is missing its id"
    assert len(set(ids)) == len(ids), "duplicate cell ids"


def test_the_notebook_has_no_committed_outputs() -> None:
    """Output blobs and execution counts are exactly the churn that makes a
    committed .ipynb unreviewable."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_every_code_cell_compiles() -> None:
    """The generator refuses a cell that will not compile, so this asserts the
    property on the shipped artefact rather than trusting the generator."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"<cell {index}>", "exec")


def test_a_cell_that_does_not_compile_is_refused() -> None:
    """Red-team the generator's own guard.

    Cell sources are Python assembled by other Python, and the escaping is easy
    to get subtly wrong. Without this the failure surfaces on Kaggle, publicly.
    """
    build = _build_module()
    with pytest.raises(SystemExit) as exc:
        build.code("def broken(:\n    pass")
    assert "does not compile" in str(exc.value)


def test_there_is_exactly_one_tagged_install_cell() -> None:
    """``run_notebook.py`` matches that tag to skip the cell.

    It used to match the substring ``"pip"`` - which also matches
    ``tuned_pipeline`` in the imports cell. The imports were skipped and the
    notebook died three cells later on a ``NameError`` that pointed at a cell
    which was completely fine.
    """
    build = _build_module()
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    tagged = [
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and build.INSTALL_TAG in "".join(cell["source"])
    ]
    assert len(tagged) == 1


def test_the_tag_does_not_match_the_imports_cell() -> None:
    """Pins the specific near-miss, so a future tag change cannot reintroduce
    it."""
    build = _build_module()
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    imports = next(
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "from calhousing" in "".join(cell["source"])
    )
    body = "".join(imports["source"])
    assert "pip" in body, "this test is pointless if the imports cell lost 'pipeline'"
    assert build.INSTALL_TAG not in body


def test_the_notebook_tells_the_reader_where_the_source_is() -> None:
    """A Kaggle notebook with no link back is a dead end."""
    body = NOTEBOOK.read_text(encoding="utf-8")
    assert "orthogonalist-222/california-housing-price" in body


def test_the_negative_results_survive_into_the_notebook() -> None:
    """The findings that contradicted the plan are the most deletable content
    in the project, and the most valuable. Pinned so a tidy-up cannot quietly
    drop them."""
    body = NOTEBOOK.read_text(encoding="utf-8").lower()
    for claim in (
        "did not earn its complexity",  # stacking
        "no ensemble is better than any other",
        "0.135",  # the measured leak
        "whitelist",  # what actually stops a leak
        "housing_median_age",  # the feature that contributes nothing
    ):
        assert claim in body, f"the notebook no longer says: {claim!r}"


def test_markdown_and_code_are_both_substantial() -> None:
    """A notebook that is all code is a script; one that is all prose is a
    blog post."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    kinds = [cell["cell_type"] for cell in notebook["cells"]]
    assert kinds.count("code") >= 10
    assert kinds.count("markdown") >= 10
