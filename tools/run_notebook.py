"""Execute the generated notebook, the way Kaggle will.

A notebook that is built but never run is a notebook that fails on Kaggle, in
front of whoever opened it. ``tools/build_notebook.py`` compiles each cell, so
a syntax error cannot ship — but compiling proves nothing about a `KeyError` on
cell 14.

This runs the real file, in order, and fails on the first cell that raises.

Two things it does **not** do, deliberately:

- It does not write outputs back into the committed notebook. Execution counts
  and output blobs are exactly the churn that makes a committed ``.ipynb``
  unreviewable, and the CI staleness gate compares bytes.
- It does not run in CI by default. The notebook fits five models; that is
  minutes, and CI already runs 240 tests against the code the notebook calls.
  It is a pre-publish gate, run by ``--check`` before a kernel push.

Run with::

    uv run python tools/run_notebook.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebook import INSTALL_TAG  # noqa: E402

DEFAULT_NOTEBOOK = Path("notebooks/california-housing-sklearn-pipeline.ipynb")


def execute(path: Path, *, timeout: int = 1800, skip_install: bool = True) -> int:
    notebook = nbformat.read(path, as_version=4)

    if skip_install:
        # The install cell pulls the package from GitHub. Locally that would
        # replace the editable install with whatever is on main - which is a
        # different thing from what is being tested. M4-S2 exercises that path
        # separately, in a clean environment.
        #
        # Matched by TAG, not by substring. Searching cell sources for "pip"
        # also matches `tuned_pipeline` in the IMPORTS cell - which is exactly
        # what happened: the imports were skipped and the notebook died three
        # cells later on `NameError: name 'load_raw' is not defined`, pointing
        # at a cell that was completely fine.
        skipped = 0
        for cell in notebook.cells:
            if cell.cell_type == "code" and INSTALL_TAG in "".join(cell.source):
                cell.source = "# install cell skipped by tools/run_notebook.py\n"
                skipped += 1
        if skipped != 1:
            raise SystemExit(
                f"expected exactly one cell tagged {INSTALL_TAG!r}, found "
                f"{skipped}. Has build_notebook.py changed?"
            )

    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(Path.cwd())}},
    )

    started = time.time()
    try:
        client.execute()
    except CellExecutionError as exc:
        print(f"\nNOTEBOOK FAILED after {time.time() - started:,.0f}s\n", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    elapsed = time.time() - started
    code_cells = sum(1 for c in notebook.cells if c.cell_type == "code")
    print(f"notebook OK: {code_cells} code cells in {elapsed:,.0f}s")
    if elapsed > 900:
        print(
            f"WARNING: {elapsed:,.0f}s is a long kernel run. Kaggle's limit is "
            "12 hours for CPU notebooks, but a reader's patience is not.",
            file=sys.stderr,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default=str(DEFAULT_NOTEBOOK))
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--with-install",
        action="store_true",
        help="run the pip-install cell too (replaces the local editable install)",
    )
    args = parser.parse_args(argv)
    return execute(
        Path(args.notebook), timeout=args.timeout, skip_install=not args.with_install
    )


if __name__ == "__main__":
    raise SystemExit(main())
