"""Stage the package as a Kaggle dataset — the notebook's offline fallback.

The kernel installs `calhousing` from GitHub. That needs the notebook's
Internet toggle to be on and GitHub to be reachable from Kaggle's runners, and
neither is guaranteed: the toggle is a per-notebook setting a reader can flip,
and a competition kernel cannot enable it at all.

So the notebook has a second path — a Kaggle **dataset** holding a wheel of
this package, attached to the kernel and installed with `--no-index`. This
script builds that dataset's payload and prints the exact command to publish
it.

Deliberately **prints** the upload command rather than running it. Publishing
to someone's Kaggle account is an outward-facing action; a tool that does it as
a side effect of "staging" is a tool that will one day publish something nobody
meant to.

Run with::

    uv run python tools/stage_kaggle.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO / "kaggle" / "src_dataset"
KERNEL_DIR = REPO / "kaggle" / "kernel"
NOTEBOOK = REPO / "notebooks" / "california-housing-sklearn-pipeline.ipynb"


def build_wheel(out: Path) -> Path:
    """Build a wheel of the package into ``out``.

    A wheel rather than a source zip: the kernel installs it with `--no-index`,
    and a source distribution would need a build backend that the offline
    runner does not have. `uv build` uses the `uv_build` backend declared in
    `pyproject.toml`.
    """
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.whl"):
        stale.unlink()

    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"wheel build failed:\n{result.stdout}\n{result.stderr}")

    wheels = sorted(out.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel in {out}, found {len(wheels)}")
    return wheels[0]


def stage_kernel() -> Path:
    """Copy the generated notebook next to its kernel metadata.

    The copy is gitignored - the notebook is committed once, at its source in
    `notebooks/`, and a second tracked copy would be a second thing to keep in
    sync.
    """
    if not NOTEBOOK.is_file():
        raise SystemExit(
            f"{NOTEBOOK} does not exist. Run tools/build_notebook.py first."
        )
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    target = KERNEL_DIR / NOTEBOOK.name
    shutil.copy2(NOTEBOOK, target)
    return target


def _display(path: Path) -> str:
    """A path to print: repo-relative when it is inside the repo, else absolute.

    ``Path.relative_to`` RAISES on a path outside its argument, so the naive
    version crashed the whole tool when pointed at a directory elsewhere -
    which is exactly what a test does, and what anybody staging from a
    checkout in a different layout would do.
    """
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _read_metadata(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-wheel", action="store_true")
    args = parser.parse_args(argv)

    dataset_meta = _read_metadata(DATASET_DIR / "dataset-metadata.json")
    kernel_meta = _read_metadata(KERNEL_DIR / "kernel-metadata.json")

    # Validate the wiring FIRST. The kernel must attach the dataset it falls
    # back to, or the fallback is a comment rather than a mechanism - and
    # discovering that after building a wheel and copying a notebook leaves
    # half-staged files behind for the next run to be confused by.
    dataset_id = dataset_meta["id"]
    if dataset_id not in kernel_meta.get("dataset_sources", []):
        raise SystemExit(
            f"kernel-metadata.json does not attach {dataset_id!r} as a dataset "
            "source, so the offline fallback could never find the wheel."
        )

    if not args.skip_wheel:
        wheel = build_wheel(DATASET_DIR)
        size_mb = wheel.stat().st_size / 1e6
        print(f"wheel:    {wheel.name}  ({size_mb:.2f} MB)")

    notebook = stage_kernel()
    print(f"notebook: {_display(notebook)}")
    print()

    print("To publish, run these yourself - this script does not upload:")
    print()
    print(f"  # 1. the package, as a dataset  ({dataset_id})")
    print(f"  kaggle datasets create  -p {_display(DATASET_DIR)} --dir-mode zip")
    print(f"  #    ...or, after the first time:")
    print(
        f"  kaggle datasets version -p {_display(DATASET_DIR)} --dir-mode zip "
        '-m "rebuild from <commit sha>"'
    )
    print()
    print(f"  # 2. the kernel  ({kernel_meta['id']})")
    print(f"  kaggle kernels push -p {_display(KERNEL_DIR)}")
    print()
    print("Then check the kernel's Internet toggle: ON uses the GitHub install,")
    print("OFF exercises the dataset fallback. Both paths should run green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
