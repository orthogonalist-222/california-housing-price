"""Refuse any tracked file the declared repo layout does not name.

UNIVERSAL_PROTOCOL.md section 7: "a path the layout does not name arrives only
through a story that edits the layout first, and CI refuses a tracked path the
declared tree does not cover."

That rule needs a machine to enforce it, because the failure it prevents is
gradual: nobody adds a stray directory on purpose, they add one file at a time
until the tree in the README and the tree on disk are different documents. The
README's tree is for humans; DECLARED below is the same tree in a form a gate
can read, and the two are kept honest by tests/test_layout.py.

Each entry is tagged with the story that introduced it, so `git blame` is not
the only way to answer "why is this path here?".

Run with::

    uv run python tools/check_layout.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

#: (glob, story) for every path this repo is allowed to track.
#:
#: ``*`` matches within one path segment; ``**/`` matches any number of
#: segments. Add an entry in the SAME story that adds the file - that is the
#: whole point of the gate.
DECLARED: list[tuple[str, str]] = [
    # --- M1-S1: scaffold ---------------------------------------------------
    (".gitattributes", "M1-S1"),
    (".gitignore", "M1-S1"),
    (".python-version", "M1-S1"),
    ("pyproject.toml", "M1-S1"),
    ("uv.lock", "M1-S1"),
    ("README.md", "M1-S1"),
    ("CLAUDE.md", "M1-S1"),
    (".github/workflows/ci.yml", "M1-S1"),
    ("docs/sign-off-log.md", "M1-S1"),
    ("docs/findings-register.md", "M1-S1"),
    ("docs/field-notes/*.md", "M1-S1"),
    ("src/calhousing/__init__.py", "M1-S1"),
    ("src/calhousing/py.typed", "M1-S1"),
    ("tools/check_layout.py", "M1-S1"),
    ("tests/test_layout.py", "M1-S1"),
    # --- M1-S2: data contract and split ------------------------------------
    ("docs/adr/*.md", "M1-S2"),
    ("src/calhousing/config.py", "M1-S2"),
    ("src/calhousing/data.py", "M1-S2"),
    ("src/calhousing/splits.py", "M1-S2"),
    ("tests/conftest.py", "M1-S2"),
    ("tests/test_data.py", "M1-S2"),
    ("tests/test_splits.py", "M1-S2"),
    # --- M1-S3: exploratory figures ----------------------------------------
    ("src/calhousing/eda.py", "M1-S3"),
    ("tests/test_eda.py", "M1-S3"),
    # --- M2-S1: the numeric block ------------------------------------------
    ("src/calhousing/preprocess/__init__.py", "M2-S1"),
    ("src/calhousing/preprocess/numeric.py", "M2-S1"),
    ("tests/test_numeric.py", "M2-S1"),
    # --- M2-S2: the categorical block --------------------------------------
    ("src/calhousing/preprocess/categorical.py", "M2-S2"),
    ("tests/test_categorical.py", "M2-S2"),
    # --- M2-S3: engineered features ----------------------------------------
    ("src/calhousing/preprocess/features.py", "M2-S3"),
    ("tests/test_features.py", "M2-S3"),
    # --- M2-S4: assembly ----------------------------------------------------
    ("src/calhousing/preprocess/assemble.py", "M2-S4"),
    ("tests/test_assemble.py", "M2-S4"),
    ("tools/measure_leakage.py", "M2-S4"),
    ("docs/pyspark-to-sklearn.md", "M2-S4"),
    # --- M3-S1: CV harness, metrics, baselines ------------------------------
    ("src/calhousing/evaluate.py", "M3-S1"),
    ("src/calhousing/models.py", "M3-S1"),
    ("tests/test_evaluate.py", "M3-S1"),
    # --- M3-S2: tuned ensembles ---------------------------------------------
    ("src/calhousing/train.py", "M3-S2"),
    ("tests/test_train.py", "M3-S2"),
    # --- M3-S3: gradient boosting libraries ---------------------------------
    ("tests/test_models.py", "M3-S3"),
    # --- M3-S4: stacking, the one-shot run, interpretation ------------------
    ("src/calhousing/interpret.py", "M3-S4"),
    ("tools/final_evaluation.py", "M3-S4"),
    ("tests/test_interpret.py", "M3-S4"),
    # --- M4-S1: the notebook is a build output ------------------------------
    ("tools/build_notebook.py", "M4-S1"),
    ("tools/run_notebook.py", "M4-S1"),
    ("notebooks/california-housing-sklearn-pipeline.ipynb", "M4-S1"),
    ("tests/test_notebook_build.py", "M4-S1"),
    # --- M4-S2: Kaggle packaging, both paths --------------------------------
    ("tools/stage_kaggle.py", "M4-S2"),
    ("kaggle/src_dataset/dataset-metadata.json", "M4-S2"),
    ("kaggle/kernel/kernel-metadata.json", "M4-S2"),
    ("tests/test_packaging.py", "M4-S2"),
    # --- M4-S3: evidence, model card, publish -------------------------------
    #
    # docs/evidence/ is the ONE exception to "artifacts are cattle": the
    # one-shot test-set record cannot be regenerated without scoring the test
    # set again, which is the failure ADR-003 exists to prevent.
    ("docs/evidence/README.md", "M4-S3"),
    ("docs/evidence/test-set-evaluation.json", "M4-S3"),
    ("docs/evidence/headline.csv", "M4-S3"),
    ("docs/evidence/*.png", "M4-S3"),
    ("docs/model-card.md", "M4-S3"),
]

#: Paths that must never be tracked, with the reason the gate gives when it
#: finds one. Checked BEFORE the allowlist so the message names the real
#: problem: "HANDOFF.md is not a deliverable" is actionable, "undeclared path"
#: sends the reader to the wrong fix.
FORBIDDEN: list[tuple[str, str]] = [
    ("HANDOFF.md", "the session log is working memory, not a deliverable"),
    ("**/kaggle.json", "Kaggle credentials must never be committed"),
    ("**/*.pem", "private keys must never be committed"),
    ("**/*.key", "private keys must never be committed"),
    (".env", "environment files carry secrets"),
    ("data/**", "raw data is regenerable; download it instead"),
    ("artifacts/**", "run outputs are cattle, rebuilt by the recipe"),
    (".idea/**", "IDE settings are personal, not project state"),
]


def _to_regex(glob: str) -> re.Pattern[str]:
    """Translate a layout glob into an anchored regex.

    ``fnmatch`` is not usable here: its ``*`` happily matches ``/``, so a
    pattern like ``docs/*.md`` would silently bless ``docs/adr/nested/x.md``
    and the gate would pass exactly the drift it exists to catch.
    """
    out = ["^"]
    i = 0
    while i < len(glob):
        if glob.startswith("**/", i):
            out.append("(?:[^/]+/)*")
            i += 3
        elif glob.startswith("**", i):
            out.append(".*")
            i += 2
        elif glob[i] == "*":
            out.append("[^/]*")
            i += 1
        else:
            out.append(re.escape(glob[i]))
            i += 1
    out.append("$")
    return re.compile("".join(out))


_DECLARED_RE = [(_to_regex(g), g, story) for g, story in DECLARED]
_FORBIDDEN_RE = [(_to_regex(g), g, why) for g, why in FORBIDDEN]


def tracked_files() -> list[str]:
    """Every path git is tracking, as forward-slash relative paths."""
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    return sorted(p for p in out.split("\0") if p)


def classify(paths: list[str]) -> tuple[list[tuple[str, str]], list[str]]:
    """Split ``paths`` into (forbidden, undeclared).

    Two lists rather than one, because the two failures have different cures
    and a gate whose refusals all read alike is a debugging tax (section 4,
    distinguishable signatures).
    """
    forbidden: list[tuple[str, str]] = []
    undeclared: list[str] = []
    for path in paths:
        hit = next((why for rx, _, why in _FORBIDDEN_RE if rx.match(path)), None)
        if hit is not None:
            forbidden.append((path, hit))
        elif not any(rx.match(path) for rx, _, _ in _DECLARED_RE):
            undeclared.append(path)
    return forbidden, undeclared


def missing(paths: list[str]) -> list[tuple[str, str]]:
    """Declared, non-glob paths that git is **not** tracking.

    Added in M4-S3, after the gap let a real defect through (F-007).

    The gate enforced one direction only - *tracked implies declared* - so a
    declared path that was never committed passed silently. That is exactly
    what happened: ``uv build`` writes a ``.gitignore`` containing ``*`` into
    its output directory, the output directory was the tracked
    ``kaggle/src_dataset/``, and so ``dataset-metadata.json`` was invisible to
    git. It existed on the machine that wrote it, the gate said "all declared",
    and CI failed on a runner that had only what was committed.

    Glob entries are skipped: ``docs/field-notes/*.md`` says *these are
    allowed*, not *at least one must exist*.
    """
    tracked = set(paths)
    return [
        (glob, story)
        for glob, story in DECLARED
        if "*" not in glob and glob not in tracked
    ]


def main() -> int:
    paths = tracked_files()
    forbidden, undeclared = classify(paths)
    absent = missing(paths)

    for path, why in forbidden:
        print(f"FORBIDDEN TRACKED PATH: {path}\n    {why}", file=sys.stderr)
    for path in undeclared:
        print(
            f"UNDECLARED TRACKED PATH: {path}\n"
            "    Not covered by DECLARED in tools/check_layout.py. If this file\n"
            "    belongs here, the story adding it must declare it first - add the\n"
            "    entry and the matching line to the tree in README.md.",
            file=sys.stderr,
        )
    for path, story in absent:
        exists = "exists on disk but is NOT tracked" if os.path.exists(path) else "does not exist"
        print(
            f"DECLARED BUT NOT TRACKED: {path}  ({story})\n"
            f"    It {exists}.\n"
            "    A declared path that was never committed passes every check on\n"
            "    the machine that wrote it and fails on every other one. If it is\n"
            "    ignored, find out why: `git check-ignore -v <path>`.",
            file=sys.stderr,
        )

    if forbidden or undeclared or absent:
        print(
            f"\nlayout gate FAILED: {len(forbidden)} forbidden, "
            f"{len(undeclared)} undeclared, {len(absent)} declared-but-untracked, "
            f"out of {len(paths)} tracked files.",
            file=sys.stderr,
        )
        return 1

    print(f"layout gate OK: {len(paths)} tracked files, all declared and present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
