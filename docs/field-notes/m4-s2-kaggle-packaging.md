# M4-S2 — Dual packaging: GitHub first, Kaggle dataset fallback

**Built.** `tools/stage_kaggle.py`, the two Kaggle metadata files, a dual-path
install cell, ADR-004, and 15 tests.

## Both paths were installed, not asserted

This is the part that mattered. "The fallback works" is the easiest sentence in
the project to write and the easiest to be wrong about.

| Path | How it was verified |
| --- | --- |
| GitHub | `uv pip install git+https://…` into a **fresh 3.11 venv**; imported `calhousing`, `TUNED`, `build_pipeline` |
| Offline wheel | Fresh 3.11 venv with only the **runtime** deps preinstalled — Kaggle's shape — then `pip install --no-index --find-links <dir> <wheel>` with **no network index**; imported the package and built a pipeline |

`--no-index` is what makes the second row worth anything: pip cannot reach PyPI
at all, so a missing dependency fails loudly instead of being quietly fetched
from the network the test was supposed to be simulating the absence of.

```
clean venv with Kaggle-like preinstalled deps
=== FALLBACK PATH: --no-index --find-links, no network index ===
install rc=0
FALLBACK OK: calhousing 0.1.0 | arms ['hgb', 'lgbm', 'rf', 'ridge', 'xgb'] | pipeline builds
```

## The compile guard earned its keep, on its first real outing

Writing the install cell produced this:

```
cell 2 does not compile: unterminated string literal (detected at line 33)

 32 |         raise SystemExit(
 33 |             "Could not install calhousing.
 34 | "
```

A `\n` written one layer up — in the generator's f-string — became a **real
newline inside a string literal** in the generated cell. Exactly the defect
`build_notebook.code()`'s docstring predicts, caught by the guard it describes,
before the notebook went anywhere near Kaggle.

The fix is not better escaping; it is not needing any. The message is printed
line by line and the `SystemExit` carries a single-line string. **A layer of
code generation you have to escape through is a layer you will eventually get
wrong** — so the cell contains no escapes at all.

## The tool prints the upload commands; it does not run them

Publishing to a live Kaggle account is outward-facing. A tool that does it as a
side effect of "staging" will one day publish something nobody meant to, so
`stage_kaggle.py` builds the wheel, copies the notebook, and then **prints**:

```
  kaggle datasets version -p kaggle/src_dataset --dir-mode zip -m "rebuild from <commit sha>"
  kaggle kernels push -p kaggle/kernel
```

`test_staging_does_not_upload` monkeypatches `subprocess.run` to raise, and
asserts it is never called. The absence of an upload is asserted, not assumed.

## The wiring the fallback depends on

The offline branch reads `/kaggle/input/calhousing-src`. That directory only
exists if the kernel **attaches** that dataset — so:

- `stage_kaggle.py` **refuses** if `kernel-metadata.json` does not list the
  dataset the fallback reads. Otherwise the fallback is a comment.
- A test asserts the notebook's hardcoded mount path matches the dataset id's
  slug. A renamed dataset with an unchanged notebook is a fallback that
  silently looks in the wrong place.

## Two brittleness bugs the tests found in the tool

1. **`Path.relative_to` raises** on a path outside its argument. The tool
   printed `DATASET_DIR.relative_to(REPO)` and crashed outright when pointed at
   a directory elsewhere — which is what a test does, and what anyone staging
   from a different checkout layout would do. Now `_display()`, which falls back
   to the absolute path.
2. **It staged before validating.** The dataset-attachment check ran *after*
   building a wheel and copying a notebook, so a misconfiguration left
   half-staged files behind for the next run to be confused by. Validation
   moved first.

Neither is dramatic. Both were found by writing tests for a script, which is
the argument for `tools/` being importable at all.

## The notebook says which path it used

Two runs of the same notebook can be running different code — GitHub is `main`,
the dataset is whatever was last staged. Silently differing behaviour between
two runs is worse than either path failing loudly, so the cell prints
`calhousing 0.1.0 installed from GitHub` or `… from the attached Kaggle dataset
(offline)`.

That divergence is a real cost of the dual path, recorded in ADR-004 rather
than glossed: **re-staging after a code change is manual and nothing enforces
it.** The alternative was a tool that uploads to a live account on its own.

## What to look at

`ADR-004`'s options table — particularly why the sibling project's
dataset-only pattern (proven on this account) was not chosen, and what that
costs.

## What to try

```
uv run python tools/stage_kaggle.py
```

Then delete `duonghongphu/calhousing-src` from `dataset_sources` in
`kaggle/kernel/kernel-metadata.json` and run it again.
