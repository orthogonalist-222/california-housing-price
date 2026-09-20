# ADR-004 — How the kernel gets the package

- **Status:** accepted
- **Date:** 2026-09-20
- **Story:** M4-S2
- **Deciders:** ML engineer; the dual-path shape was the user's call at plan time
  ("Both paths, GitHub first")

## Context

The pipeline lives in a package so it can be unit-tested — 251 tests, two Python
versions, in CI. The notebook therefore has to obtain that package at runtime,
and a Kaggle kernel has exactly two ways to receive code:

1. **`pip install` over the network**, which requires the notebook's **Internet**
   toggle to be ON.
2. **An attached dataset**, which works with Internet off.

Neither is unconditionally available. The Internet toggle is a per-notebook
setting any reader can flip; a competition kernel cannot enable it at all; and
Kaggle's runners have had outbound-network incidents. Equally, an attached
dataset holds whatever was last staged — not whatever is on `main`.

The sibling project in this account (`Wine Quality Dataset`) runs
`enable_internet: false` and ships its package as a dataset, which is proven on
this account. That was offered as the default and the user chose the dual path
instead.

## The decision

**Try GitHub first; fall back to an attached wheel.**

```python
if _pip("git+https://github.com/orthogonalist-222/california-housing-price") == 0:
    source = "GitHub"
else:
    wheels = sorted(glob.glob("/kaggle/input/calhousing-src/*.whl"))
    _pip("--no-index", "--find-links", FALLBACK_DIR, wheels[-1])
    source = "the attached Kaggle dataset (offline)"
```

- `enable_internet: true` in `kernel-metadata.json`, so the primary path is
  available by default.
- `duonghongphu/calhousing-src` attached as a dataset regardless, so the
  fallback is always *reachable* — a fallback that needs a configuration change
  to work is not a fallback.
- The notebook **prints which path it used**. A reader debugging a stale result
  needs to know whether they are running `main` or a staged wheel, and silently
  differing behaviour between two runs of the same notebook is worse than
  either path failing loudly.
- `tools/stage_kaggle.py` builds the wheel and **prints** the upload commands
  rather than running them. Publishing to someone's Kaggle account is
  outward-facing; a tool that does it as a side effect of "staging" will one day
  publish something nobody meant to.
- The staging tool **refuses** if `kernel-metadata.json` does not attach the
  dataset the fallback reads from. Otherwise the fallback is a comment rather
  than a mechanism.

## Both paths are exercised, not assumed

| Path | How it was verified |
| --- | --- |
| GitHub | `uv pip install git+https://…` into a fresh Python 3.11 venv; imported `calhousing`, `TUNED`, `build_pipeline` |
| Offline wheel | Fresh 3.11 venv with only the *runtime* deps preinstalled (Kaggle's shape), then `pip install --no-index --find-links <dir> <wheel>` with **no network index**; imported the package and built a pipeline |

The second is the one that would otherwise be decoration. `--no-index` means
pip cannot reach PyPI at all, so a missing dependency fails rather than being
quietly fetched.

## Options considered

| Option | Why not |
| --- | --- |
| GitHub only | One network incident, or one reader with Internet off, and the notebook shows a traceback instead of running |
| Dataset only (the sibling's pattern) | Proven on this account and genuinely simpler — but every code change needs a re-upload before the kernel reflects it, and the repo could stay private, which the user did not want |
| Vendor the code into the notebook | No dependency at all, and no unit tests — the package exists precisely so the pipeline is testable |
| `kernel_sources` (a Utility Script) | Kaggle-native, but the code would then live in two places with no CI gate keeping them in step |

## Consequences

- **The two paths can serve different code.** GitHub is `main`; the dataset is
  whatever was last staged. The notebook prints which one it used, and
  `stage_kaggle.py` suggests a commit sha in the dataset version message so the
  wheel is traceable.
- **Re-staging is a manual step** after a code change, and nothing enforces it.
  Accepted: the alternative is a tool that uploads to a live account on its own.
- **The repo must stay public** for the primary path. Recorded here because it
  is a constraint on the repo, not just on the notebook.
- The wheel is 0.05 MB, so the dataset is negligible to host.

## Revisit trigger

Reopen if the kernel is ever entered into a competition (Internet cannot be
enabled, so the fallback becomes the only path and should become the primary
one), or if the repo is made private.
