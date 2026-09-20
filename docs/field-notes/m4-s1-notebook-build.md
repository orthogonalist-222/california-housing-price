# M4-S1 — Notebook generator and staleness gate

**Built.** `tools/build_notebook.py` (37 cells: 16 code, 21 markdown),
`tools/run_notebook.py`, a CI job that fails when the committed notebook
differs from its generator, and 11 tests.

## Why the notebook is generated

A committed `.ipynb` is a bad thing to review. The JSON carries execution
counts, output blobs and cell ids that churn on every run, so a diff is mostly
noise and reviewers stop reading it. Generating it means **the reviewable
artefact is plain Python** and the notebook is a build output.

Committing the build output is still deliberate — Kaggle needs a file to push
and GitHub renders it — and the cost of that trade is drift. The CI job is what
pays it:

```yaml
uv run python tools/build_notebook.py
git diff --exit-code notebooks/ || (echo "::error::notebooks/ is stale" && exit 1)
```

Two properties make that gate usable rather than annoying:

- **Stable cell ids.** nbformat 4.5 requires an id on every cell. Random ids
  would make every rebuild differ, so the gate would fail on every run and
  everyone would learn to ignore it. Ids are a hash of position and content.
- **No committed outputs.** Execution counts stay `None`, outputs stay `[]`.

## The generator refuses cells that will not compile

Cell sources are Python assembled by *other* Python, and the escaping is easy
to get subtly wrong — an unescaped `\n` inside a string literal produces a cell
that only fails once it has been pushed to Kaggle and run, in public.
`code()` compiles every cell at build time and prints numbered source on
failure.

## But compiling proves nothing about cell 14

A cell can compile perfectly and raise `KeyError` on execution. So
`tools/run_notebook.py` executes the real file in order and fails on the first
cell that raises.

**It found two things.** The first was this, and it is a better lesson than the
gate itself:

> `run_notebook.py` skipped the install cell by searching cell sources for the
> substring `"pip"`. That also matches **`tuned_pipeline`** in the *imports*
> cell.

So the imports were skipped, and the notebook died three cells later on
`NameError: name 'load_raw' is not defined` — pointing at a cell that was
completely fine. I spent the first minute reading the wrong cell.

The fix is a deliberate marker, `# tag:install`, matched exactly — and
`run_notebook.py` now **refuses** if it does not find exactly one tagged cell,
because silently skipping zero cells (or two) is the same class of failure
wearing a different hat. Two tests pin it, one of which asserts specifically
that the tag does *not* match the imports cell.

The second finding was mine: `Path.write_text(newline=...)` needs Python 3.10+,
and I reached for the system `python` (3.9) instead of `uv run python` out of
habit. The traps section of `CLAUDE.md` already says everything goes through
`uv run`. Writing a rule down does not install it.

## Red-teaming the gate, and getting it wrong the first time

First attempt: hand-edit the notebook, run the build, check `git diff`. Result:
**"GATE FAILED TO NOTICE."**

The gate was fine. My red-team was backwards — `build_notebook.py` *regenerates*
the file, so it overwrote the planted edit before anything looked at it. I had
tested that the generator is deterministic, not that the gate catches drift.

Correct sequence: plant the edit, then check **before** rebuilding.

```
FAILED tests/test_notebook_build.py::test_the_committed_notebook_matches_its_generator
E   At index 113 diff: b'H' != b'C'
```

`H` from "Hand-edited", `C` from "California". Both halves observed, and the
near-miss is worth keeping: *a red-team that runs the repair before the check
proves nothing, and looks exactly like a passing test.*

## What the notebook actually says

It is a narrative, not an API tour, and the sections that matter most are the
ones reporting things that did not work:

- the measured leak is −0.135%, a fraction of the fold noise;
- no ensemble beats any other;
- stacking scored worse than its best member;
- `housing_median_age` contributes almost nothing.

`test_the_negative_results_survive_into_the_notebook` asserts each of those
claims is still present. They are the most deletable content in the project and
the most valuable, and a tidy-up pass would drop them without anybody noticing.

**Measured:** 16 code cells, **124 seconds** end to end, well inside a kernel
budget.

## What to look at

`tools/run_notebook.py` — the tag match and the `skipped != 1` refusal. Then
`test_notebook_build.py::test_the_tag_does_not_match_the_imports_cell`.

## What to try

```
uv run python tools/build_notebook.py
uv run python tools/run_notebook.py
```

Then edit a word in the committed `.ipynb` and run
`uv run pytest tests/test_notebook_build.py -q` **before** rebuilding.
