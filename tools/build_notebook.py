"""Generate the Kaggle notebook from source.

Notebooks are a poor thing to hand-edit and a worse thing to diff: the JSON
carries execution counts, output blobs and cell ids that churn on every run, so
reviewing a committed ``.ipynb`` is mostly noise. Generating it from this script
makes the reviewable artefact plain Python, and the notebook a build output.

CI runs this and fails if ``notebooks/`` differs from what it produces, so the
committed notebook can never drift from the code that claims to generate it.

Run with::

    uv run python tools/build_notebook.py
"""

from __future__ import annotations

import hashlib
import json
import pathlib

REPO = "https://github.com/orthogonalist-222/california-housing-price"
OUT = pathlib.Path("notebooks/california-housing-sklearn-pipeline.ipynb")

cells: list[dict] = []


def _cell_id(index: int, body: str) -> str:
    """A stable cell id derived from position and content.

    nbformat 4.5 requires every cell to carry an id, and Kaggle's runner warns
    that the missing-id fallback will become an error. Generating them from a
    hash rather than at random keeps the build reproducible - a random id would
    make every rebuild differ, and the CI staleness check would fail on every
    run rather than only when the content changed.
    """
    digest = hashlib.sha1(f"{index}:{body}".encode("utf-8")).hexdigest()
    return f"c{digest[:12]}"


def md(source: str) -> None:
    body = source.strip("\n")
    cells.append(
        {
            "cell_type": "markdown",
            "id": _cell_id(len(cells), body),
            "metadata": {},
            "source": body.splitlines(keepends=True),
        }
    )


def code(source: str) -> None:
    """Add a code cell, refusing anything that will not compile.

    Cell sources are Python being assembled by other Python, and the escaping
    is easy to get subtly wrong - an unescaped ``\\n`` inside a string literal
    produces a cell that only fails once it has been pushed to Kaggle and run.
    Compiling here turns a published traceback into a build failure.
    """
    body = source.strip("\n")
    try:
        compile(body, f"<cell {len(cells)}>", "exec")
    except SyntaxError as exc:
        numbered = "\n".join(
            f"{i:>3} | {line}" for i, line in enumerate(body.splitlines(), 1)
        )
        raise SystemExit(
            f"\ncell {len(cells)} does not compile: {exc}\n\n{numbered}\n"
        ) from exc
    cells.append(
        {
            "cell_type": "code",
            "id": _cell_id(len(cells), body),
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": body.splitlines(keepends=True),
        }
    )


# =============================================================================
# The narrative
# =============================================================================

md(
    """
# California Housing Prices — a scikit-learn preprocessing pipeline

**The preprocessing is the deliverable. The model is its passenger.**

This notebook rebuilds the shape of a Spark ML pipeline in scikit-learn —
`StringIndexer` → `OneHotEncoder` → `VectorAssembler` → `Pipeline` →
`CrossValidator` — on the 1990 California census extract.

It is not a tour of the API. Every design choice below was **measured first**,
and several of the measurements contradicted the plan they came from. Those
are kept, because a walkthrough where everything works as advertised teaches
you nothing about the day it does not.

| Spark ML | scikit-learn |
| --- | --- |
| `Imputer` | `SimpleImputer` / `KNNImputer` / `IterativeImputer` |
| `StringIndexer` | `OrdinalEncoder` |
| `OneHotEncoder` | `OneHotEncoder` |
| `Bucketizer` / `QuantileDiscretizer` | `KBinsDiscretizer` |
| `StandardScaler` / `MinMaxScaler` | same names |
| a UDF / `withColumn` chain | a `TransformerMixin` subclass |
| **`VectorAssembler`** | **`ColumnTransformer`** |
| `Pipeline` | `Pipeline` |
| `CrossValidator` + `ParamGridBuilder` | `RandomizedSearchCV` over the whole pipeline |

Source, tests and the full decision record: """
    + REPO
)

md(
    """
## Setup

The pipeline lives in a package rather than in this notebook, so it can be
unit-tested. 240 tests run against it in CI on Python 3.11 and 3.12.
"""
)

#: Marker on the install cell. Tooling that needs to find that cell matches
#: this exact line rather than searching for "pip" - a substring search hits
#: `tuned_pipeline` in the imports cell, which is how `run_notebook.py` came to
#: skip the imports and fail on a NameError three cells later.
INSTALL_TAG = "# tag:install"

code(
    f"""
{INSTALL_TAG}
# Two ways in, because one of them is not always available.
#
#   1. pip install from GitHub - always current, needs the notebook's Internet
#      toggle ON. A reader can flip that off, and a competition kernel cannot
#      turn it on at all.
#   2. a wheel from the attached `calhousing-src` dataset - works offline, and
#      is whatever was last staged rather than whatever is on main.
#
# The fallback is not decoration: it is the difference between this notebook
# running for a reader and showing them a traceback.
import glob
import subprocess
import sys

REPO = "git+{REPO}"
FALLBACK_DIR = "/kaggle/input/calhousing-src"


def _pip(*args) -> int:
    return subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *args]
    ).returncode


source = None
if _pip(REPO) == 0:
    source = "GitHub"
else:
    wheels = sorted(glob.glob(FALLBACK_DIR + "/*.whl"))
    if not wheels:
        # Printed line by line rather than joined with an escape. Cell sources
        # are Python assembled by other Python, and a backslash-n written one
        # layer up becomes a real newline inside a string literal down here -
        # which is precisely the defect build_notebook's compile() guard
        # caught when this cell was first written.
        print("Could not install calhousing.")
        print("  - the GitHub install failed (is the Internet toggle on?)")
        print(f"  - and no wheel was found in {{FALLBACK_DIR}}")
        raise SystemExit("Attach the calhousing-src dataset, or enable Internet.")
    if _pip("--no-index", "--find-links", FALLBACK_DIR, wheels[-1]) != 0:
        raise SystemExit(f"Found {{wheels[-1]}} but pip refused to install it.")
    source = "the attached Kaggle dataset (offline)"

import calhousing

print(f"calhousing {{calhousing.__version__}} installed from {{source}}")
"""
)

code(
    """
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from calhousing import config
from calhousing.data import load_raw
from calhousing.splits import split_train_test, coverage_report, income_band
from calhousing.evaluate import segment_table, compare
from calhousing.models import TUNED, tuned_pipeline, cross_validate_pipeline

pd.set_option("display.width", 110)
pd.set_option("display.float_format", lambda v: f"{v:,.1f}")
warnings.filterwarnings("ignore", category=UserWarning)

print("ready")
"""
)

md(
    """
## 1. The data, and the three defects that make it worth using

This dataset was chosen over `sklearn.datasets.fetch_california_housing`
precisely because it is *not* clean. A pipeline needs something to do.
"""
)

code(
    """
frame = load_raw()

print(f"shape: {frame.shape}")
print()
print("nulls:")
print(frame.isna().sum()[lambda s: s > 0].to_string())
print()
print("ocean_proximity:")
print(frame.ocean_proximity.value_counts().to_string())
print()
cap, floor = frame[config.TARGET].max(), frame[config.TARGET].min()
print(f"target cap   ${cap:,.0f}  ->  {(frame[config.TARGET] == cap).sum()} rows")
print(f"target floor ${floor:,.0f}  ->  {(frame[config.TARGET] == floor).sum()} rows")
"""
)

md(
    """
Three things to carry forward:

1. **207 missing `total_bedrooms`** — an imputer that must be fitted inside
   each fold.
2. **`ISLAND` has five rows** out of 20,640. Any fold can be fitted without
   ever seeing it.
3. **The target is censored at both ends** — 965 rows pinned at \\$500,001 and
   4 at \\$14,999. A model *cannot be right* about those rows, so every
   evaluation in this notebook reports them separately.

The censoring is the easiest one to plot and the easiest one to forget.
"""
)

code(
    """
target = frame[config.TARGET]
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.hist(target, bins=120, color="#2a6f97")
ax.axvline(config.TARGET_CAP, color="#9d0208", ls="--")
ax.annotate(f"{(target == config.TARGET_CAP).sum():,} rows at the cap",
            xy=(config.TARGET_CAP, ax.get_ylim()[1] * 0.8),
            xytext=(-12, 0), textcoords="offset points",
            ha="right", color="#9d0208")
ax.set_xlabel("median_house_value ($)"); ax.set_ylabel("block groups")
ax.set_title("The cap is a wall, not a tail", loc="left")
plt.show()
"""
)

md(
    """
## 2. The split — not the obvious one

The standard recipe stratifies on income band, because `median_income` is the
strongest single predictor and a purely random holdout distorts it.

That part is right. But measured on this file, the resulting `ISLAND` coverage
is a **lottery**:

| seed | ISLAND in train | ISLAND in test |
| --- | --- | --- |
| 0 | 4 | 1 |
| **42** | **2** | 3 |
| 7 | 5 | **0** |
| 2024 | 3 | 2 |

Seed 7 hands the test set no island at all. Seed 42 satisfied the criterion
*by luck*.

Stratifying on `income_band × ocean_proximity` does not work either —
`band 3 × ISLAND` has exactly one row, and scikit-learn refuses:
`The least populated class in y has only 1 member`.

So `stratification_key` **collapses per level**: if any composite group inside
an `ocean_proximity` level is too small to split, every row of that level is
stratified by the level name alone.
"""
)

code(
    """
train, test = split_train_test(frame)
print(f"train {len(train):,}   test {len(test):,}")
print()
print(coverage_report(train, test).to_string())
print()
for seed in (0, 7, 42, 2024):
    tr, te = split_train_test(frame, seed=seed)
    row = coverage_report(tr, te).loc["ISLAND"]
    print(f"  seed {seed:>4}:  ISLAND train={row.train}  test={row.test}")
"""
)

md(
    """
4 / 1 at every seed. That guarantee turns out to be load-bearing **for the
encoder**, two sections below — a decision made for the split holding up a
decision made for the preprocessing.
"""
)

md(
    """
## 3. The preprocessing blocks

Each block is a `Pipeline` with named steps, so a search can address them by
path (`preprocess__heavy__deskew`). Disabled steps are `"passthrough"`, not
removed — a named-but-inert step stays addressable; a removed one silently
makes a search space target nothing.
"""
)

code(
    """
from calhousing.preprocess.numeric import build_numeric_block, DomainError

block = build_numeric_block(imputer="median", deskew="log", scaler="standard")
print([name for name, _ in block.steps])

# The guard that stops a silent failure: log1p is undefined at or below -1,
# and longitude is about -124. Without this it returns NaN, quietly.
try:
    build_numeric_block(deskew="log").fit_transform(frame[config.NUMERIC_COLUMNS])
except DomainError as exc:
    print()
    print("REFUSED:", exc)
"""
)

md(
    """
`np.log1p(-124)` is `NaN`. numpy shrugs. Without that guard the NaNs reach
`StandardScaler`, which warns about an invalid divide from inside scikit-learn's
`extmath` — naming neither the column nor the step. Inside a CV loop it is a
model that trains fine and scores like noise.

### Label vs one-hot — three things that had to be measured

**`infrequent_if_exist` does nothing on its own.** With no `min_frequency` it
is identical to `"ignore"`. And `min_frequency` only creates the infrequent
bucket when a *training* category is below the threshold — which is why the
split's guarantee matters here.
"""
)

code(
    """
from calhousing.preprocess.categorical import make_onehot_encoder, make_ordinal_encoder

cat = frame[["ocean_proximity"]]
island_free = cat[cat.ocean_proximity != "ISLAND"]
one_island = pd.DataFrame({"ocean_proximity": ["ISLAND"]})

with_island = make_onehot_encoder(min_frequency=10).fit(cat)
without = make_onehot_encoder(min_frequency=10).fit(island_free)

print("fit WITH the 4-row island:")
print("  columns:", list(with_island.get_feature_names_out()))
print("  unknown ->", with_island.transform(one_island)[0], "sum", with_island.transform(one_island).sum())
print()
print("fit WITHOUT it:")
print("  columns:", list(without.get_feature_names_out()))
print("  unknown ->", without.transform(one_island)[0], "sum", without.transform(one_island).sum())
"""
)

md(
    """
A **presence** when the rare level was seen at fit time; an **absence** when it
was not. That is the split guarantee paying for itself.

**`drop="first"` destroys the distinction it looks like it keeps.** scikit-learn
permits `drop="first"` with tolerant unknown handling, and the result is that
an unseen category and the dropped baseline become the *same vector*.
"""
)

code(
    """
from sklearn.preprocessing import OneHotEncoder
from calhousing.preprocess.categorical import UnsafeEncoderError

unsafe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False).fit(island_free)
reference = unsafe.categories_[0][0]
print(f"dropped reference level: {reference!r}")
print("unknown 'ISLAND'  ->", unsafe.transform(one_island))
print(f"known   {reference!r} ->", unsafe.transform(pd.DataFrame({"ocean_proximity": [reference]})))
print()
try:
    make_onehot_encoder(drop="first", handle_unknown="ignore")
except UnsafeEncoderError as exc:
    print("REFUSED:", exc)
"""
)

md(
    """
**Declaring the categories fixes it at the source.** `OrdinalEncoder` given an
explicit level list encodes `ISLAND` correctly even from a fit that never saw
one, because the *schema* defines the levels rather than the sample.

And the ordering is an assertion, not a convenience: `ISLAND < NEAR BAY <
NEAR OCEAN < <1H OCEAN < INLAND` is increasing distance from water, so
`0 < 1 < 2 < 3 < 4` states something true. Spark's `StringIndexer` orders by
**frequency**, which asserts nothing — and any model able to use an ordering
will happily use that noise.
"""
)

code(
    """
declared = make_ordinal_encoder().fit(island_free)           # schema-defined
learned = make_ordinal_encoder(declared=False).fit(island_free)  # StringIndexer-like

print("declared categories :", float(declared.transform(one_island)[0, 0]),
      f"  (its true position: {config.OCEAN_PROXIMITY_ORDER.index('ISLAND')})")
print("learned categories  :", float(learned.transform(one_island)[0, 0]), "  (unknown marker)")
"""
)

md(
    """
## 4. The assembler — `VectorAssembler` is a `ColumnTransformer`

Six branches. Columns feed **more than one** — `total_rooms` is a denominator
in `ratios` and a de-skewed magnitude in `heavy`. A `ColumnTransformer` is a
fan-out, not a partition.
"""
)

code(
    """
from calhousing.preprocess.assemble import build_preprocessor, feature_names
from sklearn.linear_model import Ridge

X_train = train.drop(columns=[config.TARGET]); y_train = train[config.TARGET]
X_test = test.drop(columns=[config.TARGET]);   y_test = test[config.TARGET]

pre = build_preprocessor().fit(X_train)
names = feature_names(pre)
print(f"{len(names)} assembled features from {X_train.shape[1]} raw columns")
for n in names[:6] + ["..."] + names[-6:]:
    print("   ", n)
"""
)

md(
    """
## 5. What leakage actually costs here

Everyone says "fit your preprocessing inside the fold or you will leak." True.
Also usually presented with an implied magnitude nobody checks.

Measured on this dataset:

```
honest  RMSE  63,811  +/- 1,430      (preprocessing fitted inside each fold)
leaked  RMSE  63,897  +/- 1,631      (fitted once, on everything)
leak             -86   (-0.135%)     -> 0.06x the fold-to-fold spread
```

The "leaky" version is **worse**, by an amount buried in noise. Shrinking the
training set does not produce a trend either — the sign flips.

So is the discipline pointless? No — but the reason is different from the one
usually given.
"""
)

code(
    """
rng = np.random.default_rng(0)
poisoned = X_train.assign(appraisal=y_train * 0.98 + rng.normal(0, 5_000, len(y_train)))

clean_names = feature_names(build_preprocessor().fit(X_train))
poisoned_names = feature_names(build_preprocessor().fit(poisoned))

print("a target-derived column was added to the input frame.")
print("  features before:", len(clean_names))
print("  features after :", len(poisoned_names))
print("  'appraisal' reached the model:", any("appraisal" in n for n in poisoned_names))
print()
print("remainder='drop' means the assembler is a WHITELIST.")
print("Forced past it, that column takes RMSE from 63,811 to 5,121.")
"""
)

md(
    """
**The leak that matters is a leaked column, and the whitelist is what stops
it.** `remainder="passthrough"` — one word different — is how such a column
arrives without anybody choosing to put it there.

The real payoff of the pipeline on a dataset like this is that the feature set
a model can see is **declared in one place, reviewable, and enforced**. The
fold discipline keeps that true when the data is smaller, shifted, or has a
stateful step with more to learn — and it costs nothing to keep.
"""
)

md(
    """
## 6. Searching the preprocessing *and* the model

This is what a hand-rolled `fit_transform` chain cannot do. By the time you are
tuning the model, the preprocessing has already happened and its choices are
baked into the matrix.

The searches are not re-run here (25 draws × 5 folds × four models is well past
a kernel budget). Their winners are recorded in `models.TUNED`:
"""
)

code(
    """
for name in ("ridge", "rf", "hgb", "xgb", "lgbm"):
    entry = TUNED[name]
    print(f"{name:6s} preprocessing: {entry['preprocess']}")
"""
)

md(
    """
Five of LightGBM's ten winning parameters are **preprocessing** decisions — the
number of geographic clusters, the RBF width, the bin count, the de-skew
transform, the imputer — chosen by the same cross-validated evidence that chose
the learning rate.

Cross-validated results on the training split:

| model | CV RMSE | fold std | train RMSE | fit | artefact |
| --- | --- | --- | --- | --- | --- |
| ridge | 63,089 | ±1,993 | 62,875 | 0.1s | — |
| hgb | 43,571 | ±1,445 | 23,131 | 5.0s | 3.3 MB |
| rf | 42,707 | ±1,178 | 15,852 | 115.5s | **289.4 MB** |
| xgb | 42,401 | ±1,360 | 4,996 | 20.3s | 13.5 MB |
| **lgbm** | **42,166** | ±1,407 | 9,052 | **5.1s** | 6.8 MB |

**Read that table carefully: no ensemble is better than any other.** The four
span 1,405 RMSE while their own fold standard deviations run ±1,178 to ±1,407.
The whole spread is about one arm's own fold noise.

What the evidence *does* separate is cost. RandomForest is **289 MB and 115
seconds** against LightGBM's **6.8 MB and 5.1 seconds** for an
indistinguishable score.
"""
)

md(
    """
## 7. The test set, scored once

Five arms, one invocation, against a list fixed *before any of them existed*.
The test set was not looked at, plotted or described until this point.
"""
)

code(
    """
import time

results, tables = {}, {}
for arm in ("dummy", "ridge", "rf", "lgbm", "stack"):
    started = time.time()
    model = tuned_pipeline(arm).fit(X_train, y_train)
    predictions = model.predict(X_test)
    tables[arm] = segment_table(X_test, y_test, predictions)
    results[arm] = dict(tables[arm].loc["all"])
    results[arm]["seconds"] = round(time.time() - started, 1)
    if arm == "lgbm":
        best_model = model

print(compare(results).to_string())
"""
)

md(
    """
**Stacking did not earn its complexity.** It scored *worse* than its own best
member, at roughly twelve times the fit time. Four learners that agree within
their own fold noise give a meta-learner nothing to arbitrate.

And the CV protocol held: CV RMSE 42,166 → test RMSE 42,251, a gap of **0.2%**.
"""
)

code(
    """
for arm in ("dummy", "ridge", "lgbm"):
    print(f"{arm} - by segment:")
    print(tables[arm].to_string())
    print()
"""
)

md(
    """
**This is why the censored rows were kept and reported separately.**

LightGBM scores RMSE 38,939 on uncensored rows and **83,570 on censored ones —
R² of −5.1**, worse than predicting their own mean. It cannot be otherwise:
their true value is unknown and *above* the cap, so a prediction below is wrong
by an unknowable amount and one above is penalised for being closer to the
truth.

The dummy's censored R² is **−87.8**.

5% of the test set. A single headline RMSE averages it away and the model looks
uniformly competent.
"""
)

md(
    """
## 8. What the model learned
"""
)

code(
    """
from calhousing.interpret import importance_table

slice_X = X_train.sample(1500, random_state=config.RANDOM_SEED)
slice_y = y_train.loc[slice_X.index]
importance = importance_table(best_model, slice_X, slice_y, n_repeats=5)
print(importance.to_string(index=False))
"""
)

code(
    """
from sklearn.inspection import PartialDependenceDisplay

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
PartialDependenceDisplay.from_estimator(
    best_model, slice_X, ["median_income", "housing_median_age"],
    ax=axes, line_kw={"color": "#2a6f97", "lw": 2},
)
for ax in axes:
    ax.grid(alpha=0.25)
fig.suptitle("The shape the model learned", x=0.01, ha="left")
plt.tight_layout(); plt.show()
"""
)

md(
    """
**Geography outranks income.** `longitude` and `latitude` lead the ranking —
and the fact that the *raw* coordinates rank so high **after** twenty
cluster-similarity columns were engineered from them says those columns did not
exhaust the spatial signal.

`median_income` drives price monotonically from about \\$175k to \\$318k.
`housing_median_age` is **flat** — a column every tutorial keeps, contributing
almost nothing here. It is last in the importance ranking too.

Note the importance is computed on a slice of the **training** split. The test
set was spent on scores; an importance plot from it would be a second look at a
one-shot check.
"""
)

md(
    """
## What did not work

Kept deliberately, because a walkthrough where everything succeeds is not a
description of doing this work.

1. **The classic preprocessing leak is not measurable here** — −0.135%, a
   fraction of the fold noise, with a sign that flips. The leak that bites is a
   leaked *column*.
2. **No ensemble beats any other.** Four models within one arm's own fold
   noise. The decision was made on artefact size and fit time.
3. **Stacking scored worse than its best member**, at 12× the cost.
4. **`housing_median_age` contributes almost nothing** — flat partial
   dependence, last in importance.
5. **A test suite that passed while the search was entirely broken** — it tried
   only the first value of each parameter, and fitted on `X.iloc[:200]`, whose
   `RangeIndex` accidentally aligned with the bug's output.
6. **One block group with 6 households and 7,460 people** took a CV fold to
   RMSE 1,805,055 before the ratios were clipped.

## Summary

| | |
| --- | --- |
| Best model | LightGBM, tuned jointly with the preprocessing |
| Test RMSE | **42,251** (CV said 42,166) |
| Test MAE | 26,677 |
| Test R² | 0.90 |
| Uncensored RMSE | 38,939 |
| Censored RMSE | 83,570 (R² −5.1 — and necessarily so) |
| Artefact | 6.8 MB, 4.4 s to fit |

Full source, 240 tests, the decision records and every measurement above:
"""
    + REPO
)


# =============================================================================
# Write it out
# =============================================================================


def build() -> pathlib.Path:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # `\n` at the end and no trailing spaces: the file is compared byte for byte
    # by the CI staleness gate, on a different OS from the one that wrote it.
    OUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8", newline="\n")
    return OUT


if __name__ == "__main__":
    path = build()
    code_cells = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"{path}: {len(cells)} cells ({code_cells} code, {len(cells) - code_cells} markdown)")
