# M3-S3 — XGBoost and LightGBM

**Built.** Two new registry entries, pinned dependencies, a feature-name
sanitiser, and re-runs of `rf` and `hgb` so the comparison is actually fair.

## The comparison, all four arms under the same search space

Five-fold CV on the training split, seed 42. The test set is still untouched.

| model | CV RMSE | fold std | train RMSE | winning fit | artefact |
| --- | --- | --- | --- | --- | --- |
| ridge | 63,089 | ±1,993 | 62,875 | 0.1s | — |
| hgb | 43,571 | ±1,445 | 23,131 | 5.0s | 3.3 MB |
| rf | 42,707 | ±1,178 | 15,852 | 115.5s | **289.4 MB** |
| xgb | 42,401 | ±1,360 | 4,996 | 20.3s | 13.5 MB |
| **lgbm** | **42,166** | ±1,407 | 9,052 | **5.1s** | 6.8 MB |

**The four ensembles span 1,405 RMSE. The fold standard deviations are ±1,178
to ±1,445.** The entire spread between best and worst is roughly one standard
deviation of a single arm's own folds. On this evidence, *no ensemble is
better than any other*, and a write-up that crowned LightGBM on 42,166 vs
42,401 would be reading noise.

What the evidence *does* separate is cost:

- **RandomForest is 289 MB and takes 115 seconds to fit.** LightGBM is 6.8 MB
  and 5.1 seconds, for a statistically indistinguishable score. That is 43×
  the storage and 23× the time for nothing measurable. On a Kaggle kernel with
  a runtime budget and a dataset size limit, that is the whole decision.
- **XGBoost's train RMSE is 4,996 against a CV of 42,401** — an 8.5× gap. It
  fits the training data almost perfectly and generalises like the others.
  Reported because a leaderboard showing only the CV column would make it look
  like a well-behaved model.

So LightGBM is the arm M3-S4 will build around — chosen on **cost at equal
accuracy**, not on the 235 RMSE that separates it from XGBoost.

## Re-running `rf` and `hgb` was not optional

M3-S2 searched `n_clusters ∈ {5,10,15,20}` and `n_bins ∈ {3,5,8}`. Both winners
landed on the **top** of both ranges, which M3-S2's field note flagged: *a
boundary winner means the search was cut off, not converged.*

So M3-S3 widened the ranges — and then `rf` and `hgb` had been searched over a
*different space* from `xgb` and `lgbm`. Comparing them would have measured the
search space, not the model. Both were re-run. It cost about fifty minutes of
wall clock and it is the difference between a comparison and four numbers in a
table.

The re-runs moved things: `rf` 43,352 → **42,707**, `hgb` 43,860 → 43,571.
Under the old ranges `rf` looked *worse* than it is.

**And it is still pinned.** `rf` chose `n_clusters=45`, the top of the widened
range, again. Recorded rather than chased: the arms are within noise of each
other, so another widening would buy a better-looking number for one arm and no
new knowledge. The honest statement is that `n_clusters` is not converged, and
that the feature count grows linearly with it while the score does not.

## The interop defect: XGBoost will not accept our feature names

```
ValueError: feature_names must be string, and may not contain [, ] or <
```

The offending name is `categorical__ocean_proximity_<1H OCEAN`. The `<` comes
from the **dataset's own category label**, through the one-hot encoder. Every
branch of the pipeline is innocent; the library simply refuses the character.

Two rejected fixes:

- **Hand XGBoost a bare ndarray.** Works, and throws away the readable names
  the whole pipeline exists to preserve. M3-S4's importance plot would read
  `f17`.
- **Rename the category in `config`.** Edits the data to suit a library.

`SanitiseFeatureNames` rewrites the *name* instead — `<` → `lt`, `>` → `gt`,
`[`/`]` → `(`/`)` — so `ocean_proximity_lt1H OCEAN` is still obvious to a
reader. It sits between the assembler and the model, because the name a model
sees has to satisfy that model's library.

It **refuses collisions** rather than silently merging: two features sharing a
name is a worse problem than the character being avoided.

`feature_names()` now returns `pipeline[:-1].get_feature_names_out()` — every
step except the estimator — so what it reports is what the model actually saw,
after sanitising rather than before.

## A warning I chose to leave in

```
UserWarning: Bins whose width are too small (i.e., <= 1e-8) in feature 1 are
removed. Consider decreasing the number of bins.
```

`housing_median_age` has 52 distinct values and 1,273 rows piled on 52 (itself
a censoring artefact). Ask for 16 quantile bins **inside a fold** and some come
back empty, so `KBinsDiscretizer` drops them.

This is benign — the effective bin count is simply data-dependent — but it is
not suppressed, because the honest reading is that `n_bins` above about 12 is
a request the data cannot always satisfy, and a silenced warning is a fact
nobody rediscovers.

## What to look at

`SanitiseFeatureNames` in `assemble.py`, then the artefact sizes above. The
289 MB is not a footnote — it is the argument.

## What to try

```
uv run python -m calhousing.train --model lgbm --n-iter 25
```

Then run `--model rf --n-iter 20` and compare the wall clock.
