# M3-S1 — CV harness, metrics and baselines

**Built.** `evaluate.py` (metrics, including the segment split ADR-001 obliged
us to), `models.py` (registry + harness + the shared preprocessing search
space), and ADR-003.

## The baselines, measured on the training split

```
          RMSE            MAE        R²     train RMSE
dummy    118,322 ± 1,786   88,119   -0.055     118,330
linear    63,094 ± 2,004   45,261    0.700      62,872
ridge     63,089 ± 1,993   45,265    0.700      62,875
```

Three things worth reading off that table.

**The dummy's R² is negative.** It predicts the training *median*, and R² is
defined against the *mean*. A median predictor is optimal for MAE and slightly
worse than the mean for a squared-error score, so −0.055 is correct, not a bug.
Worth knowing before it appears in the notebook and alarms somebody.

**Ridge and linear are indistinguishable** — 63,089 vs 63,094, on a fold spread
of ±2,000. The penalty is doing nothing yet. That is a fact about *these 36
features*, not about regularisation, and it will change when M3-S2 starts
searching encoder choices that widen the matrix.

**Train and test RMSE are within 0.3% of each other.** These models are not
overfitting; they are underfitting. A linear model on this data leaves
something on the table, which is exactly the gap M3-S2 exists to close.

## The concept: write down what the one-shot check freezes, before running it

ADR-003 exists because the held-out test set can be scored **once**. Scoring it
twice and keeping the better number is not a smaller sin than tuning on it — it
is the same sin spread over two afternoons.

So the ADR fixes, now, before any tuned model exists: the exact arms the single
run will contain (five), what each records (the full segment table, not a
headline), and what the run makes unfalsifiable (the split, ADR-002's encoder
choices, the clipper bounds, the search budget). Anything a reader might later
want compared has to ride **inside** that one run.

It also names a limitation rather than burying it: neighbouring block groups
are spatially correlated, so random folds are mildly optimistic. Group folds
would fix that and would make every number incomparable with published results
on this dataset. Recorded as a known limitation for the model card, not
silently adopted or silently ignored.

## Two failure modes a search space has, both now tested

A key that names nothing is accepted by `set_params` at definition time and
only raises once `RandomizedSearchCV` starts fitting — minutes in, several arms
deep. Worse, the imputer entries are **objects**, not strings: a search sets
pipeline *steps*, so `"median"` would be rejected while `"passthrough"` would
be quietly accepted and let the 207 nulls through to an estimator that cannot
take them.

`test_search_space_paths_all_resolve` checks every key against the real
pipeline's parameters, and `test_every_search_space_value_is_settable` actually
fits with each first value. Both are cheap; the failure they prevent is not.

## F-005 — the fixture reproduced the dataset's shape but not its learnability

The synthetic target was `rng.uniform(20_000, 480_000)`. Pure noise.

Every test so far had been about *transformers*, where that was fine. The first
test about a **model** — "ridge must beat the median predictor" — failed against
completely correct code, because on a random target nothing can beat the median.

The fixture now derives the target from income, latitude and crowding plus
noise: `corr(target, median_income) = 0.91`. The cap, the five-row level and
the null structure are all preserved, and the tests still assert *relationships*
(a model clears the floor; the inland segment can look worse) and never
coefficients.

This is the second time a fixture gap surfaced only when a new kind of test
arrived (F-002 was the missing skew). The pattern is worth naming: **a synthetic
fixture is only as good as the properties the current tests happen to need**,
and each new layer of the system finds the next missing one.

## What to look at

`ADR-003`, specifically the "What the one test run will freeze" list. Then
`models.py::cross_validate_pipeline` — note that it takes a model and wraps it,
so no caller can accidentally cross-validate a pre-transformed matrix.

## What to try

```
uv run python -c "
from calhousing.data import load_raw
from calhousing.splits import split_train_test
from calhousing.models import cross_validate_pipeline
from calhousing import config
tr, _ = split_train_test(load_raw(verbose=False))
X, y = tr.drop(columns=[config.TARGET]), tr[config.TARGET]
for m in ('dummy','linear','ridge'):
    r = cross_validate_pipeline(m, X, y)
    print(f\"{m:8s} RMSE {r['rmse']:>9,.0f} +/- {r['rmse_std']:>6,.0f}  R2 {r['r2']:.4f}\")"
```
