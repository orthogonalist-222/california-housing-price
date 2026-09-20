# M3-S2 — Tuned tree ensembles

**Built.** `train.py` — `RandomizedSearchCV` over the whole pipeline — and two
new registry entries, `rf` and `hgb`.

## The results

Five-fold CV on the training split, seed 42. The test set is still untouched.

| model | CV RMSE | fold std | train RMSE | fit time | draws |
| --- | --- | --- | --- | --- | --- |
| ridge (M3-S1) | 63,089 | ±1,993 | 62,875 | 0.1s | — |
| **hgb** | **43,860** | ±1,712 | 21,592 | 5.2s | 25 |
| **rf** | **43,352** | ±1,236 | 16,108 | 24.0s | 20 |

Both ensembles beat the linear baseline by **~31%**. RF edges HGB by 508 RMSE —
inside the fold spread, so "RF is better" is not a claim this evidence
supports — while costing **4.6× the fit time**. That trade matters: M4 has a
Kaggle kernel budget.

Note the reversal from M3-S1. The linear models *underfit* (train and test
within 0.3%). The ensembles *overfit* — RF's train RMSE is 2.7× better than its
CV RMSE. Both are reported, always, because a leaderboard showing only the test
column hides which of the two is happening.

## The point of the story: the search reaches the preprocessing

```
[hgb] best CV RMSE 43,860
        model__l2_regularization   = 1.0
        model__learning_rate       = 0.05
        model__max_iter            = 400
        model__max_leaf_nodes      = 127
        model__min_samples_leaf    = 5
        preprocess__binned__bin__n_bins = 8
        preprocess__geo__gamma          = 0.1
        preprocess__geo__n_clusters     = 20
        preprocess__heavy__deskew       = PowerTransformer
        preprocess__heavy__impute       = SimpleImputer
```

Five of those ten winning parameters are **preprocessing** decisions. The
number of geographic clusters, the RBF width, the bin count and the de-skew
transform were chosen by the same cross-validated evidence that chose the
learning rate.

A pipeline assembled by hand out of `fit_transform` calls can tune the model.
It cannot tune the preprocessing, because by the time the model is being tuned
the preprocessing has already happened and its choices are baked into the
matrix. **That is the practical argument for the discipline** — and it is a
much better one than the leakage argument, which measured 0.05% on this dataset
(M2-S4).

Both models also converged on `n_clusters=20` and `n_bins=8`, the top of both
ranges — a signal that the ranges are too narrow, recorded here so M3-S3 can
widen them rather than rediscover it.

## The segment table earns its place

HGB on the training split:

```
               rmse      mae    r2      n
all         20258.9  13426.8   1.0  16512
uncensored  19422.8  13256.9   1.0  15746
censored    33051.9  16918.8  -0.2    766
```

**R² of −0.2 on the censored rows.** The model is worse than predicting their
mean — which is exactly what ADR-001 predicted and the reason it required this
split. A model cannot be right about a row whose true value is unknown and
above the cap. A single headline number would have averaged that away.

## F-006 — the search failed on every draw, and my test had hidden it twice

The first search run died immediately:

```
ValueError: Concatenating DataFrames from the transformer's output lead to an
inconsistent number of samples. The output may have Pandas Indexes that do not
match...
```

Raised from inside `ColumnTransformer`, naming neither the branch nor the step.

**Cause.** `build_numeric_block` calls `set_output(transform="pandas")` on the
*Pipeline*, which configures the steps that exist **at that moment**. A search
that replaces a step — `preprocess__heavy__impute=SimpleImputer()` — installs
an estimator nobody configured. It returns a bare ndarray, the branch loses its
index, and the pandas concat produces the wrong number of rows.

**Fix.** `numeric._pandas` configures every factory output at construction, so
anything a factory returns is safe to drop into a pipeline.

**The part worth keeping.** `test_every_search_space_value_is_settable` existed,
passed, and was useless — for two independent reasons:

1. It tried only `values[0]`, so two of the three imputers were never fitted.
2. It fitted on `X.iloc[:200]`. That slice has a `RangeIndex` of 0..199 — and a
   bare ndarray gets re-framed with a *default* `RangeIndex*`. **The broken
   output aligned by accident.** On a real stratified split, whose index is not
   0..n-1, the same code raised.

A test that *samples* its rows would have caught this. A test that takes the
head of the frame cannot. The test now fits every value on a shuffled index and
asserts up front that the index is not a `RangeIndex`, because that assertion is
the entire point of the test.

`error_score="raise"` is in `run_search` for the same family of reason: the
default is `np.nan`, which turns a broken configuration into a merely unlucky
one and lets a search report a winner while silently discarding half its budget.

## What to look at

`train.py` — the two lines that merge the model's space with the pipeline's,
and the `_test` variable in `main` that is deliberately unused and deliberately
named. Then `numeric._pandas`.

## What to try

```
uv run python -m calhousing.train --model hgb --n-iter 25
```

Watch the winning parameters. Then delete `_pandas` from one factory and run it
again.
