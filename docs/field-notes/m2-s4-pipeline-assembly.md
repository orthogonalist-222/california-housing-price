# M2-S4 — Assembly: ColumnTransformer + full Pipeline · **closes M2**

**Built.** `preprocess/assemble.py` — six branches into one `ColumnTransformer`,
then `build_pipeline(estimator)` returning a single estimator that preprocesses
and predicts. Plus `docs/pyspark-to-sklearn.md` and
`tools/measure_leakage.py`.

## The headline: I measured the leak, and the folklore is wrong

The plan's acceptance criterion said a pipeline that imputes before splitting
would produce "a measurably better CV score". It does not.

```
A. Preprocessing inside the fold vs. fitted once on everything
   honest  RMSE     63,811  +/- 1,430
   leaked  RMSE     63,897  +/- 1,631
   leak                -86  (-0.135% of honest)
   -> the leak is 0.06x the fold-to-fold spread
```

The "leaky" version is **worse**, by an amount buried in noise. Shrinking the
training set produces no trend either — the sign flips between n=300 and
n=1000. On this dataset, with a random split and one column 1% missing, fitting
the preprocessing outside the fold costs nothing you can measure.

I could have written the acceptance criterion as satisfied. Two runs and a
favourable seed would have produced a positive number. **Say the measurement,
not the expected measurement** — a reader told to expect something dramatic who
then measures 0.05% concludes the whole lesson was theatre, and stops believing
the parts that are true.

The lesson survives in a stronger form:

```
C. A column derived from the target, added to the input
   clean frame                       RMSE     63,811
   leaky column, through assembler   RMSE     63,811
   leaky column, forced past it      RMSE      5,121
```

**The large leak is a leaked column, and what stops it is that the assembler is
a whitelist.** `remainder="drop"` means an undeclared column never reaches the
model however target-shaped it is — the top two numbers are literally the same
run, because `appraisal` was silently discarded. `remainder="passthrough"`, one
word different, is how such a column arrives without anybody choosing to put it
there.

So on a dataset like this the pipeline's payoff is not the score. It is that
the feature set a model can see is **declared in one place, reviewable, and
enforced**. The fold discipline is what keeps that true when the data is
smaller, shifted, or has a stateful step with more to learn — and it costs
nothing to keep.

The tests were rewritten around that conclusion: they assert the *mechanism*
(whitelist, refit-per-fold, row-independence) and never a score. Scores live in
the measurement script where they can be rerun and argued with.

## The defect assembly exposed: one block group with 6 households and 7,460 people

Assembling the branches and running Ridge over a 5,000-row subsample:

```
per-fold RMSE: [61772, 69583, 63860, 65713, 1805055]
```

One fold, twenty-eight times worse than its neighbours. The cause: the
engineered ratios are **unbounded**, and this dataset contains institutions
rather than neighbourhoods.

```
population_per_household   min 0.692   med 2.818   p99.9 13.6   max 1243.3
rooms_per_household        min 0.846   med 5.229   p99.9 34.2   max  141.9
```

Four rows in the whole file. The largest is 6 households and 7,460 people — a
prison or a dormitory. Real data, not an error. Standard-scaled, that row is a
z-score in the hundreds and Ridge extrapolates from it catastrophically.

`QuantileClipper` winsorises at bounds **learned from the training rows**,
which makes it one more stateful step that has to be fitted inside the fold —
thematically perfect for this story. Clipping beats dropping (those block
groups exist and have a price; removing them changes the population being
modelled) and beats `RobustScaler` (which rescales the outlier but leaves it
just as far away).

After the fix, on the same subsample: `[61694, 69169, 63222, 65349, 72352]`.
And full-data Ridge improved from **65,838 to 63,811**, with the fold spread
down from ±1,868 to ±1,430.

That is what "exercise the deliverable as its consumer would" buys. No unit
test of any single block would have found it — each block was individually
correct.

## What to look at

`tools/measure_leakage.py`, then `docs/pyspark-to-sklearn.md`, then the
docstring of `QuantileClipper`.

## What to try

```
uv run python tools/measure_leakage.py
```

Then open `assemble.py`, change `remainder="drop"` to `"passthrough"`, and run
it again. Part C is the one that moves.

---

# M2 — milestone close

**Delivered.** The pipeline this project exists for.

| Story | Delivered | Evidence |
| --- | --- | --- |
| M2-S1 | Numeric block, and a log that refuses instead of returning NaN | 87 tests; F-001, F-002, F-003 |
| M2-S2 | Label vs one-hot, measured against sklearn rather than assumed | 113 tests; RT-008, RT-009 |
| M2-S3 | Ratios and geographic cluster similarity | 134 tests; RT-010, RT-011 |
| M2-S4 | Assembly, and the leakage measurement | 155 tests; RT-012, RT-013 |

**Accept-when for M2, all observed.** `build_pipeline()` returns one `Pipeline`
ending in an estimator; `get_feature_names_out()` names every one of its 36
output columns, branch-prefixed; the leak red-team ran and is reported with the
numbers it produced rather than the numbers expected; the PySpark→sklearn map
is written.

**Findings register:** F-001 … F-004 all closed. No `S1` or `S2` crosses this
boundary.

**Three things M2 changed about the plan**, each recorded where it will be read
again:

1. `deskew="log"` silently returned NaN on `longitude` (F-001) — now a named
   refusal, and the reason `assemble` has separate `heavy` and `plain`
   branches.
2. `infrequent_if_exist` is a no-op without `min_frequency`, and the infrequent
   bucket only exists when the rare level is present at fit time — so the
   encoder depends on M1-S2's split guarantee (ADR-002).
3. The preprocessing leak is **not measurable** on this dataset; the leak that
   matters is a leaked column, and the whitelist is what stops it
   (`docs/pyspark-to-sklearn.md`).

**Release: `v0.2.0`.**

**Next: M3** — baselines, tuned ensembles, boosting, stacking, and exactly one
test-set run.
