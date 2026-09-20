# M1-S2 — Data contract, censoring ADR and leakage-safe split

**Built.** `config.py` (one source of truth for columns, seeds, ranges),
`data.py` (load + a schema contract with three distinguishable refusals),
`splits.py` (a stratified split that guarantees rare-category coverage), and
ADR-001 on the censored target.

## The interesting part: the split is not the obvious one

The plan said "stratified split on income band, and check `ISLAND` lands in
train". That is Géron's recipe and it is right about *why* — `median_income` is
the strongest single predictor, so a plain random holdout distorts everything
measured on top of it. But measured on the real file, the `ISLAND` coverage it
produces is a **lottery**:

| seed | ISLAND in train | ISLAND in test |
| --- | --- | --- |
| 0 | 4 | 1 |
| **42** | **2** | 3 |
| 7 | 5 | **0** |
| 2024 | 3 | 2 |

Seed 7 hands the test set no island at all. Seed 42 — our seed — happened to
satisfy the acceptance criterion with two rows. **It passed by luck.** An
acceptance gate that a different seed would fail is not a gate, it is a
coincidence with a checkmark next to it.

The obvious fix, stratifying on `income_band × ocean_proximity`, does not work
either: `band 3 × ISLAND` contains exactly **one** row, and scikit-learn
refuses outright —

```
ValueError: The least populated class in y has only 1 member, which is too few.
```

So `stratification_key` builds the composite key and then **collapses per
level**: if any composite group inside an `ocean_proximity` level is too small
to split, every row of that level is stratified by the level name alone.
ISLAND's five rows become one stratum and split **4 / 1 at every seed tried**.
Nothing else in the frame is coarsened — the common levels keep their band
component, and there is a test asserting exactly that, because a collapse that
quietly swallowed the whole key would silently stop stratifying on income and
nothing would fail loudly.

**Why per level and not per group.** Collapsing only the offending group would
leave that single band-3 island row in a stratum of one — the same refusal, one
step later. Pulling the whole level together is what makes the stratum large
enough to split. There is a test named for this.

## The concept: assert invariants, never era-facts

`tests/test_splits.py` asserts no row count and no score. It asserts that the
split partitions the data, that it is reproducible, that the band mix survives
on both sides, and that **every level reaches both sides at every seed**. Those
stay true when the code changes for good reasons. A test pinned to "ISLAND
train == 4" would fire on a correct change and teach the next session to edit
the assertion instead of the code.

The one test that *does* encode a failure — `test_naive_band_stratification_is_the_lottery_this_replaces`
— asserts only that **some** seed in a sweep fails, not which. If band-only
stratification ever became reliable, that test tells us the collapse is dead
code and should go.

## Three refusals, three cures

A contract that raises one generic error passes a test that only checks "it
raised". These are the observed messages:

```
MissingColumnsError: Missing required column(s): ['median_income']. Expected [...]
ColumnTypeError:     Column(s) with the wrong dtype: total_rooms (got object, expected numeric)
ValueRangeError:     Value(s) outside the expected range: latitude: 1 row(s) outside [32.0, 43.0] (observed 45.6 .. 45.6)
```

Checks run missing → dtype → range, most fundamental first. Telling someone
their latitudes look odd when the real problem is that they loaded the wrong
file sends them down the wrong path. There is a test for the ordering.

Note what is deliberately **not** a violation: `total_bedrooms` nulls. Missing
values are the imputer's job. If the range check ever started treating NaN as
out of range, every real file would be refused — so there is a test pinning
that boundary too.

## Also found

The target is censored at **both** ends — 4 rows at a \$14 999 floor, which the
plan did not mention. Recorded in ADR-001 rather than quietly ignored.

## What to look at

`src/calhousing/splits.py` — the module docstring carries the seed table, and
`stratification_key` is fifteen lines that took the measurement above to
justify. Then `tests/test_splits.py::test_collapse_is_per_level_not_per_group`.

## What to try

```
uv run python -c "
from calhousing.data import load_raw
from calhousing.splits import split_train_test, coverage_report
print(coverage_report(*split_train_test(load_raw(), seed=7)))"
```

Change the seed. The ISLAND row stays 4 / 1. Then edit `stratification_key` to
skip the collapse and run it again.
