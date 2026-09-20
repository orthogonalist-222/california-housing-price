# Sign-off log

Every gate crossing: what was produced, who approved it, the verdict, and the
**evidence** — the observation that proved it, not the intention behind it.
Append-only; a row is never rewritten.

Roles in this project: `data scientist`, `ML engineer`, `tech lead`,
`model validator` (fresh-session assurance only — no self-sign-off).

| Date | Story | Gate | Producer | Approver | Verdict | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-20 | M1-S1 | `uv sync --dev` provisions | tech lead | tech lead | PASS | Python 3.12.13, scikit-learn 1.7.2, pandas 2.3.3, numpy 2.5.3 |
| 2026-09-20 | M1-S1 | `uv run pytest -q` | tech lead | tech lead | PASS | 6 passed |
| 2026-09-20 | M1-S1 | Layout gate, control | tech lead | tech lead | PASS | `layout gate OK: 14 tracked files, all declared.` (exit 0) |
| 2026-09-20 | M1-S1 | Layout gate, **red-team** | tech lead | tech lead | PASS | See RT-001 below |
| 2026-09-20 | M1-S1 | Story-lint, **red-team** | tech lead | tech lead | PASS | See RT-002 below |
| 2026-09-20 | M1-S2 | `uv run pytest -q` | data scientist | data scientist | PASS | 35 passed (3 real-file tests ran locally; they skip on CI) |
| 2026-09-20 | M1-S2 | Layout gate | data scientist | data scientist | PASS | `layout gate OK: 22 tracked files, all declared.` |
| 2026-09-20 | M1-S2 | Schema contract, **red-team** | data scientist | data scientist | PASS | See RT-003 below |
| 2026-09-20 | M1-S2 | Split coverage invariant | data scientist | data scientist | PASS | See RT-004 below |
| 2026-09-20 | M1-S3 | `uv run pytest -q` | data scientist | data scientist | PASS | 43 passed |
| 2026-09-20 | M1-S3 | Layout gate | data scientist | data scientist | PASS | `layout gate OK: 25 tracked files, all declared.` |
| 2026-09-20 | M1-S3 | Figure determinism | data scientist | data scientist | PASS | See RT-005 below |
| 2026-09-20 | **M1** | **Milestone gate** | data scientist | data scientist | **PASS** | All three stories' Accept-when observed; findings register has no open S1/S2. Release `v0.1.0`. |
| 2026-09-20 | M2-S1 | `uv run pytest -q` | data scientist | data scientist | PASS | 87 passed |
| 2026-09-20 | M2-S1 | Layout gate | data scientist | data scientist | PASS | `layout gate OK: 29 tracked files, all declared.` |
| 2026-09-20 | M2-S1 | log1p domain guard, **red-team** | data scientist | data scientist | PASS | See RT-006 below; finding F-001 |
| 2026-09-20 | M2-S1 | Leakage invariant on one block | data scientist | data scientist | PASS | See RT-007 below |
| 2026-09-20 | M2-S2 | `uv run pytest -q` | data scientist | data scientist | PASS | 113 passed |
| 2026-09-20 | M2-S2 | Layout gate | data scientist | data scientist | PASS | `layout gate OK: 32 tracked files, all declared.` |
| 2026-09-20 | M2-S2 | Unseen-category handling, **red-team** | data scientist | data scientist | PASS | See RT-008 below |
| 2026-09-20 | M2-S2 | `drop='first'` collision, **red-team** | data scientist | data scientist | PASS | See RT-009 below |
| 2026-09-20 | M2-S3 | `uv run pytest -q` | data scientist | data scientist | PASS | 134 passed |
| 2026-09-20 | M2-S3 | Layout gate | data scientist | data scientist | PASS | `layout gate OK: 36 tracked files, all declared.` |
| 2026-09-20 | M2-S3 | Centroids are train-only | data scientist | data scientist | PASS | See RT-010 below |
| 2026-09-20 | M2-S3 | Zero-denominator guard | data scientist | data scientist | PASS | See RT-011 below |
| 2026-09-20 | M2-S4 | `uv run pytest -q` | data scientist | data scientist | PASS | 155 passed |
| 2026-09-20 | M2-S4 | Layout gate | data scientist | data scientist | PASS | `layout gate OK: 41 tracked files, all declared.` |
| 2026-09-20 | M2-S4 | Leakage measurement | data scientist | data scientist | **PASS, with a corrected claim** | See RT-012 below |
| 2026-09-20 | M2-S4 | Institutional-outlier regression | data scientist | data scientist | PASS | See RT-013 below; finding F-004 |
| 2026-09-20 | **M2** | **Milestone gate** | data scientist | data scientist | **PASS** | All four stories' Accept-when observed; F-001..F-004 closed, no open S1/S2. Release `v0.2.0`. |
| 2026-09-20 | M3-S1 | `uv run pytest -q` | data scientist | data scientist | PASS | 176 passed |
| 2026-09-20 | M3-S1 | Layout gate | data scientist | data scientist | PASS | `layout gate OK: 46 tracked files, all declared.` |
| 2026-09-20 | M3-S1 | Baselines cross-validated | data scientist | data scientist | PASS | dummy 118,322 / linear 63,094 / ridge 63,089 - see RT-014 |
| 2026-09-20 | M3-S1 | ADR-003 freezes the one-shot run **before** it happens | data scientist | data scientist | PASS | Five arms named, segment table required, unfalsifiable list written |
| 2026-09-20 | M3-S2 | `uv run pytest -q` | data scientist | data scientist | PASS | 192 passed |
| 2026-09-20 | M3-S2 | Layout gate | data scientist | data scientist | PASS | `layout gate OK: 49 tracked files, all declared.` |
| 2026-09-20 | M3-S2 | Ensembles beat the linear baseline | data scientist | data scientist | PASS | hgb 43,860 / rf 43,352 vs ridge 63,089 - see RT-015 |
| 2026-09-20 | M3-S2 | Search reaches preprocessing, not just the model | data scientist | data scientist | PASS | 5 of 10 winning params are `preprocess__*` - RT-015 |
| 2026-09-20 | M3-S3 | `uv run pytest -q` | data scientist | data scientist | PASS | 213 passed |
| 2026-09-20 | M3-S3 | Layout gate | data scientist | data scientist | PASS | 52 tracked files, all declared |
| 2026-09-20 | M3-S3 | xgb and lgbm fit inside the pipeline | data scientist | data scientist | PASS | See RT-017 (feature-name interop) |
| 2026-09-20 | M3-S3 | Fair four-arm comparison | data scientist | data scientist | **PASS, with a negative result** | See RT-018 |
| 2026-09-20 | M3-S4 | `uv run pytest -q` | data scientist | data scientist | PASS | 240 passed |
| 2026-09-20 | M3-S4 | Layout gate | data scientist | data scientist | PASS | `layout gate OK: 55 tracked files, all declared.` |
| 2026-09-20 | M3-S4 | **The one-shot test-set run** | data scientist | data scientist | **PASS** | Five arms, one invocation, seed 42 - see RT-019. Record: `artifacts/final/test-set-evaluation.json` |
| 2026-09-20 | M3-S4 | One-shot guard, **red-team** | data scientist | data scientist | PASS | Second run refused with exit 2, record unchanged; `--rerun` still works |
| 2026-09-20 | **M3** | **Milestone gate** | data scientist | data scientist | **PASS** | All four stories observed; F-001..F-006 closed, no open S1/S2. Release `v0.3.0`. |
| 2026-09-20 | M4-S1 | `uv run pytest -q` | ML engineer | ML engineer | PASS | 251 passed |
| 2026-09-20 | M4-S1 | Layout gate | ML engineer | ML engineer | PASS | `layout gate OK: 60 tracked files, all declared.` |
| 2026-09-20 | M4-S1 | Notebook builds reproducibly | ML engineer | ML engineer | PASS | 37 cells; two builds byte-identical |
| 2026-09-20 | M4-S1 | Notebook **runs** end to end | ML engineer | ML engineer | PASS | 16 code cells in 124s - see RT-020 |
| 2026-09-20 | M4-S1 | Staleness gate, **red-team** | ML engineer | ML engineer | PASS | See RT-021, including a failed first attempt |
| 2026-09-20 | M4-S2 | `uv run pytest -q` | ML engineer | ML engineer | PASS | 266 passed |
| 2026-09-20 | M4-S2 | Layout gate | ML engineer | ML engineer | PASS | `layout gate OK: 65 tracked files, all declared.` |
| 2026-09-20 | M4-S2 | **Both install paths exercised** | ML engineer | ML engineer | PASS | See RT-022 - each installed into a clean venv |
| 2026-09-20 | M4-S2 | Staging refuses broken wiring, never uploads | ML engineer | ML engineer | PASS | See RT-023 |

## Red-team records

A safeguard that has never been seen rejecting anything is an assumption. Each
record holds **both halves**: the planted defect and the observed refusal, then
the removal and the restored pass.

### RT-001 — the layout gate refuses, distinguishably (2026-09-20)

Planted: `src/calhousing/scratch.py` (undeclared) and `kaggle.json` (a fake
credential file), both force-staged — `.gitignore` refused `kaggle.json` first,
so `git add -f` was needed to reach the gate at all.

Observed:

```
FORBIDDEN TRACKED PATH: kaggle.json
    Kaggle credentials must never be committed
UNDECLARED TRACKED PATH: src/calhousing/scratch.py
    Not covered by DECLARED in tools/check_layout.py. ...

layout gate FAILED: 1 forbidden, 1 undeclared, out of 16 tracked files.
exit=1
```

Two distinct signatures, as designed. Both files removed from the index and
from disk; the gate then reported `layout gate OK: 14 tracked files, all
declared.` with exit 0.

### RT-002 — the story-lint refuses, both failure modes (2026-09-20)

The CI job's shell logic run locally against four inputs:

| Input | Result |
| --- | --- |
| branch `docs/m3-kickoff` (ad-hoc prefix) | REFUSED — not `feat/m<M>-s<S>-<slug>` |
| branch `feat/m1-s1-scaffold`, title `M1-S2: wrong id` | REFUSED — title does not start with `M1-S1: ` |
| branch `feat/m1-s1-scaffold`, title `M1-S1: Repo scaffold, CI and protocol docs` | OK |
| branch `feat/m4-s2.1-hotfix`, title `M4-S2.1: fix the kernel` | OK — point stories are permitted |

The first row is the failure this gate exists for: the sibling program's
`docs/m3-kickoff`-style PRs are precisely what the protocol calls "a unit that
has left the map".

### RT-003 — the schema contract refuses, with three different messages (2026-09-20)

Three defects planted into a synthetic frame; the observed refusals:

```
missing column: MissingColumnsError: Missing required column(s): ['median_income']. Expected [...]
wrong dtype:    ColumnTypeError: Column(s) with the wrong dtype: total_rooms (got object, expected numeric)
out of range:   ValueRangeError: Value(s) outside the expected range: latitude: 1 row(s) outside [32.0, 43.0] (observed 45.6 .. 45.6)
```

Three classes, three cures, one shared base (`SchemaError`) so a caller can use
one `except`. Pinned by `tests/test_data.py::test_the_three_refusals_are_distinguishable`
and `::test_checks_run_most_fundamental_first`.

A fourth refusal is deliberately **not** a `SchemaError`: `DataNotFoundError`
for a missing file. An absent file and a malformed one have nothing in common
except that both stop the run.

### RT-004 — the split's rare-category guarantee (2026-09-20)

**Planted defect: the design the plan asked for.** Stratifying on income band
alone, measured on the real file:

| seed | ISLAND train | ISLAND test |
| --- | --- | --- |
| 0 | 4 | 1 |
| 42 | **2** | 3 |
| 7 | **5** | **0** |
| 2024 | 3 | 2 |

Seed 7 leaves the test set with no island. The acceptance criterion ("ISLAND
appears in train") passed at seed 42 **by luck**.

**Second rejected design:** stratify on `income_band x ocean_proximity` —
scikit-learn raises `The least populated class in y has only 1 member`, because
`band 3 x ISLAND` has exactly one row.

**Shipped design:** collapse per level. Observed on the real file, seeds
0 / 42 / 7 / 2024 — `ISLAND train=4 test=1` in **all four**. Both rejected
designs are held in place by tests
(`test_naive_band_stratification_is_the_lottery_this_replaces`,
`test_composite_key_without_collapse_is_unsplittable`) so that a future
simplification has to confront them.

### RT-005 — figure determinism, and the byte that breaks it (2026-09-20)

**Control.** Two consecutive `uv run python -m calhousing.eda` runs on the real
file produced identical SHA-256 digests for all seven PNGs:

```
01-target-censoring.png        sha256:e59ec6e12888
02-ocean-proximity-rarity.png  sha256:c21910784bfa
03-geography.png               sha256:d67b13a6b31a
04-correlations.png            sha256:9fc254286627
05-missingness.png             sha256:927be47ea31a
06-heavy-tails.png             sha256:11f3205db81e
07-income-bands-and-split.png  sha256:9f43516df611
```

**The planted defect is named rather than merely removed.** matplotlib writes
its own version string into PNG metadata by default, so an identical figure
built before and after a dependency upgrade differs in bytes.
`_save` passes `metadata={"Software": None}`, and
`test_eda.py::test_matplotlib_version_is_not_baked_into_the_png` asserts the
string is absent from the output — the guard fails the moment someone deletes
that argument, which is the only way this defect can return.

**Second control:** `test_figures_survive_a_frame_with_no_rare_level` builds
every figure from a frame with no `ISLAND` rows, so no figure may assume the
dataset it was written against.

### RT-006 — a transform that returned NaN instead of raising (2026-09-20)

**Not a planted defect — a real one, found by a test.** `deskew="log"` over the
whole numeric frame produced no error and a frame of NaNs, because `longitude`
is about −124 and `np.log1p` is undefined at or below −1. The only signal:

```
RuntimeWarning: invalid value encountered in divide
  T = new_sum / new_sample_count      (sklearn/utils/extmath.py:1149)
```

A warning from inside scikit-learn, naming neither the column nor the step.

**Fixed** by `_checked_log1p`, which refuses:

```
DomainError: log1p is undefined at or below -1, and would return NaN for:
longitude (min -124). Route these columns through a block with
deskew='none' or 'yeo-johnson' instead.
```

**Both halves pinned.** `test_log_on_an_out_of_domain_column_is_refused_by_name`
asserts the refusal fires and names `longitude`;
`test_the_guard_does_not_fire_on_valid_input` asserts it does **not** fire on
the positive count columns — a guard that refuses correct input is worse than
no guard. A third test pins that NaN is not read as "below −1". Registered as
F-001.

### RT-007 — leakage as a property of a single block (2026-09-20)

**Control.** Fit the block on train; transform the test frame whole, and
transform single rows alone. The two agree to 1e-12. A block whose transform
depended on the other rows being scored would disagree.

**The vacuity check.** That test is only meaningful if the fitted statistics
*can* move. `test_fitted_statistics_come_from_train_only` fits on train, then on
train plus wildly different rows, and asserts the imputer's learned fill values
**differ** — with an explicit failure message saying the first test cannot
detect leakage if they do not.

### Note on two findings in our own tests (2026-09-20)

F-002 and F-003 are defects in the test suite, not the code, and both were
exposed by an unrelated fixture change rather than by review. They are logged
here at the same weight as everything else: a synthetic fixture that lost the
property it existed to reproduce, and two M1-S2 tests that relied on a lucky
draw instead of planting what they asserted — the exact era-fact trap those
tests were written to warn about.

### RT-008 - a fit that never saw the island (2026-09-20)

**Planted defect: scikit-learn's own default.** `handle_unknown="error"` fitted
on an island-free frame, then handed an `ISLAND` row:

```
ValueError: Found unknown categories ['ISLAND'] in column 0 during transform
```

On this dataset that is a pipeline which trains happily and dies at scoring
time on five rows out of 20 640.

**Control.** Both shipped encoders transform the same row without raising,
parametrized over `onehot` and `ordinal`, and repeated against the real file's
actual five island rows (skipped on CI).

**A measurement that changed the design.** `infrequent_if_exist` is a no-op
without `min_frequency`, and `min_frequency` only creates the infrequent bucket
when a **training** category falls below the threshold:

| fit data | infrequent column? | unknown row |
| --- | --- | --- |
| contains the 4-row island | yes | `[0,0,0,0,1]` sum 1.0 - a presence |
| island-free | **no** | `[0,0,0,0]` sum 0.0 - an absence |

So the encoder's unknown-handling depends on M1-S2's split guarantee. Pinned by
`test_the_infrequent_bucket_needs_the_rare_level_at_fit_time`, whose whole job
is to fail if that guarantee is ever removed.

### RT-009 - `drop='first'` merges unknown into the baseline (2026-09-20)

**Planted defect, run with the guard bypassed:**

```
drop='first', handle_unknown='ignore'
dropped reference level = '<1H OCEAN'
unknown 'ISLAND'    -> [[0.0, 0.0]]
known   '<1H OCEAN' -> [[0.0, 0.0]]
IDENTICAL: True
```

**Refusal observed:**

```
UnsafeEncoderError: drop='first' with handle_unknown='ignore' encodes an
unknown category identically to the dropped reference level - the two become
the same vector and nothing downstream can separate them. Use drop=None ...
```

**Both halves, plus the sunset condition.**
`test_drop_is_allowed_when_unknowns_are_impossible` asserts the guard does not
ban the *safe* use of `drop`, and `test_the_collision_the_guard_prevents_is_real`
demonstrates the defect directly, with a failure message stating that if
scikit-learn ever stops collapsing these, the guard has become unnecessary and
should be deleted.

**Noted asymmetry:** scikit-learn warns about unknown categories **only when
`drop` is set**. The safe configuration is silent and the unsafe one is loud, so
warnings cannot be relied on to surface this.

### RT-010 - ClusterSimilarity centroids come from train only (2026-09-20)

**Control.** Fit on 400 rows, record the centroids, transform the held-out rows
and then the whole frame. `np.testing.assert_array_equal` on the centroids
before and after: unchanged.

**Vacuity check, run as its own test.** Fitting on train+test must produce
DIFFERENT centroids from fitting on train alone - otherwise the control above
detects nothing and is quietly worthless. `test_fitting_on_everything_moves_the_centroids`
asserts they differ.

This pairing is now the house pattern for every leakage test in the repo
(see also RT-007): a control, plus a test that the control is capable of
failing.

### RT-011 - the zero-denominator guard, with honest provenance (2026-09-20)

**The dataset does not motivate this guard.** Measured on the raw file:
`households == 0` in 0 rows, `total_rooms == 0` in 0 rows. The guard exists to
avoid depending on a property of one CSV, not to fix an observed defect, and
the test says so in its own docstring rather than implying otherwise.

**Planted defect:** synthetic rows with a zero `households` and a zero
`total_rooms`. Observed: every ratio finite, the affected cells equal to the
declared fill rather than `inf`.

**The other half:** `test_the_unguarded_division_really_does_produce_inf`
asserts plain division still yields `inf` here. If it ever stopped,
`safe_divide` would be unnecessary and should be deleted.

### RT-012 - the leakage red-team, and the claim it corrected (2026-09-20)

The approved plan's Accept-when expected that fitting the preprocessing outside
the fold would produce "a measurably better CV score". **It does not**, and the
measurement is recorded rather than the expectation:

```
A. Preprocessing inside the fold vs. fitted once on everything
   honest  RMSE     63,811  +/- 1,430
   leaked  RMSE     63,897  +/- 1,631
   leak                -86  (-0.135% of honest)
   -> the leak is 0.06x the fold-to-fold spread
```

The leaky variant is **worse**, well inside the noise, and shrinking the
training set produces no trend - the sign flips between n=300 and n=1 000.

**The red-team that does bite**, same script, part C:

```
   clean frame                       RMSE     63,811
   leaky column, through assembler   RMSE     63,811
   leaky column, forced past it      RMSE      5,121
```

`remainder="drop"` silently discarded the target-derived column - the top two
numbers are the same run. The assembler is a whitelist, and that is the leak
defence with a large measured payoff.

**Consequence for the tests.** `tests/test_assemble.py` asserts the mechanism -
whitelist, refit-per-fold, row-independence - and never a score. Numbers live in
`tools/measure_leakage.py`, which anyone can rerun and argue with. Full write-up
in `docs/pyspark-to-sklearn.md`.

### RT-013 - an institutional block group blows up a linear model (2026-09-20)

**Not planted. Found by running the assembled pipeline**, which is the whole
argument for exercising a deliverable as its consumer would - every individual
block was correct and no unit test would have caught this.

Ridge over a 5 000-row subsample:

```
per-fold RMSE: [61772, 69583, 63860, 65713, 1805055]
```

Cause, measured on the raw file: the ratios are unbounded and four rows are
institutions rather than neighbourhoods.

```
population_per_household   med 2.818   p99.9 13.6   max 1243.3
rooms_per_household        med 5.229   p99.9 34.2   max  141.9
```

The largest is 6 households and 7 460 people. Real data.

**Fix:** `QuantileClipper`, winsorising at bounds learned from the TRAINING rows
- one more stateful step fitted inside the fold. Clipping beats dropping (those
block groups exist and have a price) and beats `RobustScaler` (which rescales
the outlier but leaves it as far away).

**After:** `[61694, 69169, 63222, 65349, 72352]`, and full-data Ridge improved
from 65 838 to 63 811 with the fold spread down from +/-1 868 to +/-1 430.
Pinned by `test_an_institutional_block_group_does_not_blow_up_predictions`,
which asserts predictions stay inside a sane multiple of the target range
rather than pinning a number. Registered as F-004.

### RT-014 - baselines on the training split (2026-09-20)

Five-fold CV over the whole pipeline, seed 42, training split only. The test
set has not been touched.

```
          RMSE            MAE        R2      train RMSE
dummy    118,322 +/- 1,786   88,119   -0.055     118,330
linear    63,094 +/- 2,004   45,261    0.700      62,872
ridge     63,089 +/- 1,993   45,265    0.700      62,875
```

Three readings recorded so they are not rediscovered later:

- **The dummy's R2 is negative and that is correct.** It predicts the training
  MEDIAN; R2 is defined against the MEAN. A median predictor is optimal for MAE
  and slightly worse than the mean under squared error.
- **Ridge and linear are indistinguishable** (63,089 vs 63,094 on a +/-2,000
  spread). The penalty is doing nothing on these 36 features yet.
- **Train and test RMSE agree to 0.3%.** These models underfit. That gap is what
  M3-S2 exists to close.

**Search-space guards, both halves.** A key that names no real step is accepted
at definition time and raises only once a search starts fitting - minutes in.
`test_search_space_paths_all_resolve` checks every key against the real
pipeline's parameters, and `test_every_search_space_value_is_settable` fits with
each first value. The imputer entries are OBJECTS rather than strings for a
related reason: a search sets pipeline steps, so `"median"` would be rejected
while `"passthrough"` would be quietly accepted and let the 207 nulls reach an
estimator that cannot take them.

### RT-015 - tuned ensembles, and the search that reaches the preprocessing (2026-09-20)

Five-fold CV on the TRAINING split, seed 42. The test set remains untouched.

| model | CV RMSE | fold std | train RMSE | fit time | draws |
| --- | --- | --- | --- | --- | --- |
| ridge (M3-S1) | 63,089 | +/-1,993 | 62,875 | 0.1s | - |
| hgb | **43,860** | +/-1,712 | 21,592 | 5.2s | 25 |
| rf | **43,352** | +/-1,236 | 16,108 | 24.0s | 20 |

Both beat the linear baseline by ~31%. RF edges HGB by 508 RMSE, which is
INSIDE the fold spread - "RF is better" is not a claim this evidence supports -
while costing 4.6x the fit time.

Reversal from M3-S1 worth recording: the linear models underfit (train and test
within 0.3%); the ensembles overfit (RF's train RMSE is 2.7x better than its
CV). Both columns are reported in every leaderboard for exactly this reason.

**The claim under test - the search is joint.** HGB's winning draw:

```
model__l2_regularization        = 1.0
model__learning_rate            = 0.05
model__max_iter                 = 400
model__max_leaf_nodes           = 127
model__min_samples_leaf         = 5
preprocess__binned__bin__n_bins = 8
preprocess__geo__gamma          = 0.1
preprocess__geo__n_clusters     = 20
preprocess__heavy__deskew       = PowerTransformer
preprocess__heavy__impute       = SimpleImputer
```

**Five of ten winning parameters are preprocessing decisions**, chosen by the
same cross-validated evidence that chose the learning rate. A hand-rolled
fit_transform chain cannot do this: by the time the model is tuned the
preprocessing is already baked into the matrix.

**Noted for M3-S3:** both models converged on `n_clusters=20` and `n_bins=8`,
the top of both ranges. The ranges are too narrow; widen rather than rediscover.

**ADR-001's obligation paying off.** HGB, training-split segments:

```
               rmse      mae    r2      n
uncensored  19422.8  13256.9   1.0  15746
censored    33051.9  16918.8  -0.2    766
```

R2 of -0.2 on the censored rows - worse than predicting their mean, exactly as
ADR-001 anticipated. A headline number would have averaged that away.

### RT-016 - F-006: a test that passed while the search was entirely broken

**Not planted.** The first search run raised on every draw:

```
ValueError: Concatenating DataFrames from the transformer's output lead to an
inconsistent number of samples. The output may have Pandas Indexes that do not
match...
```

from inside `ColumnTransformer`, naming neither the branch nor the step.

**Cause.** `build_numeric_block` calls `set_output` on the Pipeline, configuring
the steps present AT THAT MOMENT. A search replacing a step installs an
estimator nobody configured; it returns a bare ndarray and the branch loses its
index.

**Why the existing test missed it, twice over:**

1. It fitted only `values[0]`, so two of three imputers were never tried.
2. It fitted on `X.iloc[:200]` - a `RangeIndex` 0..199. A bare ndarray is
   re-framed with a DEFAULT `RangeIndex`, so the broken output **aligned by
   accident**. On a real stratified split the same code raised.

**Fixed** by `numeric._pandas`, which configures every factory output at
construction. The test now fits EVERY value on a SHUFFLED index and asserts up
front that the index is not a `RangeIndex` - that assertion is the point of the
test. `test_a_factory_result_is_safe_to_drop_into_a_pipeline` pins the root
cause at its source.

Related hardening: `run_search` sets `error_score="raise"`. The default is
`np.nan`, which turns a broken configuration into a merely unlucky one and lets
a search report a winner while silently discarding half its budget.

### RT-017 - XGBoost refuses the dataset's own category label (2026-09-20)

**Not planted.** Registering `xgb` broke the suite immediately:

```
ValueError: feature_names must be string, and may not contain [, ] or <
```

The name is `categorical__ocean_proximity_<1H OCEAN`. The `<` comes from the
DATASET's category label, through the one-hot encoder. Every pipeline branch is
innocent; the library refuses the character.

**Rejected:** handing XGBoost a bare ndarray (works, and throws away the
readable names the pipeline exists to preserve - M3-S4's importance plot would
read `f17`), and renaming the category in `config` (edits the data to suit a
library).

**Shipped:** `SanitiseFeatureNames`, between the assembler and the model, which
rewrites the NAME (`<` -> `lt`) so `ocean_proximity_lt1H OCEAN` stays readable.
It **refuses collisions** rather than merging - two features sharing a name is
worse than the character being avoided - and there is a test for that refusal.
`feature_names()` now reports `pipeline[:-1]`, so what it returns is what the
model actually saw.

### RT-018 - the four-arm comparison, and its negative result (2026-09-20)

**The re-run was the point.** M3-S2's two winners both pinned the TOP of
`n_clusters` and `n_bins`. M3-S3 widened those ranges, which left `rf` and `hgb`
searched over a DIFFERENT space from `xgb` and `lgbm` - comparing them would
have measured the search space, not the model. Both were re-run (about fifty
minutes of wall clock). `rf` moved 43,352 -> **42,707**; `hgb` 43,860 -> 43,571.
Under the old ranges `rf` looked worse than it is.

| model | CV RMSE | fold std | train RMSE | winning fit | artefact |
| --- | --- | --- | --- | --- | --- |
| ridge | 63,089 | +/-1,993 | 62,875 | 0.1s | - |
| hgb | 43,571 | +/-1,445 | 23,131 | 5.0s | 3.3 MB |
| rf | 42,707 | +/-1,178 | 15,852 | 115.5s | **289.4 MB** |
| xgb | 42,401 | +/-1,360 | 4,996 | 20.3s | 13.5 MB |
| lgbm | **42,166** | +/-1,407 | 9,052 | **5.1s** | 6.8 MB |

**The negative result, stated plainly: no ensemble is better than any other.**
The four span 1,405 RMSE while their own fold standard deviations run +/-1,178
to +/-1,445. The whole spread is about one arm's own fold noise. Crowning
LightGBM on 42,166 vs 42,401 would be reading noise.

**What the evidence DOES separate is cost.** RandomForest is 289 MB and 115s
per fit against LightGBM's 6.8 MB and 5.1s for an indistinguishable score - 43x
the storage and 23x the time for nothing measurable. On a kernel with a runtime
budget and a dataset size limit that is the entire decision, and it is why
M3-S4 builds around LightGBM.

**Still not converged, and said so:** `rf` pinned `n_clusters=45`, the top of
the WIDENED range. Not chased - the arms are within noise, so another widening
buys a better-looking number for one arm and no new knowledge. Recorded as a
limitation instead.

**A warning deliberately left in.** `KBinsDiscretizer` reports removing
zero-width bins for `housing_median_age` (52 distinct values, 1,273 rows piled
on 52). Benign - the effective bin count is data-dependent - but not suppressed,
because the honest reading is that `n_bins` above ~12 is a request the data
cannot always satisfy, and a silenced warning is a fact nobody rediscovers.

### RT-019 - the single test-set evaluation (2026-09-20)

**Run once.** Seed 42, 16,512 train / 4,128 test, five arms in one invocation,
exactly the list ADR-003 fixed before any of them existed. Record committed at
`artifacts/final/test-set-evaluation.json` (regenerable, gitignored; the
numbers below are the evidence).

| arm | test RMSE | MAE | R2 | fit+score |
| --- | --- | --- | --- | --- |
| **lgbm** | **42,251** | 26,677 | 0.90 | 4.4s |
| stack | 42,623 | 26,815 | 0.90 | 52.0s |
| rf | 43,177 | 27,113 | 0.89 | 42.0s |
| ridge | 65,142 | 45,540 | 0.68 | 0.1s |
| dummy | 119,750 | 89,324 | -0.06 | 0.2s |

**The verdict on stacking: it did not earn its complexity.** The stack scored
42,623 - WORSE than its own best member, at 12x the fit time. Four learners
that agree within their own fold noise (RT-018) give a meta-learner nothing to
arbitrate. It is the best arm INLAND (30,186 vs 30,783), which is genuine,
small, and not enough to change the recommendation - but it is exactly the kind
of detail a headline erases, which is why the segment table is mandatory.

**The CV protocol held.** CV RMSE 42,166 -> test RMSE 42,251, a gap of 0.2%.
The winner of a 25-draw joint search generalised almost exactly as its
cross-validation said it would - checkable only because the test set was
untouched until this run.

**ADR-001 vindicated with numbers.** LightGBM by segment:

```
               rmse      mae    r2     n
uncensored  38939.4  25299.9   0.8  3925
censored    83570.0  53299.8  -5.1   203
```

Censored rows are 2.1x worse with R2 = -5.1 - worse than predicting their own
mean, and necessarily so. The dummy's censored R2 is -87.8. Averaging 5% of the
test set into one headline would have hidden the whole story.

**Interpretation, computed on a TRAINING slice** (ADR-003 spends the test set
on scores; an importance plot from it would be a second look):

```
         longitude   62,919      total_rooms   48,027
     median_income   59,250       population   38,986
          latitude   55,028       households   37,361
   ocean_proximity   31,328   total_bedrooms   17,832
                             housing_median_age  15,781
```

Geography outranks income. That the RAW coordinates rank so high AFTER twenty
cluster-similarity columns were engineered from them says those columns did not
exhaust the spatial signal. Partial dependence: `median_income` drives price
monotonically from ~175,000 to ~318,000; `housing_median_age` is FLAT, and last
in the ranking.

**The one-shot guard, both halves.** A second invocation was refused:

```
REFUSING: artifacts/final/test-set-evaluation.json already records a run at ...
The test set is a one-shot check (ADR-003). A second run that replaces the
first is indistinguishable from tuning on it.
```

exit code 2, and `test_a_second_run_is_refused` additionally asserts the
original record is UNCHANGED afterwards - a guard that refuses and then
corrupts what it protects is not a guard. `--rerun` exists deliberately: a
guard with no escape hatch gets worked around by deleting the file, which
leaves no trace, whereas `--rerun` is explicit and greppable.

### RT-020 - the notebook runs, and what running it found (2026-09-20)

Building is not running. `tools/build_notebook.py` compiles every cell, so a
syntax error cannot ship - but compiling proves nothing about a `KeyError` on
cell 14. `tools/run_notebook.py` executes the real file in order.

**Result:** 16 code cells, **124 seconds**, no failures.

**It found a defect in itself first.** `run_notebook.py` skipped the install
cell by searching sources for the substring `"pip"` - which also matches
**`tuned_pipeline`** in the IMPORTS cell. The imports were skipped and the
notebook died three cells later on `NameError: name 'load_raw' is not defined`,
pointing at a cell that was completely fine.

Fixed with an explicit marker, `# tag:install`, matched exactly. The script now
REFUSES if it does not find exactly one tagged cell - silently skipping zero or
two is the same class of failure in a different hat. Pinned by
`test_there_is_exactly_one_tagged_install_cell` and
`test_the_tag_does_not_match_the_imports_cell`.

Also confirmed separately: the package installs cleanly from GitHub into a
fresh 3.11 venv, which is the path the notebook's install cell takes on Kaggle.

### RT-021 - the staleness gate, and a red-team that was wrong first (2026-09-20)

**First attempt failed to prove anything, and looked like a pass.** Sequence
was: hand-edit the notebook, run `build_notebook.py`, check `git diff`. Result:
"GATE FAILED TO NOTICE".

The gate was fine. The red-team was backwards - the build REGENERATES the file,
so it overwrote the planted edit before anything inspected it. What that
actually tested was that the generator is deterministic, which was already
known.

**Lesson worth more than the gate: a red-team that runs the repair before the
check proves nothing, and is indistinguishable from a passing test.**

**Correct sequence** - plant, then check BEFORE rebuilding:

```
FAILED tests/test_notebook_build.py::test_the_committed_notebook_matches_its_generator
E   At index 113 diff: b'H' != b'C'
```

`H` from "Hand-edited", `C` from "California". Restored afterwards; the gate
passes again.

**Supporting properties, each tested:** stable content-hashed cell ids (random
ids would fail the gate on every run and train everyone to ignore it), no
committed outputs or execution counts, and every code cell in the shipped
artefact compiles.

**One more guard worth naming.**
`test_the_negative_results_survive_into_the_notebook` asserts the notebook
still contains each finding that contradicted the plan - the -0.135% leak, "no
ensemble is better than any other", the stacking verdict, the whitelist, and
`housing_median_age`. They are the most deletable content in the project and
the most valuable.

### RT-022 - both install paths, installed rather than asserted (2026-09-20)

"The fallback works" is the easiest sentence in this project to write and the
easiest to be wrong about. Both were run.

| Path | Verification |
| --- | --- |
| GitHub | `uv pip install git+https://...` into a FRESH 3.11 venv; imported `calhousing`, `TUNED`, `build_pipeline` |
| Offline wheel | Fresh 3.11 venv with only the RUNTIME deps preinstalled (Kaggle's shape), then `pip install --no-index --find-links <dir> <wheel>` with NO network index |

Observed:

```
clean venv with Kaggle-like preinstalled deps
=== FALLBACK PATH: --no-index --find-links, no network index ===
install rc=0
FALLBACK OK: calhousing 0.1.0 | arms ['hgb', 'lgbm', 'rf', 'ridge', 'xgb'] | pipeline builds
```

`--no-index` is what makes the second row mean anything - pip cannot reach PyPI
at all, so a missing dependency fails loudly instead of being quietly fetched
from the network whose absence was being simulated.

**The compile guard fired on its first real outing.** Writing the install cell
produced:

```
cell 2 does not compile: unterminated string literal (detected at line 33)

 32 |         raise SystemExit(
 33 |             "Could not install calhousing.
 34 | "
```

A backslash-n written one layer up, in the generator's f-string, became a REAL
newline inside a string literal in the generated cell - exactly the defect
`build_notebook.code()`'s docstring predicts, caught before the notebook went
near Kaggle. Fixed by removing the need to escape at all: the message prints
line by line and the `SystemExit` carries a single-line string.

### RT-023 - the staging tool refuses, and never uploads (2026-09-20)

**Planted defect:** metadata in which the kernel does not attach the dataset
the fallback reads. Observed refusal:

```
SystemExit: kernel-metadata.json does not attach 'someone/calhousing-src' as a
dataset source, so the offline fallback could never find the wheel.
```

**The absence of an upload is asserted, not assumed.**
`test_staging_does_not_upload` monkeypatches `subprocess.run` to raise and
asserts it is never called. Publishing to a live Kaggle account is
outward-facing; a tool that does it as a side effect of "staging" will one day
publish something nobody meant to.

**Two brittleness bugs the tests found in the tool itself:**

1. `Path.relative_to` RAISES on a path outside its argument, so the tool
   crashed outright when pointed at a directory elsewhere. Now `_display()`.
2. It staged before validating - the attachment check ran after building a
   wheel and copying a notebook, leaving half-staged files behind on a
   misconfiguration. Validation moved first.

Both found by writing tests for a script, which is the argument for `tools/`
being importable at all.
