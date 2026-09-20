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
