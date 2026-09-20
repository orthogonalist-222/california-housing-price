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
