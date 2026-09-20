# M1-S3 — EDA module and figures · **closes M1**

**Built.** `src/calhousing/eda.py`: seven figures, one command, byte-identical
on a rerun.

## The figures are not a gallery

Each function is named for the claim it supports, and each claim is cashed in
later:

| Figure | Claim | Where it is spent |
| --- | --- | --- |
| `01-target-censoring` | The cap is a wall, not a tail — 965 rows at `$500,001`, 4 at the `$14,999` floor | ADR-001; the censored/uncensored error split in M3-S1 |
| `02-ocean-proximity-rarity` | One level has five rows | the split's level collapse; `handle_unknown` in M2-S2 |
| `03-geography` | Price is spatial | earns the KMeans-similarity feature in M2-S3 |
| `04-correlations` | The four raw counts move together, not with price | why M2-S3 uses ratios instead of raw totals |
| `05-missingness` | Nulls are spread across levels, not concentrated | a simple imputer is defensible; M2-S1 still offers alternatives |
| `06-heavy-tails` | Skew 3.4–4.9, and `log1p` fixes it | the de-skew option in M2-S1 is not speculative |
| `07-income-bands-and-split` | The band mix survives the split; every level reaches both sides | M1-S2's design, shown rather than asserted |

A plot nobody acts on is decoration. If a later story deletes one of these
claims, the figure goes with it.

## The concept: a recipe that cannot reproduce its own output is not a recipe

Artifacts are cattle — `artifacts/` is gitignored and rebuilt on demand. That
is only *true* if the rebuild produces the same thing, and it does not by
default:

> **matplotlib writes its own version string into every PNG.**

So two builds of an identical figure differ in bytes, and after an unrelated
dependency upgrade a determinism gate would fail for a reason that has nothing
to do with the data. `_save` passes `metadata={"Software": None}` to strip it,
and `test_matplotlib_version_is_not_baked_into_the_png` asserts the absence
directly — the red-team for this story is naming the byte that would break it.

Measured: two consecutive builds on the real file, all seven SHA-256 digests
identical.

```
01-target-censoring.png        sha256:e59ec6e12888
02-ocean-proximity-rarity.png  sha256:c21910784bfa
03-geography.png               sha256:d67b13a6b31a
04-correlations.png            sha256:9fc254286627
05-missingness.png             sha256:927be47ea31a
06-heavy-tails.png             sha256:11f3205db81e
07-income-bands-and-split.png  sha256:9f43516df611
```

The style dictionary is fixed in the module rather than inherited from a
matplotlibrc, for the same reason: a per-user config would make one machine's
output differ from another's.

## A figure must not depend on the dataset it was written against

`test_figures_survive_a_frame_with_no_rare_level` builds everything from a
frame with **no** `ISLAND` rows. `fig_rare_category` colours levels below a
threshold red, and `fig_income_and_split` builds a coverage table — both would
happily assume the island exists. A figure that only works on one file is a
figure that will break the first time the pipeline is pointed anywhere else.

## What to look at

`eda.py::_save` — four lines, one of which is the entire determinism story.
Then `test_eda.py::test_rebuild_is_byte_identical`.

## What to try

```
uv run python -m calhousing.eda
uv run python -m calhousing.eda
```

The digests it prints are the same both times. Now delete the
`metadata={"Software": None}` argument and run the pair again.

---

# M1 — milestone close

**Delivered.** A repo that refuses bad input at three doors, and knows what its
data looks like.

| Story | Delivered | Evidence |
| --- | --- | --- |
| M1-S1 | Scaffold, CI, layout gate, story-lint | 4 CI checks green; RT-001, RT-002 |
| M1-S2 | Schema contract, leakage-safe split, ADR-001 | 35 tests; RT-003, RT-004 |
| M1-S3 | Seven deterministic figures | 43 tests; digests match across rebuilds |

**Accept-when for M1, all observed:** `uv sync --dev` provisions on 3.11 and
3.12; `pytest` green; the layout gate and story-lint each seen refusing a
planted defect; `load_raw()` returns 20 640 × 10 with exactly 207 nulls in one
column and five `ISLAND` rows; the split is reproducible and covers every level
on both sides at every seed tried; `python -m calhousing.eda` writes seven
figures whose bytes do not change on a rerun.

**Findings register:** no `S1` or `S2` open. Nothing crosses this boundary.

**Two things M1 changed about the plan**, both recorded where they will be
read again:

1. The split design (ADR-001's neighbour, `splits.py`'s docstring) — the
   approved plan's stratification would have passed its own acceptance
   criterion by luck.
2. The target is censored at **both** ends (ADR-001).

**Release: `v0.1.0`.**

**Next: M2**, the milestone this project exists for — impute, encode, engineer,
assemble.
