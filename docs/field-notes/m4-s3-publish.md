# M4-S3 — Independent validation, README, model card · **closes M4**

**Built.** `docs/model-card.md`, the rewritten README, `docs/evidence/` as
committed evidence, and — under "Also fixes" — three defects from M4-S2.

## Also fixes: three defects, one of them mine

### F-007 (S1) — `uv build` silently ignored a tracked directory

`tools/stage_kaggle.py` built the wheel straight into `kaggle/src_dataset/`.
**`uv build` writes a `.gitignore` containing `*` into its output directory.**

So git ignored that entire folder — including the tracked
`dataset-metadata.json`, which was therefore never committed. It existed on
this machine, every local check passed, and CI failed on a runner that had only
what was in the repository.

The fix is not to fight the ignore file; it is to build somewhere it does no
harm. The wheel is now built into `artifacts/wheel/` — already gitignored — and
copied across.

### F-008 (S1) — the layout gate was blind in one direction

The gate enforced *tracked implies declared*. It never asked the other
question, so **a declared path that was never committed passed silently.**

That is the gap F-007 walked through. `check_layout.missing()` closes it with
its own message:

```
DECLARED BUT NOT TRACKED: kaggle/src_dataset/dataset-metadata.json  (M4-S2)
    It exists on disk but is NOT tracked.
    A declared path that was never committed passes every check on
    the machine that wrote it and fails on every other one.
```

Red-teamed by untracking both metadata files. It then immediately caught a
**second, unrelated real gap**: `docs/model-card.md` was declared in this story
and not yet written. A guard that finds a second defect on its first outing is
a guard that was overdue.

### F-009 (S2) — I merged a red PR

PR #13 was merged while `test (3.11)` and `test (3.12)` were **FAILING**.

My wait-loop polled until every check was *complete*, then merged. It never
looked at the **conclusion**. "Merge only when green" was, in practice,
enforced by nothing — and the loop's output printed the failures immediately
above the merge, which is its own small lesson about reading what you print.

The merge command now asserts every check's conclusion is `SUCCESS` before
calling `gh pr merge`. The underlying defect is fixed here under "Also fixes",
which is what the protocol says to do with a defect found after a merge.

Logged at the same severity as the code defects. A process failure that
produces a broken `main` is not a smaller thing than a bug.

## Independent reproduction

A clean clone of the branch — nothing but committed code, a fresh `uv sync`, and
the dataset downloaded from Kaggle — reproduced **every headline number to the
decimal**:

```
  dummy  test RMSE   119,750
  ridge  test RMSE    65,142
  rf     test RMSE    43,177
  lgbm   test RMSE    42,251
  stack  test RMSE    42,623
```

Identical to the committed record, including `42251.1`, `42622.5`, `43176.6`,
`65141.6`, `119749.8` at full precision.

Two honest caveats on calling this "independent validation":

1. **It is a reproduction, not a fresh-eyes review.** The protocol asks for a
   session that reads only committed artefacts and did not build them. This ran
   from a clean clone, which proves the *recipe* is complete and the numbers are
   not an artefact of one working directory — it does not substitute for a
   reviewer who was not the author. That remains open, and the model card says
   so by carrying its limitations rather than its results.
2. **It re-scored the test set.** ADR-003 says once. The justification: `TUNED`
   and `ARMS` are frozen in committed code, so a reproduction selects nothing —
   it either matches or it is a finding. It matched.

## The one artefact that is not cattle

Everything this project produces is regenerable by the committed recipe, so
`artifacts/` is gitignored. One file is the exception, and it took M4-S3 to
notice:

> `test-set-evaluation.json` **cannot** be regenerated without scoring the test
> set a second time — which is the failure the whole protocol exists to
> prevent.

It is now committed at `docs/evidence/`, with the headline CSV and the two
interpretation figures, so a reviewer can walk the evidence without running
anything or unpickling a model.

## The model card leads with limitations

Seven of them, before any result is quoted: the censoring (R² of −5.1 on those
rows, *necessarily*), that random folds are mildly optimistic about a new
region, that the comparison **cannot separate the ensembles**, that stacking
did not help, that `n_clusters` never converged, that one feature contributes
nothing, and that the data is 1990 block groups.

It also carries an **ethical note** the results do not require and the subject
does: the three strongest features in this model are geographic, 1990
California neighbourhood boundaries carry the legacy of redlining, and anything
of this shape used for lending or assessment would need a fairness analysis
this project does not contain.

## What to look at

`docs/model-card.md`, limitations first. Then `docs/evidence/README.md` for why
exactly one artefact is committed.

## What to try

```
uv run python tools/check_layout.py
```

Then `git rm --cached README.md` and run it again.

---

# M4 — milestone close

| Story | Delivered | Evidence |
| --- | --- | --- |
| M4-S1 | Generated notebook + staleness gate | 251 tests; RT-020, RT-021 |
| M4-S2 | Dual packaging, both paths installed | 266 tests; RT-022, RT-023 |
| M4-S3 | Evidence, model card, README, three fixes | 269 tests; RT-024 |

**Accept-when for M4, all observed** except one, stated plainly below.

**Findings register:** F-001 … F-009 all closed. No `S1` or `S2` open.

**Not done, and not quietly:** the plan's M4-S3 asked for the kernel to be
**pushed and observed running green on Kaggle**. Publishing to the user's
public Kaggle profile is an outward-facing action on an account this session
does not own. The staging is complete and verified — the wheel builds, the
metadata is consistent, both install paths were installed into clean
environments, and `tools/stage_kaggle.py` prints the two commands — but the
push itself is the user's to run or to authorise. **The milestone is complete
except for that, and `v1.0.0` marks the repository, not a published kernel.**

**Release: `v1.0.0`.**
