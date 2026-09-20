# M3-S4 — Stacking, the one test run, and the verdict · **closes M3**

**Built.** A `StackingRegressor` arm, `interpret.py`, and
`tools/final_evaluation.py` — the script that spends the test set.

## The test set, scored once

Seed 42, 16,512 train / 4,128 test. Five arms, one invocation, exactly as
ADR-003 fixed **before any of them existed**.

| arm | test RMSE | MAE | R² | fit+score |
| --- | --- | --- | --- | --- |
| **lgbm** | **42,251** | 26,677 | 0.90 | 4.4s |
| stack | 42,623 | 26,815 | 0.90 | 52.0s |
| rf | 43,177 | 27,113 | 0.89 | 42.0s |
| ridge | 65,142 | 45,540 | 0.68 | 0.1s |
| dummy | 119,750 | 89,324 | −0.06 | 0.2s |

## The verdict on stacking: it did not earn its complexity

**The stack scored 42,623 — worse than its own best member.** LightGBM alone
got 42,251, in 4.4 seconds against the stack's 52.

That is a clean negative result and it is the answer to the question M3-S4 was
written to ask. Four base learners with genuinely different inductive biases,
a `RidgeCV` meta-learner that was free to downweight any of them, and the
combination lands *behind* the simplest member.

Why, most likely: RT-018 already showed the four ensembles were within their
own fold noise of each other. Learners that agree have nothing for a
meta-learner to arbitrate — it is fitting a weighted average of four nearly
identical prediction vectors, and paying an internal 5-fold cross-validation
for the privilege.

One thing the stack *did* do: **it is the best arm inland** (30,186 vs
LightGBM's 30,783), while being worse overall. Genuine, small, and not enough
to change the recommendation — but it is the kind of detail a headline number
erases, which is why the segment table is mandatory.

**Recommendation: ship LightGBM.** Best score, 6.8 MB, 4.4 seconds.

## The CV protocol held

CV RMSE was **42,166**. Test RMSE is **42,251** — a gap of 0.2%.

The searches ran 25 draws over a joint model-and-preprocessing space and the
winner generalised almost exactly as its cross-validation said it would. That
is the thing ADR-003's discipline buys, and it is only checkable *because* the
test set was untouched until this run.

## ADR-001 vindicated, with numbers

LightGBM, by segment:

```
               rmse      mae    r2     n
uncensored  38939.4  25299.9   0.8  3925
censored    83570.0  53299.8  -5.1   203
```

**The censored rows are 2.1× worse, with R² of −5.1** — the model is far worse
than predicting their own mean. It cannot be otherwise: their true value is
unknown and above the cap, so a prediction below is wrong by an unknowable
amount and a prediction above is penalised for being closer to the truth.

5% of the test set, and averaging it into a single headline would have hidden
the entire story. The dummy's censored R² is **−87.8**, which is what happens
when you predict a median for rows that are all at the maximum.

## What the model learned

Permutation importance over the raw columns (computed on a **training** slice —
ADR-003 spends the test set on scores, and an importance plot from it would be
a second look):

```
         longitude   62,919
     median_income   59,250
          latitude   55,028
       total_rooms   48,027
        population   38,986
        households   37,361
   ocean_proximity   31,328
    total_bedrooms   17,832
housing_median_age   15,781
```

**Geography outranks income.** `longitude` and `latitude` together dominate,
which is the `ClusterSimilarity` feature earning its place — the model is using
*where* far more than *what*. That the raw coordinates rank so high after the
geographic features were engineered from them says the twenty cluster columns
did not exhaust the spatial signal.

Partial dependence confirms the other half: `median_income` drives price
monotonically from ~175,000 to ~318,000 across its range, while
`housing_median_age` is **flat** — essentially no marginal effect, despite
being a column every tutorial keeps. It is last in the importance ranking too.

## The one-shot guard is real, not a note in a document

`tools/final_evaluation.py` **refuses to run twice**:

```
REFUSING: artifacts/final/test-set-evaluation.json already records a run at ...
The test set is a one-shot check (ADR-003). A second run that
replaces the first is indistinguishable from tuning on it.
```

There is a `--rerun` escape hatch, deliberately: a guard with no way out gets
worked around by deleting the file, which leaves no trace. `--rerun` is
explicit and greppable.

Both halves are tested. `test_a_second_run_is_refused` also asserts the
original record is **unchanged** after the refusal — a guard that refuses and
then corrupts what it was protecting is not a guard.

## A recipe that cannot rebuild itself is not a recipe

The tuned configurations live in `models.TUNED`, in **code**. The obvious
alternative — load `artifacts/search/*.joblib` — would have made the stack
depend on a gitignored directory that is empty on any fresh checkout.

`test_every_tuned_configuration_builds_and_fits` exists because a typo in those
transcribed numbers would otherwise be discovered *during the one-shot run*,
and that mistake costs the test set.

## What to look at

`tools/final_evaluation.py` — the refusal in `main`, and that `ARMS` is a
module constant rather than a CLI flag. A sixth arm added after the fact and
then tested is a second test run wearing a disguise.

## What to try

```
uv run python tools/final_evaluation.py
```

It refuses. Read `artifacts/final/test-set-evaluation.json` instead — the whole
record, readable without unpickling anything.

---

# M3 — milestone close

| Story | Delivered | Evidence |
| --- | --- | --- |
| M3-S1 | CV harness, segment metrics, baselines, ADR-003 | 176 tests; RT-014 |
| M3-S2 | Joint model+preprocessing search, RF and HGB | 192 tests; RT-015, RT-016 (F-006) |
| M3-S3 | XGBoost, LightGBM, a fair re-run | 213 tests; RT-017, RT-018 |
| M3-S4 | Stacking, the one test run, interpretation | 240 tests; RT-019 |

**Accept-when for M3, all observed.** Five arms cross-validated over the whole
pipeline; the search reached preprocessing as well as model parameters; the
test set was scored **exactly once**, all arms together, against ADR-003's
frozen list; permutation importance and two PDPs committed; the verdict on
stacking stated plainly.

**Findings register:** F-001 … F-006 all closed. No `S1` or `S2` crosses this
boundary.

**Three negative results M3 produced**, all kept:

1. **No ensemble is better than any other** (RT-018) — the four span less than
   one arm's own fold noise. The decision was made on cost.
2. **Stacking did not help.** Worse than its best member at 12× the cost.
3. **`housing_median_age` contributes almost nothing** — flat partial
   dependence, last in importance.

A milestone that reports three things that did not work is more useful than one
that reports four that did.

**Release: `v0.3.0`.**

**Next: M4** — the notebook, Kaggle packaging, independent validation, publish.
