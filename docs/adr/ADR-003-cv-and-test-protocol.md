# ADR-003 — The cross-validation protocol, and what the single test run freezes

- **Status:** accepted
- **Date:** 2026-09-20
- **Story:** M3-S1
- **Deciders:** data scientist

## Context

The held-out test set is a **one-shot check**. Once it is scored, every choice
baked into the thing being scored becomes unfalsifiable — and the evidence that
would have condemned one of those choices usually arrives afterwards. Scoring
it twice and keeping the better number is not a smaller sin than tuning on it;
it is the same sin, spread over two afternoons.

So the rules and the contents of that run are written down **now**, in M3-S1,
before any model beyond a baseline exists.

## The cross-validation protocol

| Decision | Value | Why |
| --- | --- | --- |
| Splitter | `KFold(n_splits=5, shuffle=True, random_state=42)` | One object, `models.build_cv`, shared by every arm. Two models scored under different folds are not comparable and the difference gets attributed to the model. |
| Stratified folds? | **No** | The target is continuous. The *train/test* boundary is stratified (M1-S2) because it is drawn once and matters; five shuffled folds of 16 512 rows reproduce the distribution closely enough that stratifying adds complexity and no signal. |
| Unit of scoring | **The whole `Pipeline`** | Never a pre-transformed matrix. `cross_validate_pipeline` wraps every estimator in `build_pipeline` so no caller has to remember. |
| Primary metric | RMSE | Comparable with published results on this dataset, and it penalises the large errors that matter for a price. |
| Secondary | MAE, R² | MAE because RMSE is dominated by the censored tail; R² for a scale-free read. |
| Reported alongside | Train score and fold std | A model whose train and test RMSE differ by 3× is overfitting regardless of how good the test number looks. |
| Segments | overall / uncensored / censored / inland / coastal | ADR-001's obligation. A model cannot be right about a censored row, and INLAND is a different price regime. |
| Seed | `config.RANDOM_SEED = 42` everywhere | One number. |

**All tuning, model selection and feature decisions use CV on the training
split only.** The test set is not looked at, plotted, or described until the
single run below.

## What the one test run will freeze

This is the list. Everything on it becomes unfalsifiable the moment the test
set is scored, so anything a reader might later want compared has to ride
**inside that one run**, measured together.

The run happens once, in **M3-S4**, and it evaluates **every arm together**:

1. `dummy` — the floor.
2. `ridge` — tuned linear baseline.
3. The best tuned tree ensemble from M3-S2.
4. The best gradient-boosting arm from M3-S3.
5. The stacked ensemble from M3-S4.

For each arm it records the full `evaluate.segment_table` — all five segments,
with `n` — not just a headline RMSE.

It also freezes, without further appeal:

- **The split.** Seed 42, stratified with the level collapse (M1-S2).
- **The preprocessing decisions.** ADR-002's encoder choices, the `QuantileClipper`
  bounds, the branch routing in `assemble.py`.
- **The censoring decision.** ADR-001: censored rows kept, reported separately.
- **The search budget.** Whatever M3-S2 and M3-S3 spent. A later "one more
  search" whose winner is then tested is a second test run wearing a disguise.

## Consequences

- If, after the test run, someone wants to know how a sixth arm would have
  done, the honest answer is *CV only* — and the model card must say the test
  number covers five arms, not six.
- The test set is scored by one function, once, and its output is committed as
  evidence in the M3-S4 PR. A second invocation is a finding, not a rerun.
- Because every arm rides inside the same run, the comparison between arms is
  valid even though each individual number is a single draw.

## Options considered

| Option | Why not |
| --- | --- |
| Nested CV, no held-out set | Statistically cleaner, and it removes the single artefact a reader can check. The Kaggle notebook needs one honest number at the end. |
| Repeated K-fold | Better variance estimate, ~5× the runtime. The kernel budget is the constraint; fold std is reported instead. |
| Score the test set per milestone | This is the failure the one-shot rule exists to prevent. |
| Group folds by geography | Defensible — neighbouring block groups are correlated, so random folds are mildly optimistic. **Noted as a known limitation** rather than adopted: it would change what the numbers mean relative to every published result on this dataset. Recorded in the model card. |

## Revisit trigger

Reopen **before** the M3-S4 run if a new arm is added to the list above, or if
the geographic-correlation limitation is judged serious enough to change the
splitter. Reopening **after** the run requires a new test set, not a new
decision.
