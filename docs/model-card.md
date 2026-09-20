# Model card — California housing price, LightGBM pipeline

- **Version:** v1.0.0
- **Date:** 2026-09-20
- **Owner:** data scientist (see `docs/sign-off-log.md` for every gate crossing)

## What it is

A scikit-learn `Pipeline` that takes the nine raw columns of the 1990
California census extract and predicts `median_house_value`. The estimator is
LightGBM; the preprocessing was tuned jointly with it.

The artefact is **6.8 MB** and fits in **4.4 seconds**.

## Intended use

**Educational.** This project exists to demonstrate a preprocessing pipeline —
the same discipline as a Spark ML pipeline, rebuilt in scikit-learn. It is a
worked example of *how to build and check* such a pipeline.

**It is not a house-price valuation tool.** The data is a 1990 census extract
describing **block groups** (600–3,000 people), not individual properties, and
1990 California prices have no bearing on any current market.

## Headline results

The test set was scored **exactly once** (ADR-003). Five arms, one invocation,
seed 42, 16,512 train / 4,128 test. Full record:
[`docs/evidence/test-set-evaluation.json`](evidence/test-set-evaluation.json).

| arm | test RMSE | MAE | R² |
| --- | --- | --- | --- |
| **lgbm** (shipped) | **42,251** | 26,677 | 0.90 |
| stack | 42,623 | 26,815 | 0.90 |
| rf | 43,177 | 27,113 | 0.89 |
| ridge | 65,142 | 45,540 | 0.68 |
| dummy (median) | 119,750 | 89,324 | −0.06 |

Cross-validated RMSE on the training split was **42,166**. The test result is
**0.2%** away from it.

### By segment — read this before quoting the headline

| segment | RMSE | MAE | R² | n |
| --- | --- | --- | --- | --- |
| all | 42,251 | 26,677 | 0.90 | 4,128 |
| uncensored | 38,939 | 25,300 | 0.83 | 3,925 |
| **censored** | **83,570** | 53,300 | **−5.1** | 203 |
| inland | 30,783 | 18,796 | 0.80 | 1,309 |
| coastal | 46,627 | 30,337 | 0.85 | 2,819 |

## Limitations

**1. The target is censored, and the model cannot be right about those rows.**
965 rows in the full dataset sit at exactly `$500,001` and 4 at `$14,999`. The
true values are unknown — above the cap and below the floor. On the censored
test rows the model scores **R² of −5.1**, worse than predicting their own
mean, and necessarily so.

ADR-001 chose to keep those rows and report them separately. **Consequence: the
headline RMSE is worse than a number produced by dropping them, and is not
comparable to notebooks that do.**

**2. Random folds are mildly optimistic.** Neighbouring block groups are
spatially correlated, and both the train/test split and the CV folds are random
(stratified on income band). Grouping folds geographically would be more
honest about generalising to a *new region* — and would make every number
incomparable to published results on this dataset. ADR-003 records the choice.
**The scores here describe interpolation within California, not extrapolation
to a new one.**

**3. The comparison cannot separate the ensembles.** RandomForest,
HistGradientBoosting, XGBoost and LightGBM span 1,405 CV RMSE while their own
fold standard deviations run ±1,178 to ±1,445. **No ensemble is better than any
other on this evidence.** LightGBM was chosen on cost: 6.8 MB and 5.1s against
RandomForest's 289 MB and 115s.

**4. Stacking did not help.** The stacked ensemble scored *worse* than its best
member at roughly 12× the fit time. It is better inland (30,186 vs 30,783) and
worse overall.

**5. `n_clusters` is not converged.** The geographic-similarity feature's
cluster count hit the top of its search range twice, including after the range
was widened. The number of clusters is a truncated search, not an optimum.

**6. One feature contributes almost nothing.** `housing_median_age` is last in
permutation importance and its partial dependence is flat.

**7. The data is old and coarse.** 1990, block-group level, California only.

## How it was built

| Stage | What | Where |
| --- | --- | --- |
| Split | Stratified on income band, with rare `ocean_proximity` levels collapsed so the 5-row `ISLAND` level reaches **both** sides at every seed | `splits.py` |
| Impute | Median (searched against mean / KNN / iterative) | `preprocess/numeric.py` |
| De-skew | Yeo-Johnson on the four count columns (skew 3.4–4.9) | `preprocess/numeric.py` |
| Encode | One-hot with `min_frequency=10`, so an unknown level is a **presence** rather than an absence; ordinal alternative uses an explicit distance-from-water ordering | `preprocess/categorical.py` |
| Bin | `KBinsDiscretizer` on income and age | `preprocess/categorical.py` |
| Engineer | Three ratios, quantile-clipped; KMeans+RBF geographic similarity | `preprocess/features.py` |
| Assemble | Six-branch `ColumnTransformer`, `remainder="drop"` | `preprocess/assemble.py` |
| Tune | `RandomizedSearchCV` over **preprocessing and model jointly**, 25 draws × 5 folds | `train.py` |

Five of the ten winning parameters are preprocessing decisions.

## What the model uses

Permutation importance over the raw columns, on a **training** slice (the test
set was spent on scores):

```
         longitude   62,919      total_rooms   48,027
     median_income   59,250       population   38,986
          latitude   55,028       households   37,361
   ocean_proximity   31,328   total_bedrooms   17,832
                          housing_median_age   15,781
```

Geography outranks income. That the *raw* coordinates rank so high **after**
twenty cluster-similarity columns were engineered from them suggests those
columns did not exhaust the spatial signal.

## Ethical notes

Block-group price models built on census geography can encode historical
residential segregation: `ocean_proximity`, latitude and longitude are proxies
for neighbourhood, and 1990 California neighbourhood boundaries carry the
legacy of redlining. **The three strongest features in this model are
geographic.** Anything of this shape used for a real decision — lending,
insurance, assessment — would need a fairness analysis this project does not
contain.

## Reproducing it

```bash
uv sync --dev
kaggle datasets download -d camnugent/california-housing-prices -p data/raw --unzip
uv run pytest -q
uv run python tools/final_evaluation.py
```

The last command **refuses** if a record already exists — the test set is a
one-shot check. Read `docs/evidence/test-set-evaluation.json` instead.

**Independent reproduction, 2026-09-20:** a clean clone of the repository, with
nothing but committed code, reproduced every headline number to the decimal —
`lgbm 42251.1`, `stack 42622.5`, `rf 43176.6`, `ridge 65141.6`,
`dummy 119749.8`.
