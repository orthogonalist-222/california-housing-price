# California Housing Price — a scikit-learn preprocessing pipeline

**The preprocessing is the deliverable. The model is its passenger.**

A Spark ML pipeline rebuilt in scikit-learn — `StringIndexer` → `OneHotEncoder`
→ `VectorAssembler` → `Pipeline` → `CrossValidator` — on the 1990 California
census extract ([`camnugent/california-housing-prices`](https://www.kaggle.com/datasets/camnugent/california-housing-prices)).

Every design choice here was **measured before it was made**, and several of the
measurements contradicted the plan they came from. Those are kept.

| | |
| --- | --- |
| **Test RMSE** | **42,251** (CV said 42,166 — a 0.2% gap) |
| Test MAE / R² | 26,677 / 0.90 |
| Model | LightGBM, tuned jointly with the preprocessing |
| Artefact | 6.8 MB, 4.4 s to fit |
| Tests | 269, on Python 3.11 and 3.12 |
| Test set scored | **once** ([ADR-003](docs/adr/ADR-003-cv-and-test-protocol.md)) |

Full record: [`docs/evidence/`](docs/evidence/) ·
Caveats that matter: [`docs/model-card.md`](docs/model-card.md)

## Four things this project found

**1. The classic preprocessing leak is not measurable here.** Fitting the
preprocessing outside the fold costs **−0.135%** of RMSE — 0.06× the
fold-to-fold spread, with a sign that flips as the sample changes. The leak
that *does* bite is a leaked **column** (63,811 → 5,121), and what stops it is
`remainder="drop"` making the assembler a whitelist.

```bash
uv run python tools/measure_leakage.py   # read the numbers, not the folklore
```

**2. No ensemble is better than any other.** RandomForest, HistGradientBoosting,
XGBoost and LightGBM span 1,405 CV RMSE while their own fold standard
deviations run ±1,178 to ±1,445. The choice was made on **cost**: 6.8 MB and
5.1 s against RandomForest's **289 MB** and 115 s.

**3. Stacking did not earn its complexity.** It scored *worse* than its own best
member, at 12× the fit time.

**4. A model cannot be right about a censored row.** 965 rows sit at the
`$500,001` cap. On those, the model scores **R² of −5.1** — worse than
predicting their own mean, and necessarily so. Every evaluation reports the
segments separately; the headline alone would hide it.

## The dataset's three defects, which are the point

| Defect | Count | What it forces |
| --- | --- | --- |
| `total_bedrooms` missing | 207 | imputation fitted inside the fold |
| `ocean_proximity == "ISLAND"` | 5 | an unseen-category trap in any naive split |
| target censored at both ends | 965 + 4 | an honest error story, reported by segment |

Chosen over `sklearn.datasets.fetch_california_housing` precisely because it is
not clean. A pipeline needs something to do.

## Quick start

```bash
uv sync --dev
uv run pytest -q
```

```bash
kaggle datasets download -d camnugent/california-housing-prices -p data/raw --unzip
```

```bash
uv run python -m calhousing.eda           # seven reproducible figures
uv run python tools/measure_leakage.py    # what leakage actually costs
uv run python -m calhousing.train --model lgbm --n-iter 25
```

## PySpark → scikit-learn

| Spark ML | scikit-learn | Where |
| --- | --- | --- |
| `Imputer` | `SimpleImputer` / `KNNImputer` / `IterativeImputer` | `preprocess/numeric.py` |
| `StringIndexer` | `OrdinalEncoder` | `preprocess/categorical.py` |
| `OneHotEncoder` | `OneHotEncoder` | `preprocess/categorical.py` |
| `Bucketizer` | `KBinsDiscretizer` | `preprocess/categorical.py` |
| `StandardScaler` | `StandardScaler` / `RobustScaler` / `MinMaxScaler` | `preprocess/numeric.py` |
| a UDF | a `TransformerMixin` subclass | `preprocess/features.py` |
| **`VectorAssembler`** | **`ColumnTransformer`** | `preprocess/assemble.py` |
| `Pipeline` | `Pipeline` | `preprocess/assemble.py` |
| `CrossValidator` + `ParamGridBuilder` | `RandomizedSearchCV` over the whole pipeline | `train.py` |

Five places the translation is *not* one-to-one:
[`docs/pyspark-to-sklearn.md`](docs/pyspark-to-sklearn.md).

## Declared repo layout

Every path is tagged with the story that adds it. `tools/check_layout.py` holds
the same tree machine-readably and CI **fails** both on a tracked path this list
does not cover *and* on a declared path that is not tracked.

```
california-housing-price/
├── .gitattributes .gitignore .python-version        M1-S1
├── pyproject.toml uv.lock README.md CLAUDE.md       M1-S1
├── .github/workflows/ci.yml                         M1-S1
├── docs/
│   ├── sign-off-log.md  findings-register.md        M1-S1
│   ├── field-notes/m<M>-s<S>-<slug>.md              one per story
│   ├── adr/ADR-001 … ADR-004                        M1-S2 … M4-S2
│   ├── pyspark-to-sklearn.md                        M2-S4
│   ├── model-card.md                                M4-S3
│   └── evidence/                                    M4-S3  (the one-shot record)
├── src/calhousing/
│   ├── config.py  data.py  splits.py                M1-S2
│   ├── eda.py                                       M1-S3
│   ├── preprocess/numeric.py                        M2-S1
│   ├── preprocess/categorical.py                    M2-S2
│   ├── preprocess/features.py                       M2-S3
│   ├── preprocess/assemble.py                       M2-S4
│   ├── evaluate.py  models.py                       M3-S1
│   ├── train.py                                     M3-S2
│   └── interpret.py                                 M3-S4
├── tests/                                           alongside each module
├── tools/
│   ├── check_layout.py                              M1-S1
│   ├── measure_leakage.py                           M2-S4
│   ├── final_evaluation.py                          M3-S4
│   ├── build_notebook.py  run_notebook.py           M4-S1
│   └── stage_kaggle.py                              M4-S2
├── notebooks/california-housing-sklearn-pipeline.ipynb   M4-S1 (generated)
└── kaggle/{kernel,src_dataset}/*-metadata.json      M4-S2
```

**Never in the repo:** `HANDOFF.md` (session log), `data/` (regenerable),
`artifacts/` (run outputs — except the one-shot record, which is promoted to
`docs/evidence/` because it *cannot* be regenerated), staged Kaggle zips and
notebook copies, `.venv/`, `.idea/`, credentials of any kind.

## How the work was organised

One story → one branch `feat/m<M>-s<S>-<slug>` → one PR `M<M>-S<S>: <title>` →
one merge commit, in plan order, one open at a time. CI refuses a PR whose
branch and title disagree.

| Milestone | Delivers | Release |
| --- | --- | --- |
| **M1** | Scaffold, layout gate, data contract, leakage-safe split, EDA | `v0.1.0` |
| **M2** | The pipeline: impute → encode → engineer → assemble | `v0.2.0` |
| **M3** | Baselines, tuned ensembles, boosting, stacking, one test run | `v0.3.0` |
| **M4** | Generated notebook, Kaggle packaging, validation, publish | `v1.0.0` |

Every story closes with a **field note** ([`docs/field-notes/`](docs/field-notes/))
saying what was built, why that way, and what to try. Every gate crossing and
every red-team is in [`docs/sign-off-log.md`](docs/sign-off-log.md); every
defect in [`docs/findings-register.md`](docs/findings-register.md), including
the ones in our own tests.

Process governed by `UNIVERSAL_PROTOCOL.md` v2.18 in `learning` mode; see
`CLAUDE.md`.
