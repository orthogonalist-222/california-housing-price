# California Housing Price — a scikit-learn preprocessing pipeline

Predicting `median_house_value` on the 1990 California census extract
([`camnugent/california-housing-prices`](https://www.kaggle.com/datasets/camnugent/california-housing-prices)),
where the **preprocessing pipeline is the deliverable** and the model is its passenger.

This dataset was chosen over scikit-learn's built-in copy because it still has
the defects that give a pipeline something to do:

| Defect | Count | What it forces |
| --- | --- | --- |
| `total_bedrooms` missing | 207 rows | imputation fitted inside the fold |
| `ocean_proximity == "ISLAND"` | 5 rows | an unseen-category trap in any naive split |
| `median_house_value` censored at `$500,001` | ~965 rows | an honest error story, reported by segment |

> Status: **M2 complete** (`v0.2.0`) — the preprocessing pipeline is built and
> assembled: impute, de-skew, encode, bin, engineer, and one `ColumnTransformer`
> that routes them. Models land in M3. This README is rewritten at M4-S3 with
> the headline numbers.
>
> See **[docs/pyspark-to-sklearn.md](docs/pyspark-to-sklearn.md)** for the
> stage-by-stage translation, and run `uv run python tools/measure_leakage.py`
> for what leakage actually costs here (less than the folklore claims — and the
> one that does bite is not the one you are told about).

## Quick start

```bash
uv sync --dev
uv run pytest -q
uv run python tools/check_layout.py
uv run python -m calhousing.eda        # seven figures -> artifacts/eda/
uv run python tools/measure_leakage.py # what leakage costs, measured
uv run python -m calhousing.train --model lgbm --n-iter 25
```

Raw data is never tracked; fetch it with:

```bash
kaggle datasets download -d camnugent/california-housing-prices -p data/raw --unzip
```

## Declared repo layout

Every path below is tagged with the story that adds it. `tools/check_layout.py`
holds the same tree in machine-readable form and CI **fails** on a tracked path
this list does not cover — a new path arrives only through a story that declares
it first.

```
california-housing-price/
├── .gitattributes                                  M1-S1
├── .gitignore                                      M1-S1
├── .python-version                     (3.12)      M1-S1
├── pyproject.toml                                  M1-S1
├── uv.lock                                         M1-S1
├── README.md                                       M1-S1
├── CLAUDE.md                                       M1-S1
├── .github/workflows/ci.yml                        M1-S1
├── docs/
│   ├── sign-off-log.md                             M1-S1
│   ├── findings-register.md                        M1-S1
│   ├── field-notes/m<M>-s<S>-<slug>.md             one per story
│   ├── adr/ADR-001-dataset-and-target-censoring.md M1-S2
│   ├── adr/ADR-002-encoding-strategy.md            M2-S2
│   ├── adr/ADR-003-cv-and-test-protocol.md         M3-S1
│   ├── adr/ADR-004-packaging-dual-path.md          M4-S2
│   ├── pyspark-to-sklearn.md                       M2-S4
│   └── model-card.md                               M4-S3
├── src/calhousing/
│   ├── __init__.py                                 M1-S1
│   ├── py.typed                                    M1-S1
│   ├── config.py                                   M1-S2
│   ├── data.py                                     M1-S2
│   ├── splits.py                                   M1-S2
│   ├── eda.py                                      M1-S3
│   ├── preprocess/numeric.py                       M2-S1
│   ├── preprocess/categorical.py                   M2-S2
│   ├── preprocess/features.py                      M2-S3
│   ├── preprocess/assemble.py                      M2-S4
│   ├── models.py                                   M3-S1
│   ├── evaluate.py                                 M3-S1
│   ├── train.py                                    M3-S2
│   └── interpret.py                                M3-S4
├── tests/                                          alongside each module
│   └── conftest.py  (synthetic fixtures)          M1-S2
├── tools/
│   ├── check_layout.py                             M1-S1
│   ├── measure_leakage.py                          M2-S4
│   ├── build_notebook.py                           M4-S1
│   └── stage_kaggle.py                             M4-S2
├── notebooks/california-housing-sklearn-pipeline.ipynb   M4-S1 (generated)
└── kaggle/
    ├── kernel/kernel-metadata.json                 M4-S2
    └── src_dataset/dataset-metadata.json           M4-S2
```

**Never in the repo:** `HANDOFF.md` (session log), `data/` (regenerable),
`artifacts/` (run outputs), staged Kaggle zips and notebook copies, `.venv/`,
`.idea/`, and credentials of any kind.

## How the work is organised

One story → one branch `feat/m<M>-s<S>-<slug>` → one PR `M<M>-S<S>: <title>` →
one merge commit, in plan order, one open at a time.

| Milestone | Delivers |
| --- | --- |
| **M1** | Foundation: scaffold, data contract, leakage-safe split, EDA |
| **M2** | The pipeline: impute → encode → engineer → assemble |
| **M3** | Models: baselines, tuned ensembles, boosting, stacking, one test run |
| **M4** | Delivery: generated notebook, Kaggle packaging, independent validation |

Process is governed by `UNIVERSAL_PROTOCOL.md` v2.18 in `learning` mode; see
`CLAUDE.md`.
