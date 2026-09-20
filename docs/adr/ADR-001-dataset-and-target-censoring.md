# ADR-001 — Dataset choice, and what to do about a censored target

- **Status:** accepted
- **Date:** 2026-09-20
- **Story:** M1-S2
- **Deciders:** data scientist (proposed), user (approved the dataset at plan time)

## Context

Three California housing datasets were available (recorded at plan time):

| Option | Rows | Categorical | Nulls | Verdict |
| --- | --- | --- | --- | --- |
| `camnugent/california-housing-prices` | 20 640 | `ocean_proximity`, 5 levels | 207 | **chosen** |
| Kaggle Playground S3E1 | synthetic | none meaningful | none | rejected |
| `sklearn.datasets.fetch_california_housing` | 20 640 | none | none | rejected |

The deliverable of this project is a preprocessing pipeline. The other two
options give a pipeline nothing to do: with no categorical column there is no
encoder to choose, and with no nulls there is no imputer to fit inside a fold.
The pipeline would be decorative, and a decorative pipeline teaches the wrong
lesson.

Spot-checked from the raw file on 2026-09-20, numbers **observed**:

```
shape                         20 640 x 10
total_bedrooms nulls          207        (the only nulls in the file)
ocean_proximity               <1H OCEAN 9136 | INLAND 6551 | NEAR OCEAN 2658
                              NEAR BAY 2290  | ISLAND 5
median_house_value max        500 001    -> 965 rows
median_house_value min         14 999    -> 4 rows
skew(rooms/bedrooms/pop/hh)   4.15 / 3.46 / 4.94 / 3.41
households == 0               0 rows
```

## The decision

**Use `camnugent/california-housing-prices` as-is, and treat the censoring as a
property of the target to be *reported*, not a defect to be removed.**

Specifically:

1. **Censored rows stay in.** They are 4.7% of the data and they are not errors:
   a house worth more than \$500 000 in 1990 was recorded as \$500 001. Dropping
   them would silently redefine the problem as "predict prices below the cap"
   and make every metric incomparable to every published result on this dataset.
2. **The target is censored at BOTH ends.** 965 rows at the \$500 001 cap and
   **4 rows at a \$14 999 floor**. The floor was not in the approved plan; it was
   found by spot-checking the raw file. It is small enough to change no
   decision, and large enough that a reader who notices it and finds it
   unmentioned would rightly distrust the rest.
3. **Every evaluation reports error split by censored / uncensored.** A model
   cannot be right about a capped row — the true value is unknown and above the
   cap — so a headline RMSE that mixes the two hides where the error lives.
   `evaluate.py` (M3-S1) carries this split as a first-class output.
4. **No target transform to "undo" the cap.** Tobit regression and similar
   censored-target models are the statistically correct tool. They are out of
   scope: this project is about the *preprocessing pipeline*, and swapping in a
   censored-likelihood estimator would move the lesson somewhere else.

## Options considered

| Option | Cost | Why not |
| --- | --- | --- |
| Drop the 965 capped rows | Loses 4.7% of data, biases the model low, breaks comparability | Redefines the problem silently |
| Clip predictions at the cap | Free, flattering to RMSE | Buys a metric, not an insight; hides the failure mode |
| Tobit / censored regression | Correct | Out of scope — moves the project's subject |
| **Keep, report by segment** | One extra column in every metrics table | **Chosen** |

## Consequences

- The headline test RMSE will be **worse** than a number produced by dropping
  the capped rows, and is not comparable to notebooks that drop them. The model
  card (M4-S3) must say so explicitly.
- `config.TARGET_CAP` and `config.TARGET_FLOOR` exist so no reader has to
  rediscover these numbers.
- The range check in `data.validate` allows the target up to \$600 000, which is
  above the cap on purpose: the contract's job is to catch a wrong *file*, not
  to re-assert the cap. A file whose target genuinely exceeded \$600 000 would
  not be this dataset.

## Revisit trigger

Reopen if **either**:

- the evaluation shows censored rows dominating total error (say, > 40% of
  squared error from 4.7% of rows) — at that point the censoring is no longer a
  footnote and a censored-likelihood model earns its scope; or
- the project's subject changes from "the preprocessing pipeline" to "the best
  achievable score on this dataset".
