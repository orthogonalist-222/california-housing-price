# PySpark ML → scikit-learn, stage by stage

The same discipline in both frameworks, with different nouns. This table is the
spine of M2 and it reappears in the notebook.

| Spark ML | scikit-learn | In this repo |
| --- | --- | --- |
| `Imputer` | `SimpleImputer` / `KNNImputer` / `IterativeImputer` | `preprocess/numeric.py` |
| `StringIndexer` | `OrdinalEncoder` | `preprocess/categorical.py` |
| `OneHotEncoder` | `OneHotEncoder` | `preprocess/categorical.py` |
| `Bucketizer` / `QuantileDiscretizer` | `KBinsDiscretizer` | `preprocess/categorical.py` |
| `StandardScaler` / `MinMaxScaler` / `RobustScaler` | same names | `preprocess/numeric.py` |
| a UDF or `withColumn` chain | a `BaseEstimator` + `TransformerMixin` subclass | `preprocess/features.py` |
| **`VectorAssembler`** | **`ColumnTransformer`** | `preprocess/assemble.py` |
| `Pipeline` | `Pipeline` | `preprocess/assemble.py` |
| `CrossValidator` + `ParamGridBuilder` | `RandomizedSearchCV` over the whole pipeline | `train.py` (M3-S2) |
| `PipelineModel` (the fitted artefact) | the fitted `Pipeline` object | pickled by `joblib` |

## Five places the translation is not one-to-one

**1. `VectorAssembler` concatenates; `ColumnTransformer` routes *and*
concatenates.** In Spark you add columns with `withColumn` and then name them
all in an assembler. In sklearn the `ColumnTransformer` chooses which columns
each branch sees. That is more power and one more thing to get wrong — hence
the by-name refusals in `RatioFeatures` and `ClusterSimilarity`, which say
"this usually means the ColumnTransformer routed the wrong columns here"
instead of raising a bare `KeyError` several frames away.

A branch is a fan-out, not a partition: `total_rooms` feeds both `ratios` (as a
denominator) and `heavy` (as a de-skewed magnitude).

**2. `StringIndexer` orders by frequency. That asserts nothing.** Our
`OrdinalEncoder` is given `categories=[OCEAN_PROXIMITY_ORDER]` — increasing
distance from water — so `0 < 1 < 2 < 3 < 4` states something true about the
world, and a tree splitting on `code <= 2` is splitting on "near the water".
A frequency ordering is noise that any model able to use an ordering will
happily use. `make_ordinal_encoder(declared=False)` reproduces the Spark
behaviour, purely so the notebook can show the contrast.

**3. Declaring the categories is not just tidiness — it fixes the unseen-level
problem.** With explicit `categories`, a level missing from a training fold
still encodes correctly, because the *schema* defines the levels rather than the
sample. This matters because `ISLAND` has five rows.

**4. `handleInvalid` and `handle_unknown` do not line up.** Spark's
`StringIndexer(handleInvalid="keep")` puts unknowns in an extra index. sklearn
needs `handle_unknown="use_encoded_value", unknown_value=-1` for the ordinal
case, and for one-hot it needs `min_frequency` set before
`handle_unknown="infrequent_if_exist"` does anything at all. See ADR-002 — all
three behaviours were measured, and all three changed the design.

**5. `setParams` vs `set_params`.** Spark's `ParamGridBuilder` addresses stage
parameters by object reference; sklearn addresses them by **string path**
(`preprocess__heavy__deskew`). The path is a public interface: rename a step and
the search space silently targets nothing, with no error. `test_assemble.py`
pins the paths for that reason.

## The shared idea

Both frameworks exist so that every **stateful** step — a median, a scaler's
mean, a bin edge, a KMeans centroid, an encoder's category list — is learned
from training rows only and reapplied unchanged to everything else.

`build_pipeline` returns a single estimator. Hand it to `cross_val_score` and
all of that is relearned per fold. Hand `cross_val_score` a pre-transformed
matrix and all of it was learned once, from everything, including the rows about
to be scored.

## But measure the payoff, because the folklore overstates it

`tools/measure_leakage.py` runs the comparison on this dataset:

```
A. Preprocessing inside the fold vs. fitted once on everything
   honest  RMSE     63,811  +/- 1,430
   leaked  RMSE     63,897  +/- 1,631
   leak                -86  (-0.135% of honest)
   -> the leak is 0.06x the fold-to-fold spread
```

The "leaked" version is **worse**, by an amount well inside the noise. Shrinking
the training set does not produce a trend either — the sign flips. On this
dataset, with a random split and one column that is 1% missing, fitting the
preprocessing outside the fold costs nothing measurable.

That is worth saying plainly. A reader told to expect a dramatic number who then
measures 0.05% concludes the whole lesson was theatre.

The lesson survives in a better form:

```
C. A column derived from the target, added to the input
   clean frame                       RMSE     63,811
   leaky column, through assembler   RMSE     63,811
   leaky column, forced past it      RMSE      5,121
```

**The large leak is a leaked column, and what stops it is that the assembler is
a whitelist.** `remainder="drop"` means an undeclared column never reaches the
model, no matter how target-shaped it is. `remainder="passthrough"` — one word
different — is how a leaked column arrives without anybody choosing to put it
there.

So the pipeline's real payoff on a dataset like this is not the score. It is
that the set of features a model can see is **declared in one place, reviewable,
and enforced**. The fold discipline is what keeps that true when the data is
smaller, shifted, or has a stateful step with more to learn — and it costs
nothing to keep.
