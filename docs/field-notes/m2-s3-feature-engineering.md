# M2-S3 — Engineered features

**Built.** `preprocess/features.py`: `RatioFeatures` (three ratios, zero-safe)
and `ClusterSimilarity` (KMeans over the map + an RBF kernel).

In a Spark pipeline these would be a `withColumn` chain or a UDF. Here they are
`TransformerMixin` subclasses, and the difference is not stylistic.

## Why a transformer and not a function

A transformer is **fitted**, so it composes into a `Pipeline` and is refitted
inside every CV fold.

For the ratios that buys nothing today — they are stateless arithmetic. For
`ClusterSimilarity` it is the whole ballgame: **its centroids are learned
parameters.** Written as a preprocessing function over the whole frame — which
is the obvious way to write it — every test row contributes to defining the
regions it is later scored against. The score improves and means nothing.

Keeping both in the same shape is deliberate. The habit has to survive the next
person who adds a stateful feature without thinking about it.

The test that matters:

```python
def test_centroids_do_not_move_when_test_rows_are_transformed(geo):
    train, test = geo.iloc[:400], geo.iloc[400:]
    transformer = ClusterSimilarity(n_clusters=5).fit(train)
    before = transformer.centroids_.copy()
    transformer.transform(test)
    transformer.transform(geo)
    np.testing.assert_array_equal(transformer.centroids_, before)
```

And immediately after it, the **vacuity check** —
`test_fitting_on_everything_moves_the_centroids`. If fitting on train+test
produced the same centres as fitting on train, the test above could not detect
anything and would be quietly worthless. Every leakage test in this repo now
has one of these; it is the cheapest way to stop a guard rotting into a
tautology.

## The features earn their place from M1-S3's figures

`04-correlations` showed the four raw counts correlating strongly with **each
other** — a big block group has more of everything — and weakly with price. The
information is in the ratios: rooms per household ≈ dwelling size, bedrooms per
room ≈ how much of it is sleeping space, population per household ≈ crowding.

`03-geography` showed the other half. Price is spatial, and neither coordinate
is monotone in it. A linear model cannot express "near the Bay Area" from a
latitude and a longitude; `ClusterSimilarity` hands it a basis of smooth bumps
over the map that it can.

Two properties are asserted rather than assumed: a row sitting exactly on
centre *k* must be most similar to *k* (otherwise the feature does not mean what
its name says), and `gamma` must actually control locality — a small gamma
makes every column approach 1 and carry no information, which is a real way to
search this feature into uselessness.

## Honesty about the zero-safe division

The plan called for zero-safe division. It is implemented. But:

> **No row in this dataset has a zero denominator.** `households` and
> `total_rooms` are never 0, measured on the raw file in M1-S2.

So this guard is not fixing an observed defect — it is refusing to depend on a
property of one CSV. A block group with no households is not impossible, only
absent here, and an `inf` reaching `StandardScaler` poisons a whole column with
a message pointing at the scaler rather than the division.

The test plants synthetic zero rows and **says in its docstring that they are
synthetic**. Writing it against the real file and implying the data forced the
guard would be a nicer story and a false one. There is also a companion
asserting that plain division really does produce `inf` here — if it ever
stopped, `safe_divide` would be unnecessary and should go.

## Routing mistakes must name themselves

Both transformers raise if handed the wrong columns:

```
ValueError: RatioFeatures needs ['households']; got columns ['latitude',
'longitude']. This usually means the ColumnTransformer routed the wrong
columns here.
```

Without that, the failure is a bare `KeyError: 'households'` from inside a
transform, several frames away from the `ColumnTransformer` entry that actually
chose the columns. M2-S4 is about to create exactly that opportunity.

## What to look at

`features.py::ClusterSimilarity.fit` — note that `sample_weight` is clipped
away from zero before it reaches KMeans, because KMeans rejects a zero weight
with a message that does not name the column it came from.

## What to try

```
uv run python -c "
from calhousing.data import load_raw
from calhousing.preprocess.features import ClusterSimilarity
import numpy as np
df = load_raw(verbose=False)
t = ClusterSimilarity(n_clusters=10).fit(df)
print(np.round(t.centroids_, 2))"
```

Those are the ten places Californians actually live. Then set
`weight_by=None` and watch them drift toward empty land.
