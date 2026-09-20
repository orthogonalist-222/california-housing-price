# M2-S2 — Categorical block: label vs one-hot

**Built.** `preprocess/categorical.py` — `OrdinalEncoder` (Spark's
`StringIndexer`), `OneHotEncoder`, and `KBinsDiscretizer` (Spark's
`Bucketizer`), each wrapped in a block that imputes first and emits pandas.
Plus ADR-002, which says which column gets which encoder and why.

## I measured before writing, and all three results changed the design

This is the habit worth taking from this story. The plan said
"`OneHotEncoder(handle_unknown="infrequent_if_exist")`" as if that phrase
solved the problem. It does not.

### 1. `infrequent_if_exist` is a no-op on its own

With no `min_frequency`, it behaves **identically** to `handle_unknown="ignore"`:
an unknown row is all zeros. Setting `min_frequency=10` gives it a bucket — but
only if a category in the **training** data is below the threshold:

```
train CONTAINS the 4-row island, min_frequency=10
  names  = [..., 'ocean_proximity_infrequent_sklearn']
  ISLAND -> [0, 0, 0, 0, 1]   sum=1.0     <- a PRESENCE

train has NO island, min_frequency=10
  names  = [...]                          <- no infrequent column at all
  ISLAND -> [0, 0, 0, 0]      sum=0.0     <- an ABSENCE
```

**So the encoder only works because the split guarantees the island is in
train.** M1-S2 collapsed the stratification key per level so `ISLAND` lands on
both sides at every seed — a decision made for the *split*, which turns out to
be load-bearing for the *encoder*. Neither is sufficient alone, and there is now
a test whose entire job is to fail if that guarantee is ever removed.

Two stories apart, one dependency. That is the kind of thing a pipeline hides
and a test has to state.

### 2. `drop="first"` destroys the distinction it looks like it keeps

```
drop='first', handle_unknown='ignore'
dropped reference level = '<1H OCEAN'
unknown 'ISLAND'    -> [[0.0, 0.0]]
known   '<1H OCEAN' -> [[0.0, 0.0]]
IDENTICAL: True
```

An unseen category and the dropped baseline are **the same vector**. sklearn
permits this. `make_onehot_encoder` refuses it — raised, not warned, because the
loss is invisible in the output: right shapes, plausible numbers, two categories
silently merged.

A companion test demonstrates the collision directly with the guard bypassed,
and its failure message says that if scikit-learn ever stops doing this, the
guard has become unnecessary strictness and should be deleted. A guard should
carry the condition under which it stops being justified.

**And the warning is backwards.** scikit-learn emits its "encoded as all zeros"
`UserWarning` *only when `drop` is set*. The safe configuration is silent; the
unsafe one is loud. Anybody relying on warnings to notice this gets it exactly
the wrong way round.

### 3. Declaring the categories removes the problem at the source

`OrdinalEncoder(categories=[OCEAN_PROXIMITY_ORDER])` encodes `ISLAND` correctly
even when the fit never saw one — the **schema** defines the levels, not the
sample. This is the ordinal encoder's real advantage here, and it has nothing to
do with ordering.

## The concept: an ordinal code must assert something true

`ISLAND < NEAR BAY < NEAR OCEAN < <1H OCEAN < INLAND` is increasing distance
from water. `0 < 1 < 2 < 3 < 4` therefore states something about the world, and
a tree splitting on `code <= 2` is splitting on "near the water".

Spark's `StringIndexer` orders by **frequency**. That ordering asserts nothing —
and any model able to use an ordering will happily use the noise. This is the
single most common way a translated PySpark pipeline quietly degrades, and the
project keeps `make_ordinal_encoder(declared=False)` around purely so the
notebook can show the contrast.

Which encoder actually wins is left open: M3-S2 searches it jointly with the
model, because trees can exploit an honest ordinal scale and linear models
generally cannot.

## What to look at

The module docstring in `categorical.py` — it carries all three measurements
verbatim, so the next reader does not have to rerun them. Then ADR-002's
decision table.

## What to try

```
uv run python -c "
import pandas as pd
from calhousing.preprocess.categorical import make_onehot_encoder
f = pd.DataFrame({'ocean_proximity': ['INLAND']*150 + ['ISLAND']*4})
e = make_onehot_encoder(min_frequency=10).fit(f)
print(list(e.get_feature_names_out()))
print(e.transform(pd.DataFrame({'ocean_proximity': ['NEAR LAKE']})))"
```

Then remove the four `ISLAND` rows from `f` and run it again — the infrequent
column disappears and the unknown becomes an absence.
