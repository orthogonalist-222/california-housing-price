# ADR-002 — Which encoder each column gets, and how unknowns are handled

- **Status:** accepted
- **Date:** 2026-09-20
- **Story:** M2-S2
- **Deciders:** data scientist

## Context

One string column, `ocean_proximity`, five levels, the rarest with **5 rows out
of 20 640**. Any fold can be fitted without ever seeing it. So "how does the
encoder behave on a category it never saw" is not an edge case to be handled
politely at the end — it is the decision.

Three behaviours were measured on scikit-learn 1.7.2 **before** this module was
written, and each one changed it.

### 1. `handle_unknown="infrequent_if_exist"` is a no-op on its own

With no `min_frequency` set it is identical to `"ignore"`: an unknown row is all
zeros. With `min_frequency=10` it maps unknowns into a real column — but **only
when a training category is actually below the threshold**:

```
train CONTAINS the 4-row island, min_frequency=10
  names  = [..., 'ocean_proximity_infrequent_sklearn']
  ISLAND -> [0, 0, 0, 0, 1]   sum=1.0
  novel  -> [0, 0, 0, 0, 1]   sum=1.0

train has NO island, min_frequency=10
  names  = [...]                     <- no infrequent column at all
  ISLAND -> [0, 0, 0, 0]      sum=0.0
```

The consequence is a dependency between two stories: **the encoder's
unknown-handling only works because M1-S2's split guarantees `ISLAND` reaches
the training side at every seed.** Neither design is sufficient alone, and
`test_the_infrequent_bucket_needs_the_rare_level_at_fit_time` says out loud what
breaks if the split guarantee is ever removed.

### 2. `drop="first"` makes an unknown identical to the baseline

```
drop='first', handle_unknown='ignore'
dropped reference level = '<1H OCEAN'
unknown 'ISLAND'    -> [[0.0, 0.0]]
known   '<1H OCEAN' -> [[0.0, 0.0]]
IDENTICAL: True
```

scikit-learn permits this. Two genuinely different categories become the same
vector, and nothing downstream can separate them. Without `drop`, a valid row
sums to 1 and an unknown sums to 0 — the distinction survives.

Worth knowing: scikit-learn emits its `will be encoded as all zeros` warning
**only when `drop` is set**. The quiet case is the no-drop one. So the louder
configuration is the *unsafe* one, which is the wrong way round for anybody
relying on warnings.

### 3. Declaring the categories removes the problem at the source

`OrdinalEncoder(categories=[OCEAN_PROXIMITY_ORDER])` encodes `ISLAND` correctly
even when the fit never saw one, because the **schema** defines the levels, not
the sample. A level outside the declared list still needs `unknown_value`.

## The decision

| Column | Encoder | Configuration | Why |
| --- | --- | --- | --- |
| `ocean_proximity` | **one-hot** (default) | `handle_unknown="infrequent_if_exist"`, `min_frequency=10`, **`drop=None`** | Makes no ordering claim; an unknown gets a positive signal via the infrequent bucket |
| `ocean_proximity` | **ordinal** (alternative, searched in M3-S2) | `categories=[OCEAN_PROXIMITY_ORDER]`, `handle_unknown="use_encoded_value"`, `unknown_value=-1` | The order is *increasing distance from water* — a real scale, not an arbitrary index |
| `median_income` | **binned** → either encoder | `KBinsDiscretizer(5, strategy="quantile")` | Spark's `Bucketizer`; the same bands the split stratifies on |
| `housing_median_age` | **binned** → either encoder | `KBinsDiscretizer(5, strategy="quantile")` | Lets a linear model express "1960s stock" without a spline |

**`drop` combined with tolerant unknown handling is refused** — `make_onehot_encoder`
raises `UnsafeEncoderError`. Not a warning: the loss it describes is invisible
in the output. The numbers look fine, the shapes are right, and two categories
have simply become one.

**The ordinal ordering is an assertion, not a convenience.** `ISLAND < NEAR BAY
< NEAR OCEAN < <1H OCEAN < INLAND` is increasing distance from water, so
`0 < 1 < 2 < 3 < 4` states something true about the world. Spark's
`StringIndexer` orders by frequency, which asserts nothing — and any model able
to use the ordering will happily use that noise. `make_ordinal_encoder(declared=False)`
reproduces the `StringIndexer` behaviour for the notebook to contrast against;
it is not used in the pipeline.

**Both encoders stay available** rather than one being chosen here. Which wins
is an empirical question, and M3-S2 answers it by searching the encoder choice
jointly with the model — trees can exploit an honest ordinal scale, linear
models generally cannot.

## Options considered

| Option | Why not |
| --- | --- |
| `handle_unknown="error"` (the default) | Trains happily, dies at scoring time on 5 rows out of 20 640 |
| `drop="first"` for collinearity | Merges unknown with the baseline; the collinearity it fixes costs nothing to a regularised or tree model |
| Target/mean encoding | The strongest option on high-cardinality columns, and genuinely tempting. Rejected: with five levels there is little to gain, and it needs nested CV to avoid leaking the target — which would obscure the pipeline lesson this project exists to teach. Revisit if a high-cardinality column is ever added. |
| One-hot only, no ordinal | Throws away a real ordering and removes the `StringIndexer` comparison that motivates the project |

## Consequences

- The pipeline carries an `infrequent_sklearn` column that exists solely because
  `ISLAND` is rare. It will appear in every feature-importance plot; the model
  card must say what it is.
- `min_frequency=10` is a threshold, and a threshold is a parameter. It is
  searchable in M3-S2 rather than fixed by taste.
- Binning `median_income` creates a column correlated with the stratification
  key. That is intentional and harmless — the split is made before any fitting —
  but it is the kind of thing that looks like leakage to a reader, so the
  notebook says so explicitly.

## Revisit trigger

Reopen if a high-cardinality categorical is added (target encoding becomes
worth its cost), or if scikit-learn changes either measured behaviour — the
tests `test_the_collision_the_guard_prevents_is_real` and
`test_without_drop_the_two_are_distinguishable` fail loudly in that case, and
each says in its failure message that the guard has become unnecessary.
