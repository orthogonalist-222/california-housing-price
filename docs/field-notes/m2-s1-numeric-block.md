# M2-S1 — Numeric block: impute → de-skew → scale

**Built.** `preprocess/numeric.py`: a three-step `Pipeline` with named steps
`impute`, `deskew`, `scale`, each selectable by string, all emitting pandas.

This is Spark's `Imputer` + `StandardScaler` rebuilt in sklearn. Four imputers,
three de-skew options, four scalers — not because the pipeline needs the
variety, but because **M3-S2 searches over them jointly with the model**, which
is what `ParamGridBuilder` does and what most tutorial pipelines never do.

## Two details that are deliberate

**Disabled steps are `"passthrough"`, not removed.** A named-but-inert step
stays addressable by a search (`preprocess__numeric__deskew`); a removed one
does not, and the search space would silently target nothing and report no
error. There is a test.

**`feature_names_out="one-to-one"` on the log transformer.** Without it the
step refuses to participate in `get_feature_names_out`, and the whole pipeline
loses the ability to name its own columns. That does not fail here — it fails
in M3-S4, as an importance plot labelled `x17`.

## The story's real finding: a transform that returns NaN instead of raising

Running the numeric block with `deskew="log"` over the whole numeric frame
produced **no error and a frame of NaNs**.

`longitude` is about −124. `np.log1p(-124)` is NaN. Numpy shrugs. The NaNs then
reached `StandardScaler`, which produced this:

```
RuntimeWarning: invalid value encountered in divide
  T = new_sum / new_sample_count      (sklearn/utils/extmath.py:1149)
```

A warning, from inside scikit-learn, in a file the reader has never opened,
about a divide — with **no mention of `longitude`**, of `log1p`, or of the
pipeline step that caused it. Had this run inside a cross-validation loop it
would have surfaced as a model that trained fine and scored like noise.

The cure is not a cleverer log. It is `_checked_log1p`, which refuses:

```
DomainError: log1p is undefined at or below -1, and would return NaN for:
longitude (min -124). Route these columns through a block with
deskew='none' or 'yeo-johnson' instead.
```

Names the column, names the cure. And both halves are pinned: one test asserts
the refusal fires on `longitude`, another asserts it does **not** fire on the
positive count columns — *a guard that refuses correct input is worse than no
guard*. A third pins that NaN is not read as "below −1", because the guard sits
downstream of the imputer today and a future reordering would otherwise turn
every missing value into a domain error.

## The concept: leakage is a property you can test on one block

`test_transform_of_a_row_does_not_depend_on_the_other_rows` fits once, then
transforms the test frame whole and transforms single rows alone. If the two
disagree, the block is reading statistics from the data it is scoring. That is
leakage, stated as something checkable **before** there is a pipeline or a
model to hide it in.

Its companion runs the check from the other side: fit on train, then fit on
train-plus-wildly-different-rows, and assert the learned fill values *move*. If
appending unseen rows changed nothing, the first test could not detect leakage
at all and would be quietly vacuous.

## And a defect in my own tests, found by an unrelated change

The synthetic fixture generated counts from a **uniform** draw. It reproduced
the real file's column *names* while losing the property those columns exist to
exercise — skew 3.4–4.9. So `test_log_deskew_actually_reduces_skew` was
asserting that a log reduces the skew of a uniform distribution, which it does
not. The fixture now draws log-normal (measured skew 2.9–3.4).

Fixing that shifted every later draw in the fixture, and
`test_collapse_is_per_level_not_per_group` — an **M1-S2** test — failed. It had
been taking whatever `make_housing(seed=3)` happened to produce and asserting
the level-collapse had fired. A change to an unrelated column turned it into a
test that asserted nothing.

That is precisely the era-fact trap those same tests warn about, sitting inside
the tests themselves. Both now use `_frame_with_singleton_island`, which
*plants* the real file's arrangement — four islands in income band 2, one alone
in band 3 — instead of hoping a seed produces it. A `pytest.skip` that quietly
excused the same problem in a neighbouring test is gone too.

## What to look at

`numeric.py::_checked_log1p`, then `test_numeric.py` — the three tests around
the domain guard, and the leakage pair.

## What to try

```
uv run python -c "
from calhousing.data import load_raw
from calhousing.preprocess.numeric import build_numeric_block
from calhousing import config
load_raw(); block = build_numeric_block(deskew='log')
block.fit_transform(load_raw(verbose=False)[config.NUMERIC_COLUMNS])"
```

Read the refusal. Then swap `func=_checked_log1p` back to `np.log1p` and run it
again — the run succeeds, and every value is NaN.
