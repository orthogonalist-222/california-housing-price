# Findings register

One writer, namespaced ids, closed **only by evidence** — never by a reply.
A row a `grep` cannot find is a finding that silently ceased to exist, so the
table's shape is fixed and parsed by the gates that read it.

Severity: `S1` blocker · `S2` must fix before the milestone closes · `S3` should
fix · `S4` note.

**Closure gate:** no `S1` or `S2` may cross a milestone boundary still open.

| id | date | severity | story | owner | finding | status | closing evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F-001 | 2026-09-20 | S2 | M2-S1 | data scientist | `deskew="log"` on a column with values ≤ −1 (`longitude`) returned a frame of NaNs with no error; the only signal was a RuntimeWarning from inside sklearn's extmath naming neither the column nor the step | **closed** | `_checked_log1p` raises `DomainError` naming the column and the cure; `test_log_on_an_out_of_domain_column_is_refused_by_name` + `test_the_guard_does_not_fire_on_valid_input` |
| F-002 | 2026-09-20 | S3 | M2-S1 | data scientist | The synthetic fixture drew count columns from a **uniform** distribution, so it reproduced the real file's column names while losing the 3.4–4.9 skew those columns exist to exercise; a de-skew test was asserting a log reduces the skew of a uniform | **closed** | Fixture now draws log-normal (measured synthetic skew 2.9–3.4); `test_log_deskew_actually_reduces_skew` passes and is no longer vacuous |
| F-003 | 2026-09-20 | S2 | M2-S1 | data scientist | Two **M1-S2** tests relied on a particular random draw producing a singleton composite group rather than planting one, so an unrelated fixture change turned one into a test that asserted nothing and broke the other; one of them also carried a `pytest.skip` that excused the same problem | **closed** | Both now use `_frame_with_singleton_island`, which plants the real file's arrangement; the skip is gone and the singleton is asserted |
| F-004 | 2026-09-20 | S1 | M2-S4 | data scientist | The engineered ratios are unbounded and the dataset contains institutional block groups (one with 6 households and 7 460 people, `population_per_household` = 1 243 vs a median of 2.8). Standard-scaled, such a row is a z-score in the hundreds; on a 5 000-row subsample one CV fold returned **RMSE 1 805 055** beside neighbours around 65 000 | **closed** | `QuantileClipper` winsorises at train-learned quantiles inside the fold; folds became `[61694, 69169, 63222, 65349, 72352]` and full-data Ridge improved 65 838 -> 63 811. Regression test: `test_an_institutional_block_group_does_not_blow_up_predictions` |
