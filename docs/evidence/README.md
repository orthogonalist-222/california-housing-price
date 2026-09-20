# Evidence

Everything else this project produces is **cattle** — figures, fitted models,
search results — regenerable by the committed recipe and therefore gitignored.

These files are the exception.

`test-set-evaluation.json` is the record of the **one-shot test-set run**
(ADR-003). It cannot be regenerated without scoring the test set a second time,
which is the failure the whole protocol exists to prevent. So it is committed:
the evidence a reviewer walks, readable without unpickling anything.

| file | what it is |
| --- | --- |
| `test-set-evaluation.json` | Every arm's full segment table, the tuned parameters used, timings, environment and timestamp |
| `headline.csv` | The five arms' overall metrics, best RMSE first |
| `01-permutation-importance.png` | Importance over raw columns, computed on a **training** slice |
| `02-partial-dependence.png` | `median_income` and `housing_median_age` |

The two figures are reproducible, and are kept here so the claims in the model
card and the README have something to point at without a reader having to run
anything.
