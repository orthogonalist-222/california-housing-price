"""California housing price prediction - a scikit-learn preprocessing pipeline.

The package is deliberately arranged around the *pipeline*, not the model::

    data.py        raw CSV -> validated frame            (M1-S2)
    splits.py      leakage-safe train/test split         (M1-S2)
    preprocess/    impute -> encode -> engineer -> assemble  (M2)
    models.py      estimator registry and search spaces  (M3)
    evaluate.py    metrics, including error by segment   (M3)

Everything stateful lives inside a scikit-learn ``Pipeline`` so that it is
fitted on training rows only, inside each cross-validation fold. That is the
same guarantee a Spark ML ``Pipeline`` gives, and it is the reason this project
exists; ``docs/pyspark-to-sklearn.md`` maps the stages one to one.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
