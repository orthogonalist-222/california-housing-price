"""The preprocessing pipeline — the deliverable of this project.

Arranged to match the Spark ML stages it was rebuilt from, one module per
concern:

===========================  ==========================  ===================
Spark ML                     scikit-learn                module
===========================  ==========================  ===================
``Imputer``, ``Scaler``      ``SimpleImputer``, ...      ``numeric``
``StringIndexer``            ``OrdinalEncoder``          ``categorical``
``OneHotEncoder``            ``OneHotEncoder``           ``categorical``
``Bucketizer``               ``KBinsDiscretizer``        ``categorical``
a UDF                        a ``TransformerMixin``      ``features``
**``VectorAssembler``**      **``ColumnTransformer``**   ``assemble``
===========================  ==========================  ===================

Every stateful step lives inside a ``Pipeline`` so it is fitted on training
rows only, inside each cross-validation fold. That is the guarantee both
frameworks exist to give, and ``assemble`` carries the red-team that shows what
it costs to lose it.
"""

from __future__ import annotations

__all__: list[str] = []
