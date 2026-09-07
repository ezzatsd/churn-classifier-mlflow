"""Build leakage-safe scikit-learn preprocessing and classification pipelines."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(
    numeric: Sequence[str],
    categorical: Sequence[str],
    model_type: str = "logreg",
    random_state: int = 42,
) -> Pipeline:
    """Return a complete preprocessing + estimator pipeline.

    All statistics and encodings are fitted inside cross-validation folds, which
    prevents data leakage from the validation set.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numeric)),
            ("cat", categorical_pipeline, list(categorical)),
        ],
        remainder="drop",
    )

    if model_type == "logreg":
        model = LogisticRegression(max_iter=1_000, random_state=random_state)
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
    else:
        raise ValueError(
            f"Unsupported model_type={model_type!r}. Choose 'logreg' or 'random_forest'."
        )

    return Pipeline(steps=[("pre", preprocessor), ("model", model)])
