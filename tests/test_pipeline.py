import numpy as np
import pandas as pd
import pytest

from src.pipeline import build_pipeline


@pytest.fixture
def toy_data():
    X = pd.DataFrame(
        {
            "tenure": [1.0, 12.0, np.nan, 48.0, 3.0, 60.0],
            "contract": ["Monthly", "Annual", "Monthly", None, "Monthly", "Two year"],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1, 0])
    return X, y


def test_build_pipeline_has_expected_steps():
    pipe = build_pipeline(["a"], ["b"], "logreg")
    assert list(pipe.named_steps) == ["pre", "model"]


def test_pipeline_fits_with_missing_values(toy_data):
    X, y = toy_data
    pipe = build_pipeline(["tenure"], ["contract"], "logreg")
    pipe.fit(X, y)
    assert pipe.predict_proba(X).shape == (6, 2)


def test_pipeline_accepts_unseen_category(toy_data):
    X, y = toy_data
    pipe = build_pipeline(["tenure"], ["contract"], "logreg").fit(X, y)
    unseen = pd.DataFrame({"tenure": [8.0], "contract": ["Never seen"]})
    assert pipe.predict(unseen).shape == (1,)


def test_random_forest_pipeline_builds():
    pipe = build_pipeline(["a"], ["b"], "random_forest", random_state=7)
    assert pipe.named_steps["model"].random_state == 7


def test_unsupported_model_fails_clearly():
    with pytest.raises(ValueError, match="Unsupported model_type"):
        build_pipeline(["a"], ["b"], "svm")
