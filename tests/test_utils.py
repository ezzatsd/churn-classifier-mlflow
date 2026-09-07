from pathlib import Path

import pandas as pd

from src.utils import classification_metrics, load_config, split_dataset


def test_config_declares_disjoint_features():
    config = load_config(Path(__file__).parents[1] / "configs/config.yaml")
    numeric = set(config["features"]["numeric"])
    categorical = set(config["features"]["categorical"])
    assert numeric
    assert categorical
    assert numeric.isdisjoint(categorical)


def test_split_is_stratified_and_reproducible():
    frame = pd.DataFrame({"x": range(100), "target": [0] * 75 + [1] * 25})
    config = {
        "data": {"target": "target", "positive_label": "1", "test_size": 0.2, "random_state": 42},
        "features": {"numeric": ["x"], "categorical": []},
    }
    first = split_dataset(frame, config)
    second = split_dataset(frame, config)
    assert first[1].index.tolist() == second[1].index.tolist()
    assert first[3].mean() == 0.25


def test_metric_keys_and_perfect_score():
    y = pd.Series([0, 0, 1, 1])
    metrics = classification_metrics(y, pd.Series([0.01, 0.2, 0.8, 0.99]).to_numpy())
    assert metrics["roc_auc"] == 1.0
    assert metrics["f1"] == 1.0
