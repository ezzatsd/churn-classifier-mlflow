"""Run schema-validated batch churn predictions from a CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.utils import load_config, resolve_path


def predict(config_path: str, input_path: str, output_path: str) -> Path:
    config = load_config(config_path)
    features = config["features"]["numeric"] + config["features"]["categorical"]
    frame = pd.read_csv(input_path)
    missing = set(features).difference(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    for column in config["features"]["numeric"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    model = joblib.load(resolve_path(config_path, config["outputs"]["model_path"]))
    probability = model.predict_proba(frame[features])[:, 1]
    result = frame.copy()
    result["churn_probability"] = probability
    result["predicted_churn"] = (probability >= 0.5).astype(int)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    print(f"Saved {len(result)} predictions to {destination}")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="artifacts/batch_predictions.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args.config, args.input, args.output)
