"""Evaluate a persisted churn pipeline and log holdout artifacts to MLflow."""

from __future__ import annotations

import argparse
import json
import os

import joblib
import mlflow

from src.utils import (
    classification_metrics,
    load_config,
    load_dataset,
    resolve_path,
    save_evaluation_artifacts,
    split_dataset,
)


def evaluate(config_path: str, model_path_override: str | None = None) -> dict[str, float]:
    config = load_config(config_path)
    data = load_dataset(config, config_path)
    _, X_test, _, y_test = split_dataset(data, config)
    model_path = (
        resolve_path(config_path, model_path_override)
        if model_path_override
        else resolve_path(config_path, config["outputs"]["model_path"])
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run `make train` first.")
    model = joblib.load(model_path)
    probability = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, probability)
    output_dir = resolve_path(config_path, config["outputs"]["evaluation_dir"])
    artifacts = save_evaluation_artifacts(y_test, probability, output_dir, X_test.index)

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    )
    mlflow.set_experiment(
        os.getenv("MLFLOW_EXPERIMENT_NAME", config["mlflow"]["experiment_name"])
    )
    with mlflow.start_run(run_name="holdout-evaluation"):
        mlflow.set_tag("evaluation.dataset", "deterministic-stratified-holdout")
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
        for artifact in artifacts.values():
            mlflow.log_artifact(str(artifact), artifact_path="evaluation")
    print(json.dumps(metrics, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model-path", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.config, args.model_path)
