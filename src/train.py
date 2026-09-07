"""Tune, evaluate, track and register the churn classifier."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.pipeline import build_pipeline
from src.utils import (
    classification_metrics,
    load_config,
    load_dataset,
    resolve_path,
    save_evaluation_artifacts,
    seed_everything,
    split_dataset,
)


def _parameter_grid(config: dict[str, Any]) -> dict[str, list[Any]]:
    """Prefix estimator parameters so GridSearchCV addresses the pipeline model."""
    return {f"model__{name}": values for name, values in config["model"]["params"].items()}


def _configure_mlflow(config: dict[str, Any]) -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    experiment = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", config["mlflow"]["experiment_name"]
    )
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return experiment


def _register_with_alias(model_name: str, model_uri: str) -> str:
    """Register the model and expose a modern alias plus the requested legacy stage."""
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    client = MlflowClient()
    client.set_registered_model_alias(model_name, "staging", result.version)
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Staging",
            archive_existing_versions=True,
        )
    except Exception as exc:  # MLflow 3 can disable deprecated stages.
        print(f"Legacy stage skipped; alias 'staging' is available: {exc}")
    return str(result.version)


def train(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config["data"]["random_state"])
    seed_everything(seed)
    data = load_dataset(config, config_path)
    X_train, X_test, y_train, y_test = split_dataset(data, config)

    pipeline = build_pipeline(
        config["features"]["numeric"],
        config["features"]["categorical"],
        config["model"]["type"],
        seed,
    )
    cv = StratifiedKFold(
        n_splits=int(config["cv"]["n_splits"]), shuffle=True, random_state=seed
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=_parameter_grid(config),
        scoring=config["cv"]["scoring"],
        cv=cv,
        n_jobs=int(config["cv"].get("n_jobs", -1)),
        refit=True,
        return_train_score=True,
        error_score="raise",
    )

    experiment = _configure_mlflow(config)
    mlflow.sklearn.autolog(
        log_input_examples=False,
        log_model_signatures=False,
        log_models=False,
        max_tuning_runs=int(config["mlflow"].get("max_tuning_runs", 20)),
        silent=True,
    )
    with mlflow.start_run(run_name=config["mlflow"]["run_name"]) as run:
        mlflow.set_tags(
            {
                "project": "telco-customer-churn",
                "dataset": "IBM Telco Customer Churn",
                "model_family": config["model"]["type"],
                "purpose": "model-selection-and-holdout-evaluation",
            }
        )
        mlflow.log_params(
            {
                "data.rows": len(data),
                "data.train_rows": len(X_train),
                "data.test_rows": len(X_test),
                "data.positive_rate": round(float(y_train.mean()), 6),
                "split.random_state": seed,
                "split.test_size": config["data"]["test_size"],
                "cv.strategy": config["cv"]["strategy"],
                "cv.n_splits": config["cv"]["n_splits"],
                "cv.scoring": config["cv"]["scoring"],
            }
        )
        mlflow.log_artifact(config_path, artifact_path="configuration")
        search.fit(X_train, y_train)
        probability = search.best_estimator_.predict_proba(X_test)[:, 1]
        metrics = classification_metrics(y_test, probability)
        mlflow.log_metric("cv_best_score", float(search.best_score_))
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
        mlflow.log_params({f"best_{key}": value for key, value in search.best_params_.items()})

        artifacts_dir = resolve_path(config_path, config["outputs"]["evaluation_dir"])
        saved = save_evaluation_artifacts(
            y_test, probability, artifacts_dir, row_ids=X_test.index
        )
        for artifact in saved.values():
            mlflow.log_artifact(str(artifact), artifact_path="holdout_evaluation")

        signature = infer_signature(X_train, search.best_estimator_.predict(X_train))
        input_example = X_train.head(3)
        model_info = mlflow.sklearn.log_model(
            sk_model=search.best_estimator_,
            name="model",
            signature=signature,
            input_example=input_example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            metadata={"decision_threshold": 0.5, "positive_class": "Churn=Yes"},
        )
        model_path = resolve_path(config_path, config["outputs"]["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(search.best_estimator_, model_path)

        cv_results_path = artifacts_dir / "cv_results.csv"
        import pandas as pd

        pd.DataFrame(search.cv_results_).to_csv(cv_results_path, index=False)
        mlflow.log_artifact(str(cv_results_path), artifact_path="cross_validation")

        version = "not_registered"
        if bool(config["mlflow"].get("register_model", True)):
            version = _register_with_alias(
                config["mlflow"]["registered_model_name"], model_info.model_uri
            )

        summary = {
            "run_id": run.info.run_id,
            "experiment": experiment,
            "registered_model": config["mlflow"]["registered_model_name"],
            "model_version": version,
            "best_params": search.best_params_,
            "cv_best_score": float(search.best_score_),
            "test_metrics": metrics,
            "model_path": str(model_path),
        }
        summary_path = artifacts_dir / "training_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(summary_path), artifact_path="summary")

    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args().config)
