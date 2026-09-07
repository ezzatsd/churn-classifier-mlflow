"""Shared loading, validation, splitting and evaluation helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and reject missing top-level sections early."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {"data", "features", "model", "cv", "mlflow"}
    missing = required.difference(config or {})
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")
    return config


def resolve_path(config_path: str | Path, configured_path: str) -> Path:
    """Resolve project-relative paths from configs/config.yaml."""
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return Path(config_path).resolve().parent.parent / path


def load_dataset(config: dict[str, Any], config_path: str | Path) -> pd.DataFrame:
    """Load and clean source-specific data types without fitting transformations."""
    csv_path = resolve_path(config_path, config["data"]["csv_path"])
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Run `make data` first."
        )
    data = pd.read_csv(csv_path)
    target = config["data"]["target"]
    required = set(config["features"]["numeric"])
    required.update(config["features"]["categorical"])
    required.add(target)
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset columns missing from CSV: {sorted(missing)}")

    # IBM stores 11 blank TotalCharges values as strings. Convert them to NaN;
    # the median imputer in the pipeline handles them without leaking test data.
    for column in config["features"]["numeric"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def split_dataset(
    data: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the deterministic stratified holdout split."""
    target = config["data"]["target"]
    features = config["features"]["numeric"] + config["features"]["categorical"]
    X = data[features].copy()
    positive_label = config["data"].get("positive_label", "Yes")
    y = (data[target].astype(str).str.strip() == str(positive_label)).astype(int)
    if y.nunique() != 2:
        raise ValueError("The configured target must contain exactly two classes.")
    return train_test_split(
        X,
        y,
        test_size=float(config["data"]["test_size"]),
        random_state=int(config["data"]["random_state"]),
        stratify=y,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def classification_metrics(y_true: pd.Series, y_probability: np.ndarray) -> dict[str, float]:
    """Return threshold and ranking metrics for the positive class."""
    predictions = (y_probability >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "average_precision": float(average_precision_score(y_true, y_probability)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def save_evaluation_artifacts(
    y_true: pd.Series,
    y_probability: np.ndarray,
    output_dir: str | Path,
    row_ids: pd.Index | None = None,
) -> dict[str, Path]:
    """Save publication-quality plots, metrics and row-level predictions."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        PrecisionRecallDisplay,
        RocCurveDisplay,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    predictions = (y_probability >= 0.5).astype(int)
    metrics = classification_metrics(y_true, y_probability)

    plt.style.use("seaborn-v0_8-whitegrid")
    plots: list[tuple[str, Any, str]] = [
        ("roc_curve.png", RocCurveDisplay.from_predictions, "ROC curve"),
        (
            "precision_recall_curve.png",
            PrecisionRecallDisplay.from_predictions,
            "Precision-recall curve",
        ),
    ]
    paths: dict[str, Path] = {}
    for filename, display, title in plots:
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        display(y_true, y_probability, ax=ax, name="Churn classifier")
        ax.set_title(title, fontweight="bold")
        fig.tight_layout()
        path = destination / filename
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths[filename] = path

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["No churn", "Churn"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Confusion matrix at threshold 0.50", fontweight="bold")
    fig.tight_layout()
    confusion_path = destination / "confusion_matrix.png"
    fig.savefig(confusion_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["confusion_matrix.png"] = confusion_path

    metrics_path = destination / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    paths["metrics.json"] = metrics_path

    prediction_frame = pd.DataFrame(
        {
            "row_id": row_ids if row_ids is not None else y_true.index,
            "actual_churn": np.asarray(y_true),
            "predicted_churn": predictions,
            "churn_probability": y_probability,
            "error": np.asarray(y_true) != predictions,
        }
    )
    predictions_path = destination / "predictions.csv"
    prediction_frame.to_csv(predictions_path, index=False)
    paths["predictions.csv"] = predictions_path
    return paths
