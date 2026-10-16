"""Unit tests for the model evaluation module."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluate import evaluate_model
import src.evaluate
import src.metrics


class DummyModel:
    """Mock model for evaluation."""

    def predict(self, X):
        return [0] * len(X)

    def predict_proba(self, X):
        return np.array([[0.8, 0.2]] * len(X))


def test_evaluate_model_success(monkeypatch, tmp_path: Path) -> None:
    """Test successful model evaluation, metric computation, and artifact saving."""
    processed_dir = tmp_path / "processed"
    metrics_dir = tmp_path / "metrics"
    models_dir = tmp_path / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    X_test = pd.DataFrame({"Feature_1": [1.0, 2.0]})
    y_test = pd.Series([0, 1])

    x_path = processed_dir / "X_test.csv"
    y_path = processed_dir / "y_test.csv"
    custom_output = metrics_dir / "custom_metrics.json"

    X_test.to_csv(x_path, index=False)
    y_test.to_csv(y_path, index=False)

    monkeypatch.setattr(src.evaluate.settings, "models_dir", models_dir)
    monkeypatch.setattr(src.evaluate.mlflow.sklearn, "load_model", lambda uri: DummyModel())
    monkeypatch.setattr(src.evaluate.mlflow, "log_metrics", lambda m: None)
    monkeypatch.setattr(
        src.evaluate,
        "compute_metrics",
        lambda m, x, y: {"f1": 0.85, "accuracy": 0.90},
    )

    metrics = evaluate_model(
        x_test_path=x_path,
        y_test_path=y_path,
        output_path=custom_output,
    )
    assert isinstance(metrics, dict)
    assert metrics["f1"] == 0.85
    assert custom_output.exists()


def test_evaluate_model_positive_rate_exception(monkeypatch, tmp_path: Path) -> None:
    """Test evaluation when positive_rate calculation raises an exception (fallback to 0.0)."""
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    X_test = pd.DataFrame({"Feature_1": [1.0]})
    y_test = pd.Series([0])

    x_path = processed_dir / "X_test.csv"
    y_path = processed_dir / "y_test.csv"
    X_test.to_csv(x_path, index=False)
    y_test.to_csv(y_path, index=False)

    monkeypatch.setattr(src.evaluate.settings, "models_dir", models_dir)
    monkeypatch.setattr(src.evaluate.mlflow.sklearn, "load_model", lambda uri: DummyModel())
    monkeypatch.setattr(src.evaluate.mlflow, "log_metrics", lambda m: None)
    monkeypatch.setattr(
        src.evaluate,
        "compute_metrics",
        lambda m, x, y: {"f1": 0.85, "accuracy": 0.90},
    )

    orig_mean = pd.Series.mean
    monkeypatch.setattr(
        pd.Series,
        "mean",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Force exception"))
        if len(self) == 1 else orig_mean(self, *args, **kwargs),
    )

    metrics = evaluate_model(x_test_path=x_path, y_test_path=y_path)
    assert metrics["positive_rate"] == 0.0


def test_evaluate_model_non_numeric_y_test(monkeypatch, tmp_path: Path) -> None:
    """Test evaluation when y_test contains categorical string labels."""
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    X_test = pd.DataFrame({"Feature_1": [1.0, 2.0]})
    y_test = pd.Series(["Existing Customer", "Attrited Customer"])

    x_path = processed_dir / "X_test.csv"
    y_path = processed_dir / "y_test.csv"
    X_test.to_csv(x_path, index=False)
    y_test.to_csv(y_path, index=False)

    monkeypatch.setattr(src.evaluate.settings, "models_dir", models_dir)
    monkeypatch.setattr(src.evaluate.mlflow.sklearn, "load_model", lambda uri: DummyModel())
    monkeypatch.setattr(src.evaluate.mlflow, "log_metrics", lambda m: None)
    monkeypatch.setattr(
        src.evaluate,
        "compute_metrics",
        lambda m, x, y: {"f1": 0.85, "accuracy": 0.90},
    )

    metrics = evaluate_model(x_test_path=x_path, y_test_path=y_path)
    assert metrics["positive_rate"] == 0.5


def test_evaluate_model_missing_x_test(tmp_path: Path) -> None:
    """Test evaluate_model raises FileNotFoundError when X_test.csv is missing."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.Series([0, 1]).to_csv(processed_dir / "y_test.csv", index=False)

    with pytest.raises(FileNotFoundError):
        evaluate_model(
            x_test_path=processed_dir / "non_existent_X_test.csv",
            y_test_path=processed_dir / "y_test.csv",
        )


def test_evaluate_model_missing_y_test(tmp_path: Path) -> None:
    """Test evaluate_model raises FileNotFoundError when y_test.csv is missing."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Feature_1": [1.0, 2.0]}).to_csv(
        processed_dir / "X_test.csv", index=False
    )

    with pytest.raises(FileNotFoundError):
        evaluate_model(
            x_test_path=processed_dir / "X_test.csv",
            y_test_path=processed_dir / "non_existent_y_test.csv",
        )


def test_evaluate_model_load_failure(monkeypatch, tmp_path: Path) -> None:
    """Test evaluate_model raises RuntimeError when model loading fails."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Feature_1": [1.0]}).to_csv(processed_dir / "X_test.csv", index=False)
    pd.Series([0]).to_csv(processed_dir / "y_test.csv", index=False)

    monkeypatch.setattr(src.evaluate.settings, "processed_dir", processed_dir)

    def mock_load_error(uri):
        raise RuntimeError("MLflow load failed")

    monkeypatch.setattr(src.evaluate.mlflow.sklearn, "load_model", mock_load_error)

    with pytest.raises(RuntimeError):
        evaluate_model()


def test_evaluate_model_below_threshold(monkeypatch, tmp_path: Path) -> None:
    """Test evaluate_model raises ValueError when F1 score is below minimum gate."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Feature_1": [1.0]}).to_csv(processed_dir / "X_test.csv", index=False)
    pd.Series([0]).to_csv(processed_dir / "y_test.csv", index=False)

    monkeypatch.setattr(src.evaluate.settings, "processed_dir", processed_dir)
    monkeypatch.setattr(src.evaluate.mlflow.sklearn, "load_model", lambda uri: DummyModel())
    monkeypatch.setattr(
        src.evaluate,
        "compute_metrics",
        lambda m, x, y: {"f1": 0.20},
    )

    with pytest.raises(ValueError):
        evaluate_model()


def test_evaluate_model_main_block(monkeypatch, tmp_path: Path) -> None:
    """Test the execution of the main block in evaluate.py."""
    import runpy

    processed_dir = tmp_path / "processed"
    metrics_dir = tmp_path / "metrics"
    models_dir = tmp_path / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    X_test = pd.DataFrame({"Feature_1": [1.0, 2.0]})
    y_test = pd.Series([0, 1])

    x_path = processed_dir / "X_test.csv"
    y_path = processed_dir / "y_test.csv"
    X_test.to_csv(x_path, index=False)
    y_test.to_csv(y_path, index=False)

    monkeypatch.setattr(src.evaluate.settings, "processed_dir", processed_dir)
    monkeypatch.setattr(src.evaluate.settings, "metrics_dir", metrics_dir)
    monkeypatch.setattr(src.evaluate.settings, "models_dir", models_dir)

    monkeypatch.setattr(src.evaluate.mlflow.sklearn, "load_model", lambda uri: DummyModel())
    monkeypatch.setattr(src.evaluate.mlflow, "log_metrics", lambda m: None)

    # A forrásalapú betöltés miatt a src.metrics-beli compute_metrics-et patcheljük,
    # amit a runpy által indított modul be fog tölteni magának!
    monkeypatch.setattr(
        src.metrics,
        "compute_metrics",
        lambda m, x, y: {"f1": 0.85, "accuracy": 0.90},
    )

    monkeypatch.setattr("sys.argv", ["evaluate.py"])
    runpy.run_module("src.evaluate", run_name="__main__")
