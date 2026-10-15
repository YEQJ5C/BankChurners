"""
Unit tests for the model evaluation module.
"""

from pathlib import Path
import json
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.evaluate import evaluate_model
from src.config import settings


def test_evaluate_model_success(tmp_path: Path, monkeypatch) -> None:
    """Test successful model evaluation with numeric target."""
    x_path = tmp_path / "X_test.csv"
    y_path = tmp_path / "y_test.csv"
    out_path = tmp_path / "metrics" / "metrics.json"

    X_df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0],
        "feature_2": [10, 20, 30, 40],
    })
    y_ser = pd.Series([0, 0, 1, 1], name="Customer_Status")

    X_df.to_csv(x_path, index=False)
    y_ser.to_csv(y_path, index=False)

    model = RandomForestClassifier(random_state=42, n_estimators=5)
    model.fit(X_df, y_ser)

    monkeypatch.setattr(
        "mlflow.sklearn.load_model", lambda uri: model
    )

    metrics = evaluate_model(
        model_uri="runs:/dummy/model",
        x_test_path=x_path,
        y_test_path=y_path,
        output_path=out_path,
    )

    assert out_path.exists()
    assert "accuracy" in metrics
    assert "positive_rate" in metrics
    assert metrics["n_valid"] == 4

    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["n_valid"] == 4


def test_evaluate_model_string_target(
    tmp_path: Path, monkeypatch
) -> None:
    """Test evaluation with string categorical target (e.g., Attrited)."""
    x_path = tmp_path / "X_test.csv"
    y_path = tmp_path / "y_test.csv"
    out_path = tmp_path / "metrics.json"

    X_df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0],
    })
    y_ser = pd.Series([
        "Existing Customer",
        "Existing Customer",
        "Attrited Customer",
        "Attrited Customer",
    ])

    X_df.to_csv(x_path, index=False)
    y_ser.to_csv(y_path, index=False)

    model = RandomForestClassifier(random_state=42, n_estimators=5)
    model.fit(X_df, [0, 0, 1, 1])

    monkeypatch.setattr(
        "mlflow.sklearn.load_model", lambda uri: model
    )
    monkeypatch.setattr(
        "src.evaluate.compute_metrics",
        lambda m, x, y, **kwargs: {"accuracy": 1.0, "f1": 1.0},
    )

    metrics = evaluate_model(
        model_uri="runs:/dummy/model",
        x_test_path=x_path,
        y_test_path=y_path,
        output_path=out_path,
    )

    assert metrics["positive_rate"] == 0.5
    assert metrics["accuracy"] == 1.0


def test_evaluate_model_positive_rate_exception(
    tmp_path: Path, monkeypatch
) -> None:
    """Test positive_rate exception handling fallback to 0.0."""
    x_path = tmp_path / "X_test.csv"
    y_path = tmp_path / "y_test.csv"
    out_path = tmp_path / "metrics.json"

    X_df = pd.DataFrame({"feature_1": [1.0, 2.0]})
    y_ser = pd.Series([0, 1])

    X_df.to_csv(x_path, index=False)
    y_ser.to_csv(y_path, index=False)

    model = RandomForestClassifier(random_state=42, n_estimators=5)
    model.fit(X_df, y_ser)

    monkeypatch.setattr(
        "mlflow.sklearn.load_model", lambda uri: model
    )
    monkeypatch.setattr(
        pd.Series, "mean", lambda self: 1 / 0
    )

    metrics = evaluate_model(
        model_uri="runs:/dummy/model",
        x_test_path=x_path,
        y_test_path=y_path,
        output_path=out_path,
    )

    assert metrics["positive_rate"] == 0.0


def test_evaluate_model_missing_x(tmp_path: Path) -> None:
    """Test that FileNotFoundError is raised if X_test.csv is missing."""
    y_path = tmp_path / "y_test.csv"
    y_path.write_text("Customer_Status\n1")

    with pytest.raises(
        FileNotFoundError, match="A teszt X fájl nem található"
    ):
        evaluate_model(
            x_test_path=tmp_path / "non_existent_x.csv",
            y_test_path=y_path,
        )


def test_evaluate_model_missing_y(tmp_path: Path) -> None:
    """Test that FileNotFoundError is raised if y_test.csv is missing."""
    x_path = tmp_path / "X_test.csv"
    x_path.write_text("feature_1\n1.0")

    with pytest.raises(
        FileNotFoundError, match="A teszt y fájl nem található"
    ):
        evaluate_model(
            x_test_path=x_path,
            y_test_path=tmp_path / "non_existent_y.csv",
        )


def test_evaluate_model_load_failure(
    tmp_path: Path, monkeypatch
) -> None:
    """Test that RuntimeError is raised if MLflow load fails."""
    x_path = tmp_path / "X_test.csv"
    y_path = tmp_path / "y_test.csv"
    X_df = pd.DataFrame({"feature_1": [1.0]})
    y_ser = pd.Series([1])
    X_df.to_csv(x_path, index=False)
    y_ser.to_csv(y_path, index=False)

    def mock_load_fail(uri):
        raise ValueError("MLflow error")

    monkeypatch.setattr("mlflow.sklearn.load_model", mock_load_fail)

    with pytest.raises(RuntimeError, match="Modell betöltési hiba"):
        evaluate_model(
            x_test_path=x_path,
            y_test_path=y_path,
        )


def test_evaluate_main_block(monkeypatch, tmp_path: Path) -> None:
    """Test execution of the main block in evaluate.py."""
    import runpy

    x_path = tmp_path / "X_test.csv"
    y_path = tmp_path / "y_test.csv"
    X_df = pd.DataFrame({"feature_1": [1.0, 2.0]})
    y_ser = pd.Series([0, 1])
    X_df.to_csv(x_path, index=False)
    y_ser.to_csv(y_path, index=False)

    monkeypatch.setattr(settings, "processed_dir", tmp_path)
    monkeypatch.setattr(settings, "metrics_dir", tmp_path)
    monkeypatch.setattr(settings, "registered_model_name", "DummyModel")

    model = RandomForestClassifier(random_state=42, n_estimators=5)
    model.fit(X_df, y_ser)

    monkeypatch.setattr(
        "mlflow.sklearn.load_model", lambda uri: model
    )

    runpy.run_module("src.evaluate", run_name="__main__")
