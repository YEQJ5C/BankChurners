"""Unit tests for the model training module."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from src.train import train_model
import src.train


def test_train_model_success(tmp_path: Path, monkeypatch) -> None:
    """Test successful model training, artifact saving, and MLflow logging."""
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    X_data = pd.DataFrame({
        "Feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "Feature_2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    y_data = pd.Series([0, 0, 0, 1, 1, 1], name="Attrition_Flag")

    x_file = processed_dir / "X.csv"
    y_file = processed_dir / "y.csv"
    X_data.to_csv(x_file, index=False)
    y_data.to_csv(y_file, index=False)

    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        src.train,
        "load_params",
        lambda: {
            "seed": 42,
            "prepare": {"test_size": 0.5},
            "train": {
                "n_estimators": 5,
                "max_depth": 3,
                "min_samples_split": 2,
            },
        },
    )

    class MockRun:
        info = type("obj", (object,), {"run_id": "mock_run_abc_123"})()

    monkeypatch.setattr(
        src.train.mlflow,
        "start_run",
        lambda: type(
            "obj",
            (object,),
            {
                "__enter__": lambda s: MockRun(),
                "__exit__": lambda s, exc_type, exc_val, exc_tb: None,
            },
        )(),
    )
    monkeypatch.setattr(src.train.mlflow, "set_experiment", lambda name: None)
    monkeypatch.setattr(src.train.mlflow, "log_param", lambda k, v: None)
    monkeypatch.setattr(src.train.mlflow, "log_metrics", lambda m: None)
    monkeypatch.setattr(
        src.train.mlflow.sklearn,
        "log_model",
        lambda model, name, input_example, registered_model_name: None,
    )
    monkeypatch.setattr(src.train, "compute_metrics", lambda m, x, y: {"f1": 0.85})

    # Stringként adjuk át az útvonalakat, hogy a str | Path ágak (49, 54. sor) is lefedésre kerüljenek
    run_id = train_model(x_path=str(x_file), y_path=str(y_file))
    assert run_id == "mock_run_abc_123"
    assert (Path("models") / "mlflow_run_id").read_text(encoding="utf-8") == "mock_run_abc_123"


def test_train_model_missing_x_file(tmp_path: Path) -> None:
    """Test train_model raises FileNotFoundError when X.csv is missing."""
    y_file = tmp_path / "y.csv"
    pd.Series([0, 1]).to_csv(y_file, index=False)

    with pytest.raises(FileNotFoundError):
        train_model(x_path=tmp_path / "non_existent_X.csv", y_path=y_file)


def test_train_model_missing_y_file(tmp_path: Path) -> None:
    """Test train_model raises FileNotFoundError when y.csv is missing."""
    x_file = tmp_path / "X.csv"
    pd.DataFrame({"A": [1, 2]}).to_csv(x_file, index=False)

    with pytest.raises(FileNotFoundError):
        train_model(x_path=x_file, y_path=tmp_path / "non_existent_y.csv")


def test_train_model_main_block(monkeypatch, tmp_path: Path) -> None:
    """Test the execution of the main block in train.py."""
    import runpy

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    X_data = pd.DataFrame({
        "Feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    y_data = pd.Series([0, 0, 0, 1, 1, 1], name="Churn")

    x_file = processed_dir / "X.csv"
    y_file = processed_dir / "y.csv"
    X_data.to_csv(x_file, index=False)
    y_data.to_csv(y_file, index=False)

    monkeypatch.setattr(src.train.settings, "processed_dir", processed_dir)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        src.train,
        "load_params",
        lambda: {
            "seed": 42,
            "prepare": {"test_size": 0.5},
            "train": {
                "n_estimators": 5,
                "max_depth": 3,
                "min_samples_split": 2,
            },
        },
    )

    class MockRun:
        info = type("obj", (object,), {"run_id": "mock_run_main_789"})()

    monkeypatch.setattr(
        src.train.mlflow,
        "start_run",
        lambda: type(
            "obj",
            (object,),
            {
                "__enter__": lambda s: MockRun(),
                "__exit__": lambda s, exc_type, exc_val, exc_tb: None,
            },
        )(),
    )
    monkeypatch.setattr(src.train.mlflow, "set_experiment", lambda name: None)
    monkeypatch.setattr(src.train.mlflow, "log_param", lambda k, v: None)
    monkeypatch.setattr(src.train.mlflow.sklearn, "log_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(src.train, "compute_metrics", lambda m, x, y: {"f1": 0.85})

    monkeypatch.setattr("sys.argv", ["train.py"])
    runpy.run_module("src.train", run_name="__main__")
