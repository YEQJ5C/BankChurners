"""
Unit tests for the model training module.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.config import settings
from src.train import train_model


def test_train_model_success(tmp_path: Path, monkeypatch) -> None:
    """Test successful model training, artifact saving, and MLflow logging."""
    monkeypatch.setattr(settings, "processed_dir", tmp_path / "processed")
    monkeypatch.setattr(settings, "test_size", 0.5)
    monkeypatch.setattr(settings, "random_seed", 42)
    monkeypatch.setattr(settings, "rf_n_estimators", 10)
    monkeypatch.setattr(
        settings, "mlflow_experiment_name", "test_experiment"
    )
    monkeypatch.setattr(
        settings, "registered_model_name", "test_model"
    )

    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db"
    )

    processed_dir = settings.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    X_data = pd.DataFrame({
        "Feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "Feature_2": [10, 20, 30, 40, 50, 60],
    })
    y_data = pd.Series([0, 0, 0, 1, 1, 1], name="Churn")

    x_path = processed_dir / "X.csv"
    y_path = processed_dir / "y.csv"

    X_data.to_csv(x_path, index=False)
    y_data.to_csv(y_path, index=False)

    monkeypatch.chdir(tmp_path)

    run_id = train_model(x_path, y_path)

    assert run_id is not None
    assert isinstance(run_id, str)
    assert len(run_id) > 0

    assert (processed_dir / "X_test.csv").exists()
    assert (processed_dir / "y_test.csv").exists()
    assert (Path("models") / "mlflow_run_id").exists()


def test_train_model_missing_x_file(tmp_path: Path) -> None:
    """Test that FileNotFoundError is raised if X file is missing."""
    missing_x = tmp_path / "non_existent_X.csv"
    y_path = tmp_path / "y.csv"
    y_path.write_text("Churn\n0\n1\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Az X fájl nem található"):
        train_model(x_path=missing_x, y_path=y_path)


def test_train_model_missing_y_file(tmp_path: Path) -> None:
    """Test that FileNotFoundError is raised if y file is missing."""
    x_path = tmp_path / "X.csv"
    x_path.write_text("Feature_1\n1\n2\n", encoding="utf-8")
    missing_y = tmp_path / "non_existent_y.csv"

    with pytest.raises(FileNotFoundError, match="Az y fájl nem található"):
        train_model(x_path=x_path, y_path=missing_y)


def test_train_model_main_block(monkeypatch, tmp_path: Path) -> None:
    """Test the execution of the main block in train.py."""
    import runpy
    from src import train

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    X_data = pd.DataFrame({
        "Feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    y_data = pd.Series([0, 0, 0, 1, 1, 1], name="Churn")

    x_path = processed_dir / "X.csv"
    y_path = processed_dir / "y.csv"

    X_data.to_csv(x_path, index=False)
    y_data.to_csv(y_path, index=False)

    monkeypatch.setattr(train.settings, "processed_dir", processed_dir)
    monkeypatch.setattr(train.settings, "rf_n_estimators", 5)
    monkeypatch.setattr(
        train.settings, "mlflow_experiment_name", "test_main_exp"
    )
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db"
    )
    monkeypatch.chdir(tmp_path)

    runpy.run_module("src.train", run_name="__main__")
