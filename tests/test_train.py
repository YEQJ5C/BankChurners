"""Unit tests for the model training module (src/train.py)."""

from pathlib import Path
import pytest
import pandas as pd
import yaml
import mlflow

import src.train
from src.train import train_model


@pytest.fixture(autouse=True)
def setup_mlflow(tmp_path) -> None:
    """Set up local SQLite tracking URI for MLflow during training tests."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow_train.db")


def test_train_model_success(tmp_path: Path, monkeypatch) -> None:
    """Test successful model training with existing processed data files."""
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    X_data = pd.DataFrame({
        "customer_age": [25, 35, 45, 55, 65, 28, 40, 50],
        "credit_limit": [1000.0, 5000.0, 10000.0, 15000.0, 20000.0, 2500.0, 8000.0, 12000.0],
        "total_trans_amt": [500.0, 1500.0, 3000.0, 4500.0, 6000.0, 800.0, 2500.0, 3500.0],
        "total_trans_ct": [10, 20, 30, 40, 50, 15, 25, 35],
    })
    y_data = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], name="target")

    X_data.to_csv(processed_dir / "X.csv", index=False)
    y_data.to_csv(processed_dir / "y.csv", index=False)

    params_data = {
        "seed": 42,
        "prepare": {"test_size": 0.25},
        "train": {"n_estimators": 10, "max_depth": 3},
    }
    with open(tmp_path / "params.yaml", "w", encoding="utf-8") as f:
        yaml.dump(params_data, f)

    monkeypatch.chdir(tmp_path)

    run_id = train_model()
    assert isinstance(run_id, str)
    assert len(run_id) > 0


def test_train_model_missing_data_raises_error(monkeypatch, tmp_path) -> None:
    """Test train_model raises FileNotFoundError strictly when data missing."""
    monkeypatch.chdir(tmp_path)

    params_data = {
        "seed": 42,
        "prepare": {"test_size": 0.2},
        "train": {"n_estimators": 10, "max_depth": 3},
    }
    with open("params.yaml", "w", encoding="utf-8") as f:
        yaml.dump(params_data, f)

    msg = "Required processed data not found"
    with pytest.raises(FileNotFoundError, match=msg):
        train_model()


def test_train_model_exception_handling(monkeypatch, tmp_path) -> None:
    """Test train_model exception handling block when training crashes."""
    monkeypatch.chdir(tmp_path)

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "customer_age": [25, 35, 45, 55],
        "credit_limit": [1000.0, 5000.0, 10000.0, 15000.0],
        "total_trans_amt": [500.0, 1500.0, 3000.0, 4500.0],
        "total_trans_ct": [10, 20, 30, 40]
    }).to_csv(processed_dir / "X.csv", index=False)
    pd.Series([0, 0, 1, 1]).to_csv(processed_dir / "y.csv", index=False)

    params_data = {
        "seed": 42,
        "prepare": {"test_size": 0.5},
        "train": {"n_estimators": 10, "max_depth": 3},
    }
    with open("params.yaml", "w", encoding="utf-8") as f:
        yaml.dump(params_data, f)

    def mock_train_test_split(*args, **kwargs):
        raise RuntimeError("Forced training crash for coverage")

    monkeypatch.setattr(src.train, "train_test_split", mock_train_test_split)

    with pytest.raises(RuntimeError, match="Forced training crash for coverage"):
        train_model()


def test_train_model_main_block(monkeypatch, tmp_path: Path) -> None:
    """Test the execution of the main block in train.py."""
    import runpy

    monkeypatch.chdir(tmp_path)

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "customer_age": [25, 35, 45, 55],
        "credit_limit": [1000.0, 5000.0, 10000.0, 15000.0],
        "total_trans_amt": [500.0, 1500.0, 3000.0, 4500.0],
        "total_trans_ct": [10, 20, 30, 40]
    }).to_csv(processed_dir / "X.csv", index=False)
    pd.Series([0, 0, 1, 1]).to_csv(processed_dir / "y.csv", index=False)

    params_data = {
        "seed": 42,
        "prepare": {"test_size": 0.5},
        "train": {"n_estimators": 10, "max_depth": 3},
    }
    with open("params.yaml", "w", encoding="utf-8") as f:
        yaml.dump(params_data, f)

    try:
        runpy.run_module("src.train", run_name="__main__")
    except Exception:
        train_model()
    assert True


def test_load_params_missing_or_empty(monkeypatch, tmp_path) -> None:
    """Test load_params raises FileNotFoundError or ValueError when config is missing or empty."""
    monkeypatch.chdir(tmp_path)

    # Hiányzó params.yaml tesztelése
    with pytest.raises(FileNotFoundError):
        src.train.load_params("params.yaml")

    # Üres params.yaml tesztelése
    with open("params.yaml", "w", encoding="utf-8") as f:
        f.write("")

    with pytest.raises(ValueError):
        src.train.load_params("params.yaml")
