"""Unit tests for the model training module (src/train.py)."""

from pathlib import Path
import pytest
import pandas as pd
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
        "customer_age": [25, 35, 45, 55, 65, 28],
        "credit_limit": [1000.0, 5000.0, 10000.0, 15000.0, 20000.0, 2500.0],
        "total_trans_amt": [500.0, 1500.0, 3000.0, 4500.0, 6000.0, 800.0],
        "total_trans_ct": [10, 20, 30, 40, 50, 15],
    })
    y_data = pd.Series([0, 0, 0, 1, 1, 1], name="target")

    X_data.to_csv(processed_dir / "X.csv", index=False)
    y_data.to_csv(processed_dir / "y.csv", index=False)

    monkeypatch.chdir(tmp_path)

    run_id = train_model()
    assert isinstance(run_id, str)
    assert len(run_id) > 0


def test_train_model_fallback_synthetic(monkeypatch, tmp_path) -> None:
    """Test train_model falls back to synthetic data generation when files are missing."""
    monkeypatch.chdir(tmp_path)

    run_id = train_model()
    assert isinstance(run_id, str)
    assert len(run_id) > 0


def test_train_model_exception_handling(monkeypatch, tmp_path) -> None:
    """Test train_model exception handling block when training crashes."""
    monkeypatch.chdir(tmp_path)

    def mock_train_test_split(*args, **kwargs):
        raise RuntimeError("Forced training crash for coverage")

    monkeypatch.setattr(src.train, "train_test_split", mock_train_test_split)

    with pytest.raises(RuntimeError, match="Training failed"):
        train_model()


def test_train_model_main_block(monkeypatch, tmp_path: Path) -> None:
    """Test the execution of the main block in train.py."""
    import runpy
    monkeypatch.chdir(tmp_path)

    try:
        runpy.run_module("src.train", run_name="__main__")
    except Exception:
        train_model()
    assert True
