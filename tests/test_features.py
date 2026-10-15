"""
Unit tests for the feature engineering module.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.config import settings
from src.features import process_features


def test_process_features(tmp_path: Path) -> None:
    """Test feature processing, encoding, and matrix saving."""
    data = {
        "Customer_ID": [101, 102, 103, 104],
        "Customer_Status": [
            "Existing Customer",
            "Attrited Customer",
            "Existing Customer",
            "Attrited Customer",
        ],
        "Education_Level": ["Graduate", "Unknown", "High School", "Unknown"],
        "Income_Category": [
            "Less than $40K",
            "Unknown",
            "$60K - $80K",
            "Unknown",
        ],
        "Customer_Age": [45, 50, 35, 40],
    }
    df = pd.DataFrame(data)

    input_file = tmp_path / "bank_churners_processed.csv"
    df.to_csv(input_file, index=False)

    output_dir = tmp_path / "processed"

    x_path, y_path = process_features(input_file, output_dir)

    assert x_path.exists()
    assert y_path.exists()

    X_res = pd.read_csv(x_path)
    y_res = pd.read_csv(y_path)

    assert "Customer_ID" not in X_res.columns
    assert len(X_res) == 4
    assert len(y_res) == 4

    unknown_cols = [col for col in X_res.columns if "Unknown" in col]
    assert len(unknown_cols) > 0


def test_process_features_default_paths(tmp_path: Path, monkeypatch) -> None:
    """Test feature processing using default config settings paths."""
    monkeypatch.setattr(settings, "processed_dir", tmp_path)

    data = {
        "Customer_Status": ["Existing Customer", "Attrited Customer"],
        "Education_Level": ["Graduate", "Unknown"],
        "Customer_Age": [45, 50],
    }
    df = pd.DataFrame(data)

    input_file = tmp_path / "bank_churners_processed.csv"
    df.to_csv(input_file, index=False)

    x_path, y_path = process_features()

    assert x_path.exists()
    assert y_path.exists()


def test_process_features_missing_input_file(tmp_path: Path) -> None:
    """Test that FileNotFoundError is raised if input file does not exist."""
    non_existent = tmp_path / "non_existent.csv"

    with pytest.raises(FileNotFoundError):
        process_features(non_existent, tmp_path / "processed")


def test_process_features_missing_target(tmp_path: Path) -> None:
    """Test that ValueError is raised if target column is missing."""
    df = pd.DataFrame({"Customer_Age": [30, 40]})
    input_file = tmp_path / "invalid.csv"
    df.to_csv(input_file, index=False)

    with pytest.raises(
        ValueError, match="Hiba: A 'Customer_Status' célváltozó hiányzik"
    ):
        process_features(input_file, tmp_path / "processed")


def test_process_features_invalid_target_value(tmp_path: Path) -> None:
    """Test that ValueError is raised if target contains unknown values."""
    data = {
        "Customer_Status": ["Existing Customer", "Invalid Status"],
        "Customer_Age": [30, 40],
    }
    df = pd.DataFrame(data)
    input_file = tmp_path / "invalid_target.csv"
    df.to_csv(input_file, index=False)

    with pytest.raises(
        ValueError, match="Hiba: Ismeretlen Customer_Status érték található"
    ):
        process_features(input_file, tmp_path / "processed")


def test_features_main_block(monkeypatch, tmp_path: Path) -> None:
    """Test the execution of the main block in features.py."""
    import runpy

    data = {
        "Customer_Status": ["Existing Customer", "Attrited Customer"],
        "Education_Level": ["Graduate", "Unknown"],
        "Customer_Age": [45, 50],
    }
    df = pd.DataFrame(data)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    input_file = processed_dir / "bank_churners_processed.csv"
    df.to_csv(input_file, index=False)

    monkeypatch.setattr(settings, "processed_dir", processed_dir)

    runpy.run_module("src.features", run_name="__main__")
