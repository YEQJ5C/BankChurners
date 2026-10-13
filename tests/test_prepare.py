"""Unit tests for the data preparation module (src/prepare.py).

Covers schema validation checks, fail-fast exceptions, unknown values
warning branch, nested output directory creation, negative transaction checks,
and the __main__ execution block.
"""

import runpy
import pandas as pd
import pytest

from src.prepare import prepare_data


def test_prepare_data_file_not_found(tmp_path):
    """Missing input file should raise FileNotFoundError."""
    missing_file = tmp_path / "non_existent.csv"
    output_file = tmp_path / "output.csv"

    with pytest.raises(FileNotFoundError):
        prepare_data(missing_file, output_file)


@pytest.fixture
def valid_base_df():
    """Fixture providing a valid minimal DataFrame for schema rules."""
    return pd.DataFrame({
        "Customer_ID": [1, 2],
        "Age": [30, 45],
        "Gender": ["M", "F"],
        "Dependent_Count": [1, 2],
        "Education_Level": ["Graduate", "College"],
        "Marital_Status": ["Single", "Married"],
        "Income_Category": ["$40K - $60K", "$60K - $80K"],
        "Card_Category": ["Blue", "Blue"],
        "Credit_Limit": [5000.0, 10000.0],
        "Total_Revolving_Bal": [1000, 1500],
        "Avg_Utilization_Ratio": [0.2, 0.5],
        "Total_Trans_Amt": [1500.0, 3000.0],
        "Total_Trans_Ct": [50, 80],
        "Customer_Status": [
            "Existing Customer",
            "Attrited Customer",
        ],
    })


def test_prepare_data_success_with_unknowns(tmp_path, valid_base_df):
    """Successful prep with 'Unknown' values and nested output dir."""
    input_file = tmp_path / "valid_input.csv"
    output_file = (
        tmp_path / "nested" / "subfolder" / "processed_output.csv"
    )

    df_with_unknowns = valid_base_df.copy()
    df_with_unknowns.loc[0, "Education_Level"] = "Unknown"

    df_with_unknowns.to_csv(input_file, index=False)

    prepare_data(input_file, output_file)

    assert output_file.exists(), "A kimeneti fájlnak léteznie kell!"
    df_result = pd.read_csv(output_file)
    assert not df_result.empty, "A kimeneti DataFrame nem lehet üres!"
    assert len(df_result) == len(valid_base_df)
    assert "Customer_Status" in df_result.columns


def test_prepare_data_empty_dataframe(tmp_path, valid_base_df):
    """Empty dataframe should raise ValueError."""
    input_file = tmp_path / "empty.csv"
    output_file = tmp_path / "output.csv"

    valid_base_df.iloc[0:0].to_csv(input_file, index=False)

    with pytest.raises(ValueError, match="üres"):
        prepare_data(input_file, output_file)


def test_prepare_data_missing_column(tmp_path, valid_base_df):
    """Missing mandatory column should raise ValueError."""
    input_file = tmp_path / "missing_col.csv"
    output_file = tmp_path / "output.csv"

    df_bad = valid_base_df.drop(columns=["Age"])
    df_bad.to_csv(input_file, index=False)

    with pytest.raises(ValueError, match="hiányzik a datasetből"):
        prepare_data(input_file, output_file)


def test_prepare_data_duplicate_ids(tmp_path, valid_base_df):
    """Duplicate customer IDs should raise ValueError."""
    input_file = tmp_path / "duplicates.csv"
    output_file = tmp_path / "output.csv"

    df_bad = valid_base_df.copy()
    df_bad.loc[1, "Customer_ID"] = df_bad.loc[0, "Customer_ID"]
    df_bad.to_csv(input_file, index=False)

    with pytest.raises(ValueError, match="[Dd]uplikált"):
        prepare_data(input_file, output_file)


def test_prepare_data_null_age(tmp_path, valid_base_df):
    """Null values in Age should raise ValueError."""
    input_file = tmp_path / "null_age.csv"
    output_file = tmp_path / "output.csv"

    df_bad = valid_base_df.copy()
    df_bad.loc[0, "Age"] = None
    df_bad.to_csv(input_file, index=False)

    with pytest.raises(ValueError):
        prepare_data(input_file, output_file)


def test_prepare_data_invalid_age_min(tmp_path, valid_base_df):
    """Age under 18 should raise ValueError."""
    input_file = tmp_path / "young_age.csv"
    output_file = tmp_path / "output.csv"

    df_bad = valid_base_df.copy()
    df_bad.loc[0, "Age"] = 16
    df_bad.to_csv(input_file, index=False)

    with pytest.raises(ValueError):
        prepare_data(input_file, output_file)


def test_prepare_data_invalid_age_max(tmp_path, valid_base_df):
    """Age over 100 should raise ValueError."""
    input_file = tmp_path / "old_age.csv"
    output_file = tmp_path / "output.csv"

    df_bad = valid_base_df.copy()
    df_bad.loc[0, "Age"] = 105
    df_bad.to_csv(input_file, index=False)

    with pytest.raises(ValueError):
        prepare_data(input_file, output_file)


def test_prepare_data_invalid_credit_limit(tmp_path, valid_base_df):
    """Non-positive credit limit should raise ValueError."""
    input_file = tmp_path / "bad_credit.csv"
    output_file = tmp_path / "output.csv"

    df_bad = valid_base_df.copy()
    df_bad.loc[0, "Credit_Limit"] = 0.0
    df_bad.to_csv(input_file, index=False)

    with pytest.raises(ValueError):
        prepare_data(input_file, output_file)


def test_prepare_data_invalid_utilization(tmp_path, valid_base_df):
    """Utilization ratio outside [0, 1] should raise ValueError."""
    input_file = tmp_path / "bad_util.csv"
    output_file = tmp_path / "output.csv"

    df_bad = valid_base_df.copy()
    df_bad.loc[0, "Avg_Utilization_Ratio"] = 1.5
    df_bad.to_csv(input_file, index=False)

    with pytest.raises(ValueError):
        prepare_data(input_file, output_file)


def test_prepare_data_negative_transactions_count(
    tmp_path, valid_base_df
):
    """Negative transaction count should raise ValueError."""
    input_file = tmp_path / "bad_trans_ct.csv"
    output_file = tmp_path / "output.csv"

    df_bad = valid_base_df.copy()
    df_bad.loc[0, "Total_Trans_Ct"] = -5
    df_bad.to_csv(input_file, index=False)

    with pytest.raises(ValueError):
        prepare_data(input_file, output_file)


def test_prepare_data_negative_transactions_amount(
    tmp_path, valid_base_df
):
    """Negative transaction amount should raise ValueError."""
    input_file = tmp_path / "bad_trans_amt.csv"
    output_file = tmp_path / "output.csv"

    df_bad = valid_base_df.copy()
    df_bad.loc[0, "Total_Trans_Amt"] = -100.0
    df_bad.to_csv(input_file, index=False)

    with pytest.raises(ValueError):
        prepare_data(input_file, output_file)


def test_prepare_data_invalid_status(tmp_path, valid_base_df):
    """Invalid customer status should raise ValueError."""
    input_file = tmp_path / "bad_status.csv"
    output_file = tmp_path / "output.csv"

    df_bad = valid_base_df.copy()
    df_bad.loc[0, "Customer_Status"] = "Unknown_Status"
    df_bad.to_csv(input_file, index=False)

    with pytest.raises(ValueError):
        prepare_data(input_file, output_file)


def test_prepare_main_block(tmp_path, monkeypatch, valid_base_df):
    """Test the if __name__ == '__main__': execution block."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()

    input_file = raw_dir / "BankChurners.csv"
    valid_base_df.to_csv(input_file, index=False)

    monkeypatch.setattr("src.prepare.settings.raw_dir", raw_dir)
    monkeypatch.setattr("src.prepare.settings.processed_dir", processed_dir)

    runpy.run_module("src.prepare", run_name="__main__")

    assert (processed_dir / "bank_churners_processed.csv").exists()
