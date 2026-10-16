"""Feature Engineering Module for BankChurners Pipeline.

Handles categorical encoding while preserving 'Unknown' categories,
separates features and target, validates target mapping, and saves
processed matrices.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import settings
from src.logger import get_logger

logger = get_logger("churn.features")


def process_features(
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Loads prepared data, separates features and target, encodes

    categorical variables while preserving 'Unknown' levels, and
    saves the full processed X and y matrices.

    Returns:
        tuple[Path, Path]: Paths to the saved X.csv and y.csv files.
    """
    if input_path is None:
        input_path = (
            settings.processed_dir / "bank_churners_processed.csv"
        )
    else:
        input_path = Path(input_path)

    if output_dir is None:
        output_dir = settings.processed_dir
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Prepared adatok betöltése",
        extra={"input_file": str(input_path)},
    )
    df = pd.read_csv(input_path)

    # Drop Customer_ID if present
    if "Customer_ID" in df.columns:
        df = df.drop(columns=["Customer_ID"])

    # Verify target column existence
    if "Customer_Status" not in df.columns:
        logger.error(
            "A 'Customer_Status' célváltozó hiányzik a datasetből."
        )
        raise ValueError(
            "Hiba: A 'Customer_Status' célváltozó hiányzik a datasetből."
        )

    # Explicit mapping for target variable
    y = df["Customer_Status"].map(
        {  # type: ignore[reportArgumentType] - Explicit mapping dict for target series.
            "Existing Customer": 0,
            "Attrited Customer": 1,
        }
    )

    # Validate target values
    if y.isna().any():  # type: ignore[reportAttributeAccessIssue] - Pandas Series isna/any check.
        logger.error(
            "Ismeretlen Customer_Status érték található az adatokban."
        )
        raise ValueError(
            "Hiba: Ismeretlen Customer_Status érték található."
        )

    X = df.drop(columns=["Customer_Status"])

    # Ensure categorical columns treat 'Unknown' as a valid category
    cat_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    logger.info(
        "Kategorikus oszlopok azonosítva",
        extra={"cat_cols": cat_cols},
    )

    for col in cat_cols:
        X[col] = X[col].fillna("Unknown").astype(str)

    # One-hot encode categorical features while keeping 'Unknown' levels
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    # Define output paths for full processed matrices
    x_path = output_dir / "X.csv"
    y_path = output_dir / "y.csv"

    # Save matrices
    X_encoded.to_csv(x_path, index=False)
    y.to_csv(y_path, index=False)

    logger.info(
        "Processált adatok sikeresen elmentve",
        extra={"x_path": str(x_path), "y_path": str(y_path)},
    )
    return x_path, y_path


if __name__ == "__main__":
    process_features()
