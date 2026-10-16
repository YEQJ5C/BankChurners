"""
Data Preparation Module for BankChurners Pipeline.
Handles raw data loading, schema validation, fail-fast checks,
and persisted cleaning.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import settings
from src.logger import get_logger

# A központi egyedi logger használata
logger = get_logger("churn.prepare")

REQUIRED_COLUMNS = [
    "Customer_ID",
    "Age",
    "Gender",
    "Dependent_Count",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Utilization_Ratio",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Customer_Status",
]

ALLOWED_STATUSES = {
    "Existing Customer",
    "Attrited Customer",
}


def prepare_data(
    input_path: str | Path,
    output_path: str | Path,
) -> None:
    input_file = Path(input_path)
    output_file = Path(output_path)

    logger.info(
        "Adatok betöltése",
        extra={"input_file": str(input_file)},
    )

    if not input_file.exists():
        raise FileNotFoundError(
            "Hiba: A megadott nyers adatfájl "
            f"nem található: {input_file}"
        )

    df = pd.read_csv(input_file)

    logger.info(
        "Adatok betöltve",
        extra={
            "row_count": len(df),
            "column_count": len(df.columns),
        },
    )
    if df.empty:
        raise ValueError("Hiba: A betöltött DataFrame üres!")

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(
                f"Hiba: A kötelező '{col}' oszlop "
                "hiányzik a datasetből!"
            )

    logger.info(
        "Minden kötelező oszlop sikeresen "
        "ellenőrizve."
    )

    # 1. Azonosító (Customer_ID) egyediség ellenőrzése
    if df["Customer_ID"].duplicated().any():  # type: ignore
        raise ValueError(
            "Hiba: Duplikált Customer_ID "
            "észlelhető az adatok között!"
        )
    logger.info(
        "Customer_ID egyediség validálva.",
        extra={"column": "Customer_ID"},
    )

    # 2. Null értékek és értéktartományok validálása az Age-re
    if not df["Age"].notna().all():  # type: ignore
        raise ValueError(
            "Hiba: Hiányzó (NaN) érték "
            "található az 'Age' oszlopban!"
        )

    if df["Age"].min() < 18:
        raise ValueError(
            "Hiba: Nem engedélyezett 18 év "
            "alatti ügyfél az adatok között!"
        )
    if df["Age"].max() > 100:
        raise ValueError(
            "Hiba: Valószínűtlen életkor "
            "(> 100 év) észlelhető."
        )
    logger.info(
        "Életkor tartományok rendben",
        extra={
            "min_age": int(df["Age"].min()),
            "max_age": int(df["Age"].max()),
        },
    )

    # 3. Hitelkeret és utilizációs arány ellenőrzés
    if df["Credit_Limit"].min() <= 0:
        raise ValueError(
            "Hiba: Létezik 0 vagy negatív "
            "hitelkeret!"
        )

    if not df["Avg_Utilization_Ratio"].between(
        0, 1
    ).all():  # type: ignore
        raise ValueError(
            "Hiba: Az Avg_Utilization_Ratio "
            "kívül esik a [0, 1] intervallumon!"
        )
    logger.info(
        "Hitelkeret és utilizációs mutatók "
        "validálva."
    )

    # 3b. Tranzakciós mutatók tartományának ellenőrzése
    if df["Total_Trans_Ct"].min() < 0:
        raise ValueError(
            "Hiba: Negatív tranzakciószám "
            "(Total_Trans_Ct) észlelhető!"
        )
    if df["Total_Trans_Amt"].min() < 0:
        raise ValueError(
            "Hiba: Negatív tranzakcióösszeg "
            "(Total_Trans_Amt) észlelhető!"
        )
    logger.info(
        "Tranzakciós mutatók tartománya "
        "validálva."
    )

    # 4. Target változó (Customer_Status) értékek ellenőrzése
    if not df["Customer_Status"].isin(
        ALLOWED_STATUSES  # type: ignore
    ).all():  # type: ignore
        raise ValueError(
            "Hiba: Érvénytelen Customer_Status "
            "érték található az adatok között!"
        )
    logger.info(
        "Target változó validálva",
        extra={
            "target": "Customer_Status",
            "classes": len(ALLOWED_STATUSES),
        },
    )

    # 5. Ismeretlen értékek figyelése (tiszta strukturált logolással)
    unknown_mask = df == "Unknown"
    unknown_counts = unknown_mask.sum()
    total_unknown = unknown_counts.sum()

    if total_unknown > 0:
        logger.warning(
            "Ismeretlen értékek észlelve",
            extra={
                "unknown_count": int(
                    total_unknown
                )
            },
        )
        breakdown = {
            col: int(count)
            for col, count in unknown_counts.items()
            if count > 0
        }
        logger.info(
            "Ismeretlen értékek oszloponkénti "
            "bontása",
            extra={"breakdown": breakdown},
        )

    # 6. Célkönyvtár létrehozása (csak ha szükséges) és mentés
    if not output_file.parent.exists():
        output_file.parent.mkdir(
            parents=True, exist_ok=True
        )
        logger.info(
            "Feldolgozott adatok mappája "
            "létrehozva",
            extra={
                "output_dir": str(
                    output_file.parent
                )
            },
        )

    df.to_csv(output_file, index=False)
    logger.info(
        "Adatok sikeresen elmentve",
        extra={"output_file": str(output_file)},
    )


if __name__ == "__main__":
    RAW_DATA = (
        settings.raw_dir / "BankChurners.csv"
    )
    PROCESSED_DATA = (
        settings.processed_dir
        / "bank_churners_processed.csv"
    )

    prepare_data(RAW_DATA, PROCESSED_DATA)
