"""
Model Evaluation Module for BankChurners Pipeline.
Loads a trained model and the specific test matrices (X_test, y_test),
computes comprehensive metrics using the shared metrics module,
and persists them into metrics/metrics.json safely.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.sklearn  # noqa: F401
import pandas as pd

from src.config import settings
from src.logger import get_logger
from src.metrics import compute_metrics

logger = get_logger("churn.evaluate")


def evaluate_model(
    model_uri: str | Path | None = None,
    x_test_path: str | Path | None = None,
    y_test_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, float]:
    """
    Evaluates a trained model on held-out test data.
    """
    if x_test_path is None:
        x_test_path = settings.processed_dir / "X_test.csv"
    else:
        x_test_path = Path(x_test_path)

    if y_test_path is None:
        y_test_path = settings.processed_dir / "y_test.csv"
    else:
        y_test_path = Path(y_test_path)

    if output_path is None:
        output_path = settings.metrics_dir / "metrics.json"
    else:
        output_path = Path(output_path)

    if not x_test_path.exists():
        logger.error(
            "Az X_test mátrix nem található.",
            extra={"x_test_path": str(x_test_path)},
        )
        raise FileNotFoundError(
            f"Hiba: A teszt X fájl nem található: {x_test_path}"
        )

    if not y_test_path.exists():
        logger.error(
            "Az y_test célváltozó nem található.",
            extra={"y_test_path": str(y_test_path)},
        )
        raise FileNotFoundError(
            f"Hiba: A teszt y fájl nem található: {y_test_path}"
        )

    logger.info(
        "Teszthalmaz betöltése",
        extra={
            "x_test_path": str(x_test_path),
            "y_test_path": str(y_test_path),
        },
    )
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    logger.info(
        "Teszthalmaz betöltve",
        extra={
            "rows": len(X_test),
            "features": len(X_test.columns),
        },
    )

    if model_uri is None:
        model_uri = (
            f"models:/{settings.registered_model_name}/latest"
        )

    logger.info(
        "Modell betöltése",
        extra={"model_uri": str(model_uri)},
    )
    try:
        model = mlflow.sklearn.load_model(str(model_uri))
    except Exception as e:
        logger.error(
            "Modell betöltési hiba.",
            extra={
                "model_uri": str(model_uri),
                "error": str(e),
            },
        )
        raise RuntimeError(
            f"Modell betöltési hiba ({model_uri}): {e}"
        )

    logger.info("Metrikák számítása...")
    base_metrics = compute_metrics(model, X_test, y_test)

    try:
        if pd.api.types.is_numeric_dtype(y_test):
            positive_rate = float((y_test == 1).mean())
        else:
            positive_rate = float(
                (
                    y_test.astype(str)
                    .str.lower()
                    .isin(
                        [
                            "1",
                            "true",
                            "attrited customer",
                            "yes",
                        ]
                    )
                ).mean()
            )
    except Exception:
        positive_rate = 0.0

    metrics_payload = {
        **base_metrics,
        "n_valid": len(X_test),
        "positive_rate": round(positive_rate, 4),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(
        output_path, "w", encoding="utf-8"
    ) as f:
        json.dump(metrics_payload, f, indent=2)

    logger.info(
        "Kiértékelés kész, metrics.json generálva",
        extra={
            "output_path": str(output_path),
            **metrics_payload,
        },
    )

    return metrics_payload
