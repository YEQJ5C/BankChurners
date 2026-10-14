"""
Model training module for the Bank
Churn prediction pipeline.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.model_selection import (
    train_test_split,
)

from src.config import settings
from src.logger import get_logger
from src.metrics import compute_metrics

logger = get_logger("churn.train")


def train_model(
    x_path: str | Path | None = None,
    y_path: str | Path | None = None,
) -> str:
    """
    Loads features, splits data, trains
    a Random Forest model, persists test
    sets, and logs everything to MLflow.
    """
    if x_path is None:
        x_path = (
            settings.processed_dir / "X.csv"
        )
    else:
        x_path = Path(x_path)

    if y_path is None:
        y_path = (
            settings.processed_dir / "y.csv"
        )
    else:
        y_path = Path(y_path)

    if not x_path.exists():
        raise FileNotFoundError(
            f"Hiba: Az X fájl nem található: "
            f"{x_path}"
        )

    if not y_path.exists():
        raise FileNotFoundError(
            f"Hiba: Az y fájl nem található: "
            f"{y_path}"
        )

    logger.info(
        "Adatok betöltése modelltanításhoz",
        extra={
            "x_path": str(x_path),
            "y_path": str(y_path),
        },
    )
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).squeeze(
        "columns"
    )

    logger.info(
        "Tanítási adatok betöltve",
        extra={
            "rows": len(X),
            "features": len(X.columns),
            "target_distribution": (
                y.value_counts().to_dict()
            ),
        },
    )

    logger.info(
        "Train/test split végrehajtása",
        extra={
            "test_size": settings.test_size,
            "random_seed": (
                settings.random_seed
            ),
        },
    )
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=settings.test_size,
            random_state=(
                settings.random_seed
            ),
            stratify=y,
        )
    )

    logger.info(
        "Train/test split kész",
        extra={
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        },
    )

    settings.processed_dir.mkdir(
        parents=True, exist_ok=True
    )
    x_test_path = (
        settings.processed_dir / "X_test.csv"
    )
    y_test_path = (
        settings.processed_dir / "y_test.csv"
    )

    X_test.to_csv(
        x_test_path, index=False
    )
    y_test.to_csv(
        y_test_path, index=False
    )

    logger.info(
        "Teszthalmazok sikeresen elmentve "
        "a lemezre az evaluate számára",
        extra={
            "x_test_path": str(
                x_test_path
            ),
            "y_test_path": str(
                y_test_path
            ),
        },
    )

    n_estimators = (
        settings.rf_n_estimators
    )

    mlflow.set_experiment(
        settings.mlflow_experiment_name
    )

    with mlflow.start_run() as run:
        mlflow.log_param(
            "n_estimators", n_estimators
        )
        mlflow.log_param(
            "random_seed",
            settings.random_seed,
        )
        mlflow.log_param(
            "test_size", settings.test_size
        )

        logger.info(
            "Random Forest modell "
            "inicializálása",
            extra={
                "algorithm": (
                    "RandomForestClassifier"
                ),
                "n_estimators": (
                    n_estimators
                ),
            },
        )
        model = RandomForestClassifier(
            random_state=(
                settings.random_seed
            ),
            n_estimators=n_estimators,
        )
        model.fit(X_train, y_train)

        logger.info(
            "Modell betanítva"
        )

        metrics = compute_metrics(
            model, X_test, y_test
        )

        logger.info(
            "Kiértékelés elkészült",
            extra=metrics,
        )

        mlflow.log_metrics(metrics)

        artifact_path = "model"
        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=(
                settings.registered_model_name
            ),
        )

        logger.info(
            "Modell sikeresen logolva "
            "MLflow-ba",
            extra={
                "artifact_path": (
                    artifact_path
                ),
                "registered_model_name": (
                    settings.registered_model_name
                ),
            },
        )

        run_id = run.info.run_id

    logger.info(
        "MLflow futás befejeződve",
        extra={"run_id": run_id},
    )

    logger.info(
        "Modelltanítás sikeresen "
        "befejeződött"
    )
    return run_id


if __name__ == "__main__":
    train_model()
