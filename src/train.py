"""Model training module for the Bank Churn prediction pipeline."""

from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn  # noqa: F401 - Unused import required for side-effect MLflow initialization.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config import settings
from src.logger import get_logger
from src.metrics import compute_metrics
from src.utils import load_params

logger = get_logger("churn.train")


def train_model(
    x_path: str | Path | None = None,
    y_path: str | Path | None = None,
) -> str:
    """Loads features, splits data, trains a Random Forest model,

    persists test sets, logs everything to MLflow with input examples,
    and saves the run_id to models/mlflow_run_id.
    """
    # 1. Paraméterek betöltése a params.yaml-ből (Single Source of Truth)
    params = load_params()

    seed = params.get("seed", 42)

    # A prepare.test_size a 'prepare' blokkból érkezik
    prepare_params = params.get("prepare", {})
    test_size = prepare_params.get("test_size", 0.2)

    # A train hiperparaméterek a 'train' blokkból érkeznek
    train_params = params.get("train", {})
    n_estimators = train_params.get("n_estimators", 100)
    max_depth = train_params.get("max_depth", None)
    min_samples_split = train_params.get("min_samples_split", 2)
    random_seed = train_params.get("random_seed", seed)

    if x_path is None:
        x_path = settings.processed_dir / "X.csv"
    else:
        x_path = Path(x_path)

    if y_path is None:
        y_path = settings.processed_dir / "y.csv"
    else:
        y_path = Path(y_path)

    if not x_path.exists():
        raise FileNotFoundError(
            f"Hiba: Az X fájl nem található: {x_path}"
        )

    if not y_path.exists():
        raise FileNotFoundError(
            f"Hiba: Az y fájl nem található: {y_path}"
        )

    logger.info(
        "Adatok betöltése modelltanításhoz",
        extra={
            "x_path": str(x_path),
            "y_path": str(y_path),
        },
    )
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).squeeze("columns")

    logger.info(
        "Tanítási adatok betöltve",
        extra={
            "rows": len(X),
            "features": len(X.columns),
            "target_distribution": (
                y.value_counts().to_dict()  # type: ignore[reportAttributeAccessIssue]
            ),
        },
    )

    logger.info(
        "Train/test split végrehajtása",
        extra={
            "test_size": test_size,
            "random_seed": random_seed,
        },
    )
    raw_split = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )
    X_train, X_test, y_train, y_test = raw_split  # type: ignore[reportAssignmentType]

    X_train_df: pd.DataFrame = X_train  # type: ignore[reportAssignmentType]
    X_test_df: pd.DataFrame = X_test  # type: ignore[reportAssignmentType]
    y_train_ser: pd.Series = y_train  # type: ignore[reportAssignmentType]
    y_test_ser: pd.Series = y_test  # type: ignore[reportAssignmentType]

    logger.info(
        "Train/test split kész",
        extra={
            "train_rows": len(X_train_df),
            "test_rows": len(X_test_df),
        },
    )

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    x_test_path = settings.processed_dir / "X_test.csv"
    y_test_path = settings.processed_dir / "y_test.csv"

    X_test_df.to_csv(x_test_path, index=False)
    y_test_ser.to_csv(y_test_path, index=False)

    logger.info(
        "Teszthalmazok sikeresen elmentve a lemezre az evaluate számára",
        extra={
            "x_test_path": str(x_test_path),
            "y_test_path": str(y_test_path),
        },
    )

    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run() as run:
        # 2. Minden paraméter naplózása az MLflow-ba
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", str(max_depth))
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("random_seed", random_seed)
        mlflow.log_param("test_size", test_size)

        logger.info(
            "Random Forest modell inicializálása",
            extra={
                "algorithm": "RandomForestClassifier",
                "n_estimators": n_estimators,
                "max_depth": str(max_depth),
                "min_samples_split": min_samples_split,
            },
        )

        # 3. Hiperparaméterek átadása a modellnek
        model = RandomForestClassifier(
            random_state=random_seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        model.fit(
            X_train_df, y_train_ser
        )  # type: ignore[reportArgumentType]

        logger.info("Modell betanítva")

        metrics = compute_metrics(
            model, X_test_df, y_test_ser
        )  # type: ignore[reportArgumentType]

        logger.info(
            "Kiértékelés elkészült",
            extra=metrics,
        )

        mlflow.log_metrics(metrics)

        input_example = (
            X_train_df.iloc[[0]]  # type: ignore[reportAttributeAccessIssue]
        )
        mlflow.sklearn.log_model(  # type: ignore[reportPrivateImportUsage, reportArgumentType]
            model,
            name="model",
            input_example=input_example,
            registered_model_name=(
                settings.registered_model_name
            ),
        )

        logger.info(
            "Modell sikeresen logolva MLflow-ba",
            extra={
                "registered_model_name": (
                    settings.registered_model_name
                ),
            },
        )

        run_id = run.info.run_id

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    run_id_file = models_dir / "mlflow_run_id"
    run_id_file.write_text(run_id, encoding="utf-8")

    logger.info(
        "Run ID sikeresen elmentve a models/mlflow_run_id fájlba",
        extra={
            "run_id": run_id,
            "path": str(run_id_file),
        },
    )

    logger.info(
        "MLflow futás befejeződve",
        extra={"run_id": run_id},
    )

    logger.info("Modelltanítás sikeresen befejeződött")
    return run_id


if __name__ == "__main__":
    train_model()
