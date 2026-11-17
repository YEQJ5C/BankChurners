"""Model training module using MLflow for tracking and native model logging."""

import os
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.logger import get_logger

logger = get_logger("churn.train")


def load_params(params_path: str = "params.yaml") -> dict:
    """Load hyperparameters and configuration with strict error handling."""
    if not os.path.exists(params_path):
        logger.error(f"Konfigurációs fájl nem található: {params_path}")
        raise FileNotFoundError(f"Configuration file not found at {params_path}")

    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
        if not params:
            raise ValueError(f"Configuration file {params_path} is empty.")
        return params


def train_model() -> str:
    """Train churn model and save artifacts for DVC and MLflow Registry."""
    logger.info("-> [TRAIN] Adatok betöltése a DVC features stádiumból...")

    # MLflow beállítások
    default_uri = "sqlite:///mlflow.db"
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default_uri)
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    experiment_name = "BankChurners_Experiment"
    mlflow.set_experiment(experiment_name)

    # Paraméterek betöltése szigorú hibakezeléssel
    params = load_params()
    train_params = params.get("train", {})
    prepare_params = params.get("prepare", {})

    n_estimators = int(train_params.get("n_estimators", 100))
    max_depth_val = train_params.get("max_depth", 5)
    max_depth = int(max_depth_val) if max_depth_val is not None else None
    test_size = float(prepare_params.get("test_size", 0.2))
    random_state = int(params.get("seed", 42))

    x_path = "data/processed/X.csv"
    y_path = "data/processed/y.csv"

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        logger.error(f"Hiányzó bemeneti fájlok: {x_path} vagy {y_path}")
        raise FileNotFoundError(
            "Required processed data not found. Run 'dvc repro' first."
        )

    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"-> [MLFLOW] Tréning futás indítva, run_id: {run_id}")

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", str(max_depth))
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)  # type: ignore[arg-type]
        rec = recall_score(y_test, preds, zero_division=0)      # type: ignore[arg-type]
        f1 = f1_score(y_test, preds, zero_division=0)          # type: ignore[arg-type]

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Natív MLflow mentés (pickle alapú, stabil. Minden mással probléma volt!)
        mlflow.sklearn.log_model(  # type: ignore[attr-defined]
            model,
            "model",
            conda_env=None,
            registered_model_name="bank_churn_model",
        )
        logger.info("-> [MLFLOW] Modell logolva és regisztrálva.")

        # --- DVC KIMENETEK ---
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        pd.DataFrame(X_test).to_csv("data/processed/X_test.csv", index=False)
        pd.DataFrame({"Attrition_Flag": y_test}).to_csv(
            "data/processed/y_test.csv", index=False
        )

        with open("models/mlflow_run_id", "w", encoding="utf-8") as f:
            f.write(run_id)

        logger.info("-> [DVC] Kimenetek sikeresen elmentve!")
        return run_id


if __name__ == "__main__":
    train_model()
