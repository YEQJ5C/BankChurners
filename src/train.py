"""Model training module using MLflow for tracking and native model logging."""

import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.logger import get_logger

logger = get_logger("churn.train")


def train_model() -> str:
    """Train the churn prediction model and save required artifacts for DVC and MLflow Registry."""
    logger.info("-> [TRAIN] Adatok betöltése a DVC features stádiumból...")

    # MLflow beállítások
    default_uri = "sqlite:///mlflow.db"
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default_uri)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    experiment_name = "BankChurners_Experiment"
    mlflow.set_experiment(experiment_name)

    # A features stádium által generált X.csv és y.csv betöltése
    x_path = "data/processed/X.csv"
    y_path = "data/processed/y.csv"
    
    try:
        if os.path.exists(x_path) and os.path.exists(y_path):
            X = pd.read_csv(x_path)
            y = pd.read_csv(y_path)
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]
        else:
            logger.warning("A feldolgozott X.csv vagy y.csv nem található. Szintetikus adatok generálása...")
            np.random.seed(42)
            n_samples = 200
            X = pd.DataFrame({
                "customer_age": np.random.randint(20, 70, size=n_samples),
                "credit_limit": np.random.uniform(1000, 20000, size=n_samples),
                "total_trans_amt": np.random.uniform(500, 5000, size=n_samples),
                "total_trans_ct": np.random.randint(10, 100, size=n_samples)
            })
            y = pd.Series(np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"-> [MLFLOW] Tréning futás indítva, run_id: {run_id}")

            # Hiperparaméterek
            n_estimators = 100
            max_depth = 5
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            # Modell tanítása párhuzamosan
            model = RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                n_jobs=-1, 
                random_state=42
            )
            model.fit(X_train, y_train)

            # Predikciók és metrikák
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            # Natív MLflow mentés és regisztráció a Model Registry-be
            mlflow.sklearn.log_model(
                model, 
                "model", 
                conda_env=None,
                registered_model_name="bank_churn_model"
            )
            logger.info(f"-> [MLFLOW] Modell sikeresen logolva és regisztrálva. Run ID: {run_id}")

            # --- DVC KIMENETEK LÉTREHOZÁSA ---
            os.makedirs("data/processed", exist_ok=True)
            os.makedirs("models", exist_ok=True)

            pd.DataFrame(X_test).to_csv("data/processed/X_test.csv", index=False)
            pd.DataFrame({"Attrition_Flag": y_test}).to_csv("data/processed/y_test.csv", index=False)

            with open("models/mlflow_run_id", "w") as f:
                f.write(run_id)

            logger.info("-> [DVC] Kimenetek (X_test.csv, y_test.csv, mlflow_run_id) sikeresen elmentve!")
            return run_id

    except Exception as e:
        logger.error(f"Hiba történt a tréning során: {e}")
        raise RuntimeError(f"Training failed: {str(e)}")


if __name__ == "__main__":
    train_model()
