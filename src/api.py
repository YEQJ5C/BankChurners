"""FastAPI application module for serving the BankChurners churn prediction model."""

import os
import glob
import joblib
import subprocess
import threading
from typing import Any, Dict, Tuple, Optional
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.logger import get_logger

logger = get_logger("churn.api")

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

app = FastAPI(
    title="Bank Churners Prediction API",
    description="Production-grade MLOps API with robust subprocess training and zero-training startup.",
    version="1.0.0",
)

model: Any = None


class CustomerData(BaseModel):
    """Pydantic model representing input features for customer churn prediction."""

    customer_age: int = Field(..., ge=18, le=100, description="Customer age")
    credit_limit: float = Field(..., gt=0, description="Credit limit on account")
    total_trans_amt: float = Field(..., ge=0, description="Total transaction amount")
    total_trans_ct: int = Field(..., ge=0, description="Total transaction count")


def load_model() -> bool:
    """Load the machine learning model directly from local filesystem artifacts."""
    global model
    logger.info("-> [LOAD_MODEL] Modell közvetlen betöltése a lemezről...")

    try:
        search_patterns = ["mlruns/**/model.pkl", "/app/mlruns/**/model.pkl", "models/**/*.pkl"]
        model_paths = []
        for pattern in search_patterns:
            model_paths.extend(glob.glob(pattern, recursive=True))

        if not model_paths:
            logger.warning("Nem található .pkl modell fájl a lemezen.")
            return False

        latest_model_path = max(model_paths, key=os.path.getmtime)
        logger.info(f"-> [LOAD_MODEL] Modell fájl megtalálva: {latest_model_path}")

        model = joblib.load(latest_model_path)
        logger.info(
            "Modell sikeresen betöltve",
            extra={"model_path": latest_model_path},
        )
        return True
    except Exception as e:
        model = None
        logger.warning(
            "Nem sikerült betölteni a modellt",
            extra={"error": str(e)},
        )
        return False


def background_initialization() -> None:
    """Initialize model in a background daemon thread strictly without training on startup."""
    logger.info("-> [BACKGROUND_INIT] Induláskori modell-ellenőrzés (Zéró tréning induláskor)...")
    if load_model():
        logger.info("-> [BACKGROUND_INIT] Modell sikeresen betöltve induláskor.")
    else:
        logger.info("-> [BACKGROUND_INIT] Nincs betöltött modell induláskor. Használd a /train végpontot.")


@app.on_event("startup")
def startup_event() -> None:
    """FastAPI startup event handler starting background initialization."""
    thread = threading.Thread(target=background_initialization, daemon=True)
    thread.start()


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint returning service status and model loaded state."""
    return {
        "status": "healthy",
        "model_loaded": "true" if model is not None else "false",
    }


@app.post("/train")
def train_model_endpoint() -> Dict[str, Any]:
    """Trigger model training via REST API by executing the training script as a subprocess."""
    global model
    logger.info("-> [TRAIN_ENDPOINT] Tréning indítása subprocess-en keresztül...")

    try:
        # Futtatjuk a src.train modulként a tréninget
        result = subprocess.run(
            ["python", "-m", "src.train"],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("Tréning script sikeresen lefutott", extra={"stdout": result.stdout[:200]})

        # Újratöltjük a frissen mentett modellt
        if load_model():
            return {
                "status": "success",
                "message": "Model trained and reloaded successfully via subprocess.",
            }
        else:
            raise RuntimeError("Training script finished, but the model file could not be loaded.")

    except subprocess.CalledProcessError as e:
        logger.error(f"Training script failed with stderr: {e.stderr}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {e.stderr.strip()}",
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}",
        )


def make_prediction(ml_model: Any, data: CustomerData) -> Tuple[int, Optional[float]]:
    """Perform prediction using the loaded model with robust feature name alignment."""
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    if hasattr(ml_model, "feature_names_in_"):
        expected_features = ml_model.feature_names_in_
        aligned_df = pd.DataFrame(columns=expected_features)

        mapping = {
            "Age": "customer_age",
            "Credit_Limit": "credit_limit",
            "Total_Trans_Amt": "total_trans_amt",
            "Total_Trans_Ct": "total_trans_ct",
        }

        for col in expected_features:
            assigned = False
            for model_col, api_col in mapping.items():
                if col.lower() == model_col.lower() and api_col in df.columns:
                    aligned_df.loc[0, col] = df.loc[0, api_col]
                    assigned = True
                    break
            
            if not assigned:
                if col in df.columns:
                    aligned_df.loc[0, col] = df.loc[0, col]
                else:
                    aligned_df.loc[0, col] = 0.0

        features = aligned_df
    else:
        features = df

    prediction = int(ml_model.predict(features)[0])
    probability = None
    if hasattr(ml_model, "predict_proba"):
        probs = ml_model.predict_proba(features)
        probability = float(probs[0][1])
    return prediction, probability


@app.post("/predict")
def predict(data: CustomerData) -> Dict[str, Any]:
    """Predict customer churn based on input features, logging request and prediction."""
    logger.info("Incoming prediction request", extra={"request_data": data.dict()})

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded yet. Please trigger /train first.",
        )
    try:
        prediction, probability = make_prediction(model, data)
        response_data = {
            "prediction": prediction,
            "churn_probability": probability,
            "received_data": data.dict(),
        }
        logger.info("Prediction successful", extra={"response_data": response_data})
        return response_data
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )
