"""FastAPI REST API module for the BankChurners MLOps pipeline."""

from typing import Any, Dict, Optional
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import settings
from src.logger import get_logger
from src.predict import make_prediction

logger = get_logger("churn.api")

app = FastAPI(
    title="BankChurners MLOps API",
    description=(
        "Production-grade API for customer churn "
        "prediction and model training."
    ),
    version="1.0.0",
)

# Globális modell változó
model: Optional[Any] = None


def load_model() -> None:
    """Load the trained MLflow model from the Model Registry."""
    global model
    try:
        model_uri = f"models:/{settings.registered_model_name}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(
            "Modell sikeresen betöltve a regisztrációból",
            extra={"model_uri": model_uri},
        )
    except Exception as e:
        logger.warning(
            "Nem sikerült betölteni a modellt a regisztrációból",
            extra={"model_uri": model_uri, "error": str(e)},
        )
        model = None


@app.on_event("startup")
def startup_event() -> None:
    """Initialize application services on startup."""
    load_model()


class ChurnPredictionRequest(BaseModel):
    """Pydantic model for BankChurners prediction input validation."""

    customer_age: int = Field(
        ..., ge=18, le=100, description="Customer age"
    )
    credit_limit: float = Field(
        ..., gt=0, description="Credit limit"
    )
    total_trans_amt: float = Field(
        ..., gt=0, description="Total transaction amount"
    )
    total_trans_ct: int = Field(
        ..., gt=0, description="Total transaction count"
    )


@app.get("/health", summary="Health Check")
def health_check() -> Dict[str, str]:
    """Verify the health status of the API and model availability."""
    return {
        "status": "healthy",
        "model_loaded": "true" if model is not None else "false",
    }


@app.post("/train", summary="Trigger Model Training")
def trigger_training() -> Dict[str, Any]:
    """Trigger the training pipeline programmatically via the REST API."""
    logger.info("Tréning pipeline indítása API-n keresztül...")
    try:
        from src.train import train_model

        run_id = train_model()
        # Újratöltjük a modellt a friss regisztrációból
        load_model()
        logger.info(
            "A tréning sikeresen lefutott",
            extra={"run_id": run_id},
        )
        return {
            "status": "success",
            "message": "Model trained and registered successfully via API.",
            "run_id": run_id,
        }
    except Exception as e:
        logger.error(
            "A tréning sikertelen volt az API-n keresztül",
            extra={"error": str(e)},
        )
        raise HTTPException(
            status_code=500, detail=f"Training failed: {str(e)}"
        )


@app.post("/predict", summary="Predict Customer Churn")
def predict(payload: ChurnPredictionRequest) -> Dict[str, Any]:
    """Endpoint for customer churn prediction using the registered model."""
    if model is None:
        logger.warning(
            "Predikciós kérés érkezett, de a modell nincs betöltve."
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. "
                "Please train or register a model first."
            ),
        )

    try:
        # Pydantic v2 kompatibilis model_dump() használata
        input_dict = payload.model_dump()
        prediction, probability = make_prediction(model, input_dict)
        logger.info(
            "Predikció sikeresen generálva",
            extra={"prediction": int(prediction)},
        )
    except Exception as e:
        logger.error(
            "A predikció sikertelen volt",
            extra={"error": str(e)},
        )
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}"
        )

    return {
        "prediction": int(prediction),
        "churn_probability": (
            float(probability) if probability is not None else None
        ),
        "received_data": input_dict,
    }
