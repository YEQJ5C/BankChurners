"""FastAPI REST API module for the BankChurners MLOps pipeline."""

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.config import settings

app = FastAPI(title="BankChurners MLOps API", version="1.0.0")

# Globális modell változó
model = None


def load_model() -> None:
    """Load the trained MLflow model using the stored run_id."""
    global model
    run_id_file = settings.models_dir / "mlflow_run_id"
    if run_id_file.exists():
        try:
            run_id = run_id_file.read_text(encoding="utf-8").strip()
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
        except Exception:
            model = None
    else:
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


@app.get("/health")
def health_check() -> dict[str, str]:
    """Verify the health status of the API.

    Returns:
        dict[str, str]: A dictionary containing the health status.
    """
    return {"status": "healthy"}


@app.post("/predict")
def predict(payload: ChurnPredictionRequest) -> dict[str, str | int | float]:
    """Endpoint for customer churn prediction using the loaded model."""
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model is not loaded or trained yet."
        )

    input_data = pd.DataFrame(
        [{
            "customer_age": payload.customer_age,
            "credit_limit": payload.credit_limit,
            "total_trans_amt": payload.total_trans_amt,
            "total_trans_ct": payload.total_trans_ct,
        }]
    )

    try:
        prediction = int(model.predict(input_data)[0])
        probability = (
            float(model.predict_proba(input_data)[0][1])
            if hasattr(model, "predict_proba")
            else None
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}"
        )

    return {
        "prediction": prediction,
        "churn_probability": probability,
        "received_age": payload.customer_age,
    }
