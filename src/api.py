"""FastAPI REST API module for the BankChurners MLOps pipeline."""

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="BankChurners MLOps API", version="1.0.0")


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
def predict(payload: ChurnPredictionRequest) -> dict[str, str]:
    """Placeholder endpoint for customer churn prediction."""
    return {
        "prediction": "Not Implemented Yet",
        "received_age": str(payload.customer_age),
    }
