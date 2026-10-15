"""Unit tests for the FastAPI REST API module."""

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health_check() -> None:
    """Test the /health endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint() -> None:
    """Test the /predict endpoint with valid and invalid payloads."""
    valid_data = {
        "customer_age": 45,
        "credit_limit": 12500.0,
        "total_trans_amt": 1500.0,
        "total_trans_ct": 42,
    }
    response = client.post("/predict", json=valid_data)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Not Implemented Yet"
    assert data["received_age"] == "45"

    invalid_data = {
        "customer_age": 10,
        "credit_limit": 12500.0,
        "total_trans_amt": 1500.0,
        "total_trans_ct": 42,
    }
    response_invalid = client.post("/predict", json=invalid_data)
    assert response_invalid.status_code == 422
