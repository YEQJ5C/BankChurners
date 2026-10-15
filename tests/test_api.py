"""Unit tests for the FastAPI REST API module."""

from fastapi.testclient import TestClient
from src.api import app
import src.api


def test_health_check() -> None:
    """Test the /health endpoint returns healthy status."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


def test_predict_model_not_loaded() -> None:
    """Test /predict returns 503 when model is not loaded."""
    with TestClient(app) as client:
        src.api.model = None
        valid_data = {
            "customer_age": 45,
            "credit_limit": 12500.0,
            "total_trans_amt": 1500.0,
            "total_trans_ct": 42,
        }
        response = client.post("/predict", json=valid_data)
        assert response.status_code == 503


def test_predict_endpoint_success(monkeypatch) -> None:
    """Test the /predict endpoint with a mocked loaded model."""
    with TestClient(app) as client:
        monkeypatch.setattr(src.api, "model", object())
        valid_data = {
            "customer_age": 45,
            "credit_limit": 12500.0,
            "total_trans_amt": 1500.0,
            "total_trans_ct": 42,
        }
        response = client.post("/predict", json=valid_data)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "Pending feature mapping"
        assert data["received_age"] == 45


def test_load_model_with_existing_file(monkeypatch, tmp_path) -> None:
    """Test load_model when mlflow_run_id file exists."""
    fake_run_file = tmp_path / "mlflow_run_id"
    fake_run_file.write_text("fake_run_id_123", encoding="utf-8")
    monkeypatch.setattr(src.api.settings, "models_dir", tmp_path)

    monkeypatch.setattr(
        src.api.mlflow.sklearn, "load_model", lambda uri: "dummy_model"
    )

    src.api.load_model()
    assert src.api.model == "dummy_model"


def test_load_model_file_not_found(monkeypatch, tmp_path) -> None:
    """Test load_model when mlflow_run_id file does not exist."""
    monkeypatch.setattr(src.api.settings, "models_dir", tmp_path)
    src.api.model = "initial_value"
    src.api.load_model()
    assert src.api.model is None


def test_load_model_exception(monkeypatch, tmp_path) -> None:
    """Test load_model when an exception occurs during loading."""
    fake_run_file = tmp_path / "mlflow_run_id"
    fake_run_file.write_text("fake_run_id_123", encoding="utf-8")
    monkeypatch.setattr(src.api.settings, "models_dir", tmp_path)

    def mock_load_error(uri: str) -> None:
        raise RuntimeError("Load failed")

    monkeypatch.setattr(src.api.mlflow.sklearn, "load_model", mock_load_error)
    src.api.model = "initial_value"
    src.api.load_model()
    assert src.api.model is None


def test_predict_validation() -> None:
    """Test the /predict endpoint validation with incorrect payloads."""
    with TestClient(app) as client:
        invalid_data = {
            "customer_age": 10,
            "credit_limit": 12500.0,
            "total_trans_amt": 1500.0,
            "total_trans_ct": 42,
        }
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422


def test_startup_event_triggers_load(monkeypatch, tmp_path) -> None:
    """Test that application startup triggers load_model."""
    fake_run_file = tmp_path / "mlflow_run_id"
    fake_run_file.write_text("startup_run_id", encoding="utf-8")
    monkeypatch.setattr(src.api.settings, "models_dir", tmp_path)
    monkeypatch.setattr(
        src.api.mlflow.sklearn, "load_model", lambda uri: "startup_model"
    )
    src.api.model = None

    with TestClient(app):
        assert src.api.model == "startup_model"
