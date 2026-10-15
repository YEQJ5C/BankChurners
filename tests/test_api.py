"""Unit tests for the FastAPI REST API module."""

from fastapi.testclient import TestClient
from src.api import app
import src.api


class DummyModel:
    """Mock model for testing predictions."""

    def predict(self, X):
        return [1]

    def predict_proba(self, X):
        return [[0.2, 0.8]]


class DummyModelNoProba:
    """Mock model without probability for testing None probability branch."""

    def predict(self, X):
        return [1]


class FaultyModel:
    """Mock model that raises an exception during prediction."""

    def predict(self, X):
        raise RuntimeError("Prediction error")


def test_health_check_not_loaded() -> None:
    """Test the /health endpoint when model is not loaded."""
    with TestClient(app) as client:
        src.api.model = None
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "healthy",
            "model_loaded": "false",
        }


def test_health_check_loaded(monkeypatch) -> None:
    """Test the /health endpoint when model is loaded."""
    with TestClient(app) as client:
        monkeypatch.setattr(src.api, "model", DummyModel())
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "healthy",
            "model_loaded": "true",
        }


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
        monkeypatch.setattr(src.api, "model", DummyModel())
        valid_data = {
            "customer_age": 45,
            "credit_limit": 12500.0,
            "total_trans_amt": 1500.0,
            "total_trans_ct": 42,
        }
        response = client.post("/predict", json=valid_data)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 1
        assert data["churn_probability"] == 0.8
        assert data["received_data"]["customer_age"] == 45


def test_predict_endpoint_success_no_proba(monkeypatch) -> None:
    """Test the /predict endpoint when probability is None."""
    with TestClient(app) as client:
        monkeypatch.setattr(src.api, "model", DummyModelNoProba())
        valid_data = {
            "customer_age": 45,
            "credit_limit": 12500.0,
            "total_trans_amt": 1500.0,
            "total_trans_ct": 42,
        }
        response = client.post("/predict", json=valid_data)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 1
        assert data["churn_probability"] is None


def test_predict_endpoint_exception(monkeypatch) -> None:
    """Test the /predict endpoint returns 500 when prediction fails."""
    with TestClient(app) as client:
        monkeypatch.setattr(src.api, "model", FaultyModel())
        valid_data = {
            "customer_age": 45,
            "credit_limit": 12500.0,
            "total_trans_amt": 1500.0,
            "total_trans_ct": 42,
        }
        response = client.post("/predict", json=valid_data)
        assert response.status_code == 500


def test_load_model_success(monkeypatch) -> None:
    """Test load_model when MLflow registry loading succeeds."""
    monkeypatch.setattr(
        src.api.mlflow.sklearn, "load_model", lambda uri: DummyModel()
    )
    src.api.load_model()
    assert isinstance(src.api.model, DummyModel)


def test_load_model_exception(monkeypatch) -> None:
    """Test load_model when an exception occurs during registry loading."""
    def mock_load_error(uri: str) -> None:
        raise RuntimeError("Load failed")

    monkeypatch.setattr(
        src.api.mlflow.sklearn, "load_model", mock_load_error
    )
    src.api.model = "initial_value"
    src.api.load_model()
    assert src.api.model is None


def test_train_endpoint_success(monkeypatch) -> None:
    """Test the /train endpoint successfully triggers training."""
    with TestClient(app) as client:
        monkeypatch.setattr(
            "src.train.train_model", lambda: "mock_run_id_999"
        )
        monkeypatch.setattr(
            src.api.mlflow.sklearn, "load_model", lambda uri: DummyModel()
        )
        response = client.post("/train")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["run_id"] == "mock_run_id_999"


def test_train_endpoint_exception(monkeypatch) -> None:
    """Test the /train endpoint returns 500 when training fails."""
    with TestClient(app) as client:
        def mock_train_error():
            raise RuntimeError("Training error")

        monkeypatch.setattr("src.train.train_model", mock_train_error)
        response = client.post("/train")
        assert response.status_code == 500


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


def test_startup_event_triggers_load(monkeypatch) -> None:
    """Test that application startup triggers load_model."""
    monkeypatch.setattr(
        src.api.mlflow.sklearn, "load_model", lambda uri: DummyModel()
    )
    src.api.model = None

    with TestClient(app):
        assert isinstance(src.api.model, DummyModel)
