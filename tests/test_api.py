"""Unit tests for the FastAPI application module (src/api.py)."""

from fastapi.testclient import TestClient
import pytest
import mlflow
import pandas as pd
import numpy as np
import joblib

import src.api
from src.api import app


# Globális Dummy Model a pickling / joblib tesztekhez
class GlobalDummyModel:
    """Global dummy model class that can be pickled by joblib."""
    def predict(self, df):
        return np.array([1])
    def predict_proba(self, df):
        return np.array([[0.2, 0.8]])


@pytest.fixture(autouse=True)
def speed_up_api_tests(monkeypatch, tmp_path) -> None:
    """Speed up API tests by mocking background init and using local SQLite for MLflow."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow_api.db")
    monkeypatch.setattr(src.api, "background_initialization", lambda: None)


client = TestClient(app)


def test_health_check_not_loaded(monkeypatch) -> None:
    """Test /health endpoint when the model is not loaded."""
    monkeypatch.setattr(src.api, "model", None)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == "false"


def test_health_check_loaded(monkeypatch) -> None:
    """Test /health endpoint when the model is loaded."""
    monkeypatch.setattr(src.api, "model", "dummy_model")
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == "true"


def test_predict_model_not_loaded(monkeypatch) -> None:
    """Test /predict endpoint returns 503 when model is not loaded."""
    monkeypatch.setattr(src.api, "model", None)
    payload = {
        "customer_age": 30,
        "credit_limit": 5000.0,
        "total_trans_amt": 1000.0,
        "total_trans_ct": 20,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 503
    assert "Model is not loaded yet" in response.json()["detail"]


def test_predict_endpoint_success(monkeypatch) -> None:
    """Test successful prediction from /predict endpoint using real make_prediction."""
    class DummyModelWithFeatures:
        feature_names_in_ = ["customer_age", "credit_limit", "total_trans_amt", "total_trans_ct"]
        def predict(self, df):
            return np.array([1])
        def predict_proba(self, df):
            return np.array([[0.1, 0.9]])

    monkeypatch.setattr(src.api, "model", DummyModelWithFeatures())
    payload = {
        "customer_age": 45,
        "credit_limit": 12000.0,
        "total_trans_amt": 2500.0,
        "total_trans_ct": 40,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["churn_probability"] == 0.9
    assert "received_data" in data


def test_predict_endpoint_success_no_proba(monkeypatch) -> None:
    """Test prediction when model has no predict_proba."""
    class DummyModelNoProba:
        def predict(self, df):
            return np.array([0])

    monkeypatch.setattr(src.api, "model", DummyModelNoProba())
    payload = {
        "customer_age": 45,
        "credit_limit": 12000.0,
        "total_trans_amt": 2500.0,
        "total_trans_ct": 40,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0
    assert data["churn_probability"] is None


def test_predict_endpoint_exception(monkeypatch) -> None:
    """Test /predict handling when prediction raises an exception."""
    monkeypatch.setattr(src.api, "model", "dummy_model")

    def mock_raise(*args, **kwargs):
        raise RuntimeError("Prediction computation failed")

    monkeypatch.setattr(src.api, "make_prediction", mock_raise)
    payload = {
        "customer_age": 45,
        "credit_limit": 12000.0,
        "total_trans_amt": 2500.0,
        "total_trans_ct": 40,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 500
    assert "Prediction failed" in response.json()["detail"]


def test_load_model_real_file(monkeypatch, tmp_path) -> None:
    """Test mlflow_load_model and load_model by creating a real temporary pkl file."""
    model_dir = tmp_path / "mlruns" / "1" / "artifacts" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "model.pkl"
    
    joblib.dump(GlobalDummyModel(), model_file)
    monkeypatch.chdir(tmp_path)
    
    res = src.api.load_model()
    assert res is True
    assert src.api.model is not None


def test_load_model_exception(monkeypatch) -> None:
    """Test load_model when an exception occurs during registry loading."""
    def mock_load_error(uri: str) -> None:
        raise RuntimeError("Load failed")

    monkeypatch.setattr(
        src.api, "mlflow_load_model", mock_load_error
    )
    src.api.model = "initial_value"
    src.api.load_model()
    assert src.api.model is None


def test_train_endpoint_success(monkeypatch) -> None:
    """Test /train endpoint triggers training successfully."""
    monkeypatch.setattr("src.train.train_model", lambda: "run_id_123")
    monkeypatch.setattr(src.api, "load_model", lambda: True)

    response = client.post("/train")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["run_id"] == "run_id_123"


def test_train_endpoint_exception(monkeypatch) -> None:
    """Test /train endpoint handles training exception."""
    def mock_train_error():
        raise RuntimeError("Training crashed")

    monkeypatch.setattr("src.train.train_model", mock_train_error)

    response = client.post("/train")
    assert response.status_code == 500
    assert "Training failed" in response.json()["detail"]


def test_predict_validation() -> None:
    """Test Pydantic validation rules on /predict payload."""
    response = client.post(
        "/predict",
        json={
            "customer_age": 15,
            "credit_limit": 5000.0,
            "total_trans_amt": 1000.0,
            "total_trans_ct": 20,
        },
    )
    assert response.status_code == 422


def test_startup_event_triggers_load(monkeypatch) -> None:
    """Test FastAPI startup event triggers background initialization."""
    triggered = []
    monkeypatch.setattr(
        src.api.threading,
        "Thread",
        lambda target, daemon: type("DummyThread", (object,), {
            "start": lambda self: triggered.append(True)
        })()
    )
    src.api.startup_event()
    assert len(triggered) == 1
    assert triggered[0] is True


def test_background_initialization_branches(monkeypatch) -> None:
    """Test all branches of background_initialization function for full coverage."""
    monkeypatch.undo()

    # Case 1: load_model succeeds immediately
    monkeypatch.setattr(src.api, "load_model", lambda: True)
    src.api.background_initialization()

    # Case 2: load_model fails first, then train_model and subsequent load_model succeed
    calls_2 = [False, True]
    monkeypatch.setattr(src.api, "load_model", lambda: calls_2.pop(0))
    monkeypatch.setattr("src.train.train_model", lambda: "mock_run_999")
    src.api.background_initialization()

    # Case 3: load_model fails first, train succeeds, but subsequent load_model fails
    calls_3 = [False, False]
    monkeypatch.setattr(src.api, "load_model", lambda: calls_3.pop(0))
    monkeypatch.setattr("src.train.train_model", lambda: "mock_run_999")
    src.api.background_initialization()

    # Case 4: load_model fails, train_model raises exception
    monkeypatch.setattr(src.api, "load_model", lambda: False)

    def mock_train_fail():
        raise RuntimeError("Train fail")
    monkeypatch.setattr("src.train.train_model", mock_train_fail)
    src.api.background_initialization()


def test_mlflow_load_model_no_files(monkeypatch) -> None:
    """Test mlflow_load_model raises error when no model files exist."""
    monkeypatch.setattr("glob.glob", lambda *args, **kwargs: [])
    with pytest.raises(RuntimeError, match="Nem található .pkl modell fájl"):
        src.api.mlflow_load_model("runs:/dummy/model")


def test_train_endpoint_fallback_run_id(monkeypatch) -> None:
    """Test /train endpoint fallback when train_model lacks return or attribute."""
    class DummyTrainModule:
        pass
    
    # Helyes objektum-alapú attribútumcsere a train modulra
    import sys
    dummy_mod = DummyTrainModule()
    monkeypatch.setitem(sys.modules, "src.train", dummy_mod)
    monkeypatch.setattr(src.api, "load_model", lambda: True)

    response = client.post("/train")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["run_id"] == "default_run_id"


def test_make_prediction_feature_alignment_mapping(monkeypatch) -> None:
    """Test make_prediction exercises mapping and fallback branches in feature alignment."""
    class ComplexDummyModel:
        feature_names_in_ = ["Age", "Credit_Limit", "Unknown_Extra_Col"]
        def predict(self, df):
            return np.array([1])
        def predict_proba(self, df):
            return np.array([[0.3, 0.7]])

    model_obj = ComplexDummyModel()
    data = src.api.CustomerData(
        customer_age=50, credit_limit=15000.0, total_trans_amt=3000.0, total_trans_ct=30
    )
    pred, prob = src.api.make_prediction(model_obj, data)
    assert pred == 1
    assert prob == 0.7
