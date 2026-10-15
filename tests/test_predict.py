"""Unit tests for the prediction module (src/predict.py)."""

from __future__ import annotations

import pandas as pd

from src.predict import make_prediction


class DummyModelWithProba:
    """Mock model with predict and predict_proba returning 2D array."""
    def predict(self, X: pd.DataFrame) -> list[int]:
        return [1] * len(X)

    def predict_proba(self, X: pd.DataFrame) -> list[list[float]]:
        return [[0.2, 0.8]] * len(X)


class DummyModel1DProba:
    """Mock model with predict_proba returning 1D array."""
    def predict(self, X: pd.DataFrame) -> list[int]:
        return [0] * len(X)

    def predict_proba(self, X: pd.DataFrame) -> list[float]:
        return [0.7, 0.3]


class DummyModelNoProba:
    """Mock model without predict_proba."""
    def predict(self, X: pd.DataFrame) -> list[int]:
        return [0] * len(X)


def test_make_prediction_with_dict() -> None:
    """Test prediction using a dictionary input."""
    model = DummyModelWithProba()
    input_data = {"feature1": 10, "feature2": 20}
    pred, prob = make_prediction(model, input_data)
    assert pred == 1
    assert prob == 0.8


def test_make_prediction_with_dataframe() -> None:
    """Test prediction using a pandas DataFrame input."""
    model = DummyModelWithProba()
    input_data = pd.DataFrame([{"feature1": 10, "feature2": 20}])
    pred, prob = make_prediction(model, input_data)
    assert pred == 1
    assert prob == 0.8


def test_make_prediction_with_1d_proba() -> None:
    """Test prediction with a model returning 1D probability array."""
    model = DummyModel1DProba()
    input_data = {"feature1": 5}
    pred, prob = make_prediction(model, input_data)
    assert pred == 0
    assert prob == 0.7


def test_make_prediction_without_proba() -> None:
    """Test prediction with a model that lacks predict_proba method."""
    model = DummyModelNoProba()
    input_data = {"feature1": 5}
    pred, prob = make_prediction(model, input_data)
    assert pred == 0
    assert prob is None
