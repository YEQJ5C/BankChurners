"""Tests for the prediction module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.predict import make_prediction


class DummyModel:
    """Mock model for testing predictions."""

    def __init__(self, has_proba: bool = True):
        self.feature_names_in_ = ["col1", "col2", "cat_col"]
        self.has_proba = has_proba

    def predict(self, X: pd.DataFrame) -> list[int]:
        return [1] * len(X)

    def predict_proba(self, X: pd.DataFrame) -> list[list[float]]:
        if self.has_proba:
            return [[0.2, 0.8]] * len(X)
        return []


class DummyModelEdgeCases:
    """Mock model for testing edge cases in predictions."""

    def __init__(self, proba_shape_type: str = "1d"):
        self.feature_names_in_ = ["col1", "col2", "cat_col"]
        self.proba_shape_type = proba_shape_type

    def predict(self, X: pd.DataFrame) -> list[int]:
        return [0] * len(X)

    def predict_proba(self, X: pd.DataFrame):
        if self.proba_shape_type == "1d":
            return np.array([0.7])
        return [[0.3, 0.7]]


def test_make_prediction_with_dict():
    """Test prediction with dictionary input."""
    model = DummyModel()
    input_data = {"col1": 10, "col2": 20, "cat_col": "A"}
    pred, prob = make_prediction(model, input_data)
    assert pred == 1
    assert prob == 0.8


def test_make_prediction_with_dataframe_and_drop_cols():
    """Test prediction with DataFrame input and columns to be dropped."""
    model = DummyModel()
    input_df = pd.DataFrame(
        [
            {
                "col1": 5,
                "col2": 15,
                "cat_col": "B",
                "Customer_ID": "12345",
                "Customer_Status": "Existing Customer",
            }
        ]
    )
    pred, prob = make_prediction(model, input_df)
    assert pred == 1
    assert prob == 0.8


def test_make_prediction_without_proba():
    """Test prediction when model does not support predict_proba."""
    model = DummyModel(has_proba=False)
    input_data = {"col1": 1, "col2": 2, "cat_col": "C"}
    pred, prob = make_prediction(model, input_data)
    assert pred == 1
    assert prob is None


def test_make_prediction_with_missing_categorical():
    """Test prediction handling missing values in categorical columns."""
    model = DummyModel()
    input_data = {"col1": 10, "col2": 20, "cat_col": None}
    pred, prob = make_prediction(model, input_data)
    assert pred == 1
    assert prob == 0.8


def test_make_prediction_proba_1d():
    """Test prediction when predict_proba returns a 1D array."""
    model = DummyModelEdgeCases(proba_shape_type="1d")
    input_data = {"col1": 1, "col2": 2, "cat_col": "A"}
    pred, prob = make_prediction(model, input_data)
    assert pred == 0
    assert prob == 0.7
