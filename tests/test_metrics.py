"""
Unit tests for the metrics computation module.
"""

from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.metrics import compute_metrics, slice_metrics


class DummyModelWithoutProba(BaseEstimator, ClassifierMixin):
    """Dummy classifier that does not implement predict_proba."""

    def fit(self, X, y):
        """Fit dummy model."""
        return self

    def predict(self, X):
        """Return zeros for prediction."""
        return [0] * len(X)


def test_compute_metrics_with_proba() -> None:
    """Test compute_metrics using a model with predict_proba."""
    X = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0],
        "feature_2": [10, 20, 30, 40],
    })
    y = pd.Series([0, 0, 1, 1])

    model = RandomForestClassifier(random_state=42, n_estimators=5)
    model.fit(X, y)

    metrics = compute_metrics(model, X, y, threshold=0.5)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_compute_metrics_without_proba() -> None:
    """Test compute_metrics using a model without predict_proba."""
    X = pd.DataFrame({"feature_1": [1.0, 2.0]})
    y = pd.Series([0, 1])

    model = DummyModelWithoutProba()
    metrics = compute_metrics(model, X, y)

    assert metrics["roc_auc"] == 0.0
    assert "accuracy" in metrics


def test_slice_metrics_success() -> None:
    """Test slice_metrics when groups meet the min_rows condition."""
    # Csak numerikus oszlopokat adunk X-nek, hogy a RandomForest ne akasszon ki
    X = pd.DataFrame({
        "group": [1] * 35 + [2] * 35,
        "feature": range(70),
    })
    y = pd.Series([0, 1] * 35)

    model = RandomForestClassifier(random_state=42, n_estimators=5)
    model.fit(X, y)

    res = slice_metrics(
        model, X, y, column="group", threshold=0.5, min_rows=30
    )

    assert "f1_slice_group_1" in res
    assert "f1_slice_group_2" in res


def test_slice_metrics_skipped_due_to_min_rows() -> None:
    """Test slice_metrics when group size is below min_rows."""
    X = pd.DataFrame({
        "group": [1] * 5 + [2] * 35,
        "feature": range(40),
    })
    y = pd.Series([0, 1] * 20)

    model = RandomForestClassifier(random_state=42, n_estimators=5)
    model.fit(X, y)

    res = slice_metrics(
        model, X, y, column="group", threshold=0.5, min_rows=30
    )

    assert "f1_slice_group_1" not in res
    assert "f1_slice_group_2" in res


def test_slice_metrics_missing_column() -> None:
    """Test slice_metrics returns empty dict if column is not in X."""
    X = pd.DataFrame({"feature": [1, 2, 3]})
    y = pd.Series([0, 1, 0])

    model = RandomForestClassifier(random_state=42, n_estimators=5)
    model.fit(X, y)

    res = slice_metrics(model, X, y, column="non_existent")
    assert res == {}
