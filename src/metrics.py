"""
Core metrics computation module -- Single Source of Truth.
Used by both train.py and evaluate.py to ensure identical
metric calculations.
"""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
import pandas as pd


def compute_metrics(
    model: Pipeline | ClassifierMixin,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Computes comprehensive classification metrics using predict_proba
    and a threshold. Acts as the single source of truth across
    training and evaluation.
    """
    # Ellenőrizzük, hogy a modell támogatja-e a valószínűségi becslést
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        y_pred = (proba >= threshold).astype(int)
        roc_auc = float(roc_auc_score(y, proba))
    else:
        y_pred = model.predict(X)
        roc_auc = 0.0

    return {
        "accuracy": round(
            float(accuracy_score(y, y_pred)), 4
        ),
        "precision": round(
            float(
                precision_score(
                    y, y_pred, zero_division=0
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    y, y_pred, zero_division=0
                )
            ),
            4,
        ),
        "f1": round(
            float(
                f1_score(
                    y, y_pred, zero_division=0
                )
            ),
            4,
        ),
        "roc_auc": round(roc_auc, 4),
    }


def slice_metrics(
    model: Pipeline | ClassifierMixin,
    X: pd.DataFrame,
    y: pd.Series,
    column: str,
    threshold: float = 0.5,
    min_rows: int = 30,
) -> dict[str, float]:
    """
    Computes per-segment F1 scores to detect performance disparities
    across groups. Essential for model cards and promotion gates.
    """
    out: dict[str, float] = {}
    if column not in X.columns:
        return out

    for value, idx in X.groupby(column).groups.items():
        if len(idx) < min_rows:
            continue
        sub_metrics = compute_metrics(
            model, X.loc[idx], y.loc[idx], threshold
        )
        key = (
            f"f1_slice_{column}_{value}"
            .replace(" ", "_")
        )
        out[key] = sub_metrics["f1"]

    return out
