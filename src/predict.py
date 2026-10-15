"""Prediction module for generating inference using trained models."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def make_prediction(
    model: Any, input_data: dict | pd.DataFrame
) -> tuple[int, float | None]:
    """Makes a prediction using the provided model and input data.

    Args:
        model: The trained machine learning model (e.g., loaded from MLflow).
        input_data: Features as a dictionary or pandas DataFrame.

    Returns:
        A tuple containing:
            - prediction (int): The predicted class (0 or 1).
            - probability (float | None): The churn probability if available,
              else None.
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data

    # Predikció futtatása
    prediction = int(model.predict(df)[0])

    # Valószínűség meghatározása, ha a modell támogatja
    probability = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)
        probs_arr = np.array(probs)
        if probs_arr.ndim == 2 and probs_arr.shape[1] >= 2:
            probability = float(probs_arr[0, 1])
        elif probs_arr.size > 0:
            probability = float(probs_arr.flatten()[0])

    return prediction, probability
