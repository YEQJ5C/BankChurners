"""Prediction module for generating inference using trained models."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def make_prediction(
    model: Any, input_data: dict | pd.DataFrame
) -> tuple[int, float | None]:
    """Makes a prediction using model and input data.

    Safely transforms input dictionary or DataFrame
    to match model expected features.
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    # Drop target or ID columns if accidentally passed
    for col in ["Customer_Status", "Customer_ID"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Handle categorical columns uniformly
    cat_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown").astype(str)

    # One-hot encoding
    df_encoded = pd.get_dummies(df, drop_first=False)

    # Align columns with model expectations if available
    if hasattr(model, "feature_names_in_"):
        expected_features = (
            model.feature_names_in_  # type: ignore[reportAttributeAccessIssue] - Dynamic attr.
        )
        df_encoded = df_encoded.reindex(
            columns=expected_features, fill_value=0
        )

    # Run prediction
    raw_preds = model.predict(
        df_encoded
    )  # type: ignore[reportAttributeAccessIssue, reportUnknownMemberType] - Dynamic predict.
    prediction = int(raw_preds[0])

    # Calculate probability if supported
    probability = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(
            df_encoded
        )  # type: ignore[reportAttributeAccessIssue, reportUnknownMemberType] - Dynamic proba.
        probs_arr = np.array(probs)
        if probs_arr.ndim == 2 and probs_arr.shape[1] >= 2:
            probability = float(probs_arr[0, 1])
        elif probs_arr.size > 0:
            probability = float(probs_arr.flatten()[0])

    return prediction, probability
