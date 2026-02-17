# src/modeling03/evaluate.py

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_mape(y_true, y_pred) -> float:
    """
    Compute Mean Absolute Percentage Error (MAPE).

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        MAPE percentage.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    non_zero_mask = y_true != 0

    if not np.any(non_zero_mask):
        raise ValueError(
            "MAPE cannot be computed because all true values are zero."
        )

    y_true = y_true[non_zero_mask]
    y_pred = y_pred[non_zero_mask]

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate_regression_model(y_true, y_pred) -> Dict[str, float]:
    """
    Compute regression evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    Dict[str, float]
        Dictionary containing MAE, RMSE, MAPE, R2.
    """

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = calculate_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "R2": r2
    }


def build_metrics_dataframe(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Convert model evaluation results into a comparison DataFrame.

    Parameters
    ----------
    results : Dict[str, Dict[str, float]]
        Dictionary structured as:
        {
            "model_name": {"MAE": ..., "RMSE": ..., ...},
            ...
        }

    Returns
    -------
    pd.DataFrame
        Tabular comparison of model metrics.
    """

    if not results:
        raise ValueError("Results dictionary is empty.")

    df = pd.DataFrame(results).T

    if "MAPE (%)" not in df.columns:
        raise ValueError(
            "'MAPE (%)' column not found in results. Cannot sort models."
        )

    df = df.sort_values(by="MAPE (%)")

    return df
