# src/modeling03/data_preparation.py

"""
Model Data Preparation Module
==============================

Purpose:
--------
Prepare feature matrix (X) and target vector (y)
for machine learning models.

Responsibilities:
-----------------
- Separate features and target
- Remove non-feature columns
- Ensure numeric-only matrix (if required)
- Return X, y safely

This module does NOT:
- Perform missing handling
- Perform duplicate handling
- Perform data splitting
- Perform feature engineering

It only prepares model-ready inputs.
"""

from typing import Tuple, Dict
import pandas as pd


def prepare_model_data(
    df: pd.DataFrame,
    config: Dict,
    drop_columns: Tuple[str, ...] = ()
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X and y from processed feature dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataset.
    config : Dict
        Configuration dictionary.
    drop_columns : Tuple[str]
        Optional columns to exclude from features
        (e.g., date column for linear/tree models).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")

    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary.")

    if (
        "data_schema" not in config
        or "sales" not in config["data_schema"]
        or "target_column" not in config["data_schema"]["sales"]
    ):
        raise ValueError(
            "Missing 'data_schema.sales.target_column' in configuration."
        )

    target_col = config["data_schema"]["sales"]["target_column"]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    y = df[target_col].copy()

    # Drop target + user-specified columns
    feature_df = df.drop(columns=[target_col] + list(drop_columns), errors="ignore")

    # Ensure numeric-only features for sklearn models
    X = feature_df.select_dtypes(include=["number"]).copy()

    if X.empty:
        raise ValueError("No numeric features found for modeling.")

    return X, y
