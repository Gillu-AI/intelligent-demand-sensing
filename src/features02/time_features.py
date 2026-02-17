# src/features02/time_features.py

from typing import List
import pandas as pd


REQUIRED_COLUMNS = ["date"]


def validate_datetime_column(df: pd.DataFrame, date_col: str) -> None:
    """
    Validate that the date column exists and is datetime type.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    date_col : str
        Name of the datetime column.

    Raises
    ------
    ValueError
        If column is missing or not datetime.
    """
    if date_col not in df.columns:
        raise ValueError(f"Required date column '{date_col}' not found in DataFrame")

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise ValueError(
            f"Column '{date_col}' must be datetime type. "
            f"Current dtype: {df[date_col].dtype}"
        )


def add_time_features(
    df: pd.DataFrame,
    date_col: str,
    features: List[str] | None = None,
) -> pd.DataFrame:
    """
    Add time-derived features based on a datetime column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (assumed schema-validated).
    date_col : str
        Datetime column to derive features from.
    features : list[str] or None
        Subset of time features to compute. If None, all are computed.

    Returns
    -------
    pd.DataFrame
        DataFrame with new time-based feature columns added.
    """
    validate_datetime_column(df, date_col)

    df = df.copy()

    all_features = {
    "day_of_week": df[date_col].dt.weekday,
    "is_weekend": (df[date_col].dt.weekday >= 5).astype(int),
    "week_of_year": df[date_col].dt.isocalendar().week.astype(int),
    "month": df[date_col].dt.month,
    "quarter": df[date_col].dt.quarter,
    "is_month_start": df[date_col].dt.is_month_start.astype(int),
    "is_month_end": df[date_col].dt.is_month_end.astype(int),
    }


    selected_features = features or list(all_features.keys())

    invalid = set(selected_features) - set(all_features.keys())
    if invalid:
        raise ValueError(f"Invalid time features requested: {sorted(invalid)}")

    for feature_name in selected_features:
        df[feature_name] = all_features[feature_name]

    return df
