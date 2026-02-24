# src/features02/lag_features.py

from typing import Dict
import pandas as pd


def _prepare_time_order(
    df: pd.DataFrame,
    date_col: str
) -> pd.DataFrame:
    """
    Ensure dataframe is properly datetime-typed and sorted.

    This function enforces:
    - date column exists
    - date column is datetime
    - dataframe sorted ascending by date

    This prevents leakage and eliminates reliance on upstream sorting.
    """

    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found for lag validation.")

    # Enforce datetime
    df[date_col] = pd.to_datetime(df[date_col], errors="raise")

    # Sort safely
    df = df.sort_values(by=date_col).reset_index(drop=True)

    return df


def add_lag_features(
    df: pd.DataFrame,
    config: Dict,
    date_col: str
) -> pd.DataFrame:
    """
    Add lag-based features using shift operations.

    Fully leakage-safe:
    - Always sorted by date internally
    - Uses shift(lag)
    - No future access

    Parameters
    ----------
    df : pd.DataFrame
    config : Dict
    date_col : str

    Returns
    -------
    pd.DataFrame
    """

    if not config.get("features", {}).get("lag_features", {}).get("enabled", False):
        return df

    df = df.copy()

    df = _prepare_time_order(df, date_col)

    target_col = config["data_schema"]["sales"]["target_column"]
    lags = config["features"]["lag_features"]["lags"]

    if not isinstance(lags, list) or not all(isinstance(l, int) and l > 0 for l in lags):
        raise ValueError("Lag values must be a list of positive integers.")

    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found for lag features.")

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    config: Dict,
    date_col: str
) -> pd.DataFrame:
    """
    Add rolling mean and rolling standard deviation features.

    Fully leakage-safe:
    - Always sorted internally
    - Uses shift(1) before rolling
    """

    if not config.get("features", {}).get("rolling_features", {}).get("enabled", False):
        return df

    df = df.copy()

    df = _prepare_time_order(df, date_col)

    target_col = config["data_schema"]["sales"]["target_column"]
    windows = config["features"]["rolling_features"]["windows"]

    if not isinstance(windows, list) or not all(isinstance(w, int) and w > 0 for w in windows):
        raise ValueError("Rolling windows must be a list of positive integers.")

    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found for rolling features.")

    shifted_series = df[target_col].shift(1)

    for window in windows:
        df[f"{target_col}_rolling_mean_{window}"] = (
            shifted_series.rolling(window=window).mean()
        )

        df[f"{target_col}_rolling_std_{window}"] = (
            shifted_series.rolling(window=window).std()
        )

    return df
