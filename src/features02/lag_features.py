# src/features02/lag_features.py

from typing import Dict
import pandas as pd


def _validate_sorted(df: pd.DataFrame, date_col: str) -> None:
    """
    Validate that the DataFrame is sorted in ascending order by the date column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    date_col : str
        Name of the date column.

    Raises
    ------
    ValueError
        If the DataFrame is not sorted by date.
    
    Purpose
    -------
    Ensures that lag and rolling computations operate strictly
    on past data. Prevents silent leakage caused by unsorted input.
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found for lag validation.")

    if not df[date_col].is_monotonic_increasing:
        raise ValueError(
            "DataFrame must be sorted by date before applying lag features."
        )


def add_lag_features(
    df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """
    Add lag-based features to the dataset using shift operations.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed and time-sorted dataset.
    config : Dict
        Full configuration dictionary. Uses:
        - data_schema.sales.date_column
        - data_schema.sales.target_column
        - features.lag_features.lags

    Returns
    -------
    pd.DataFrame
        DataFrame with lag feature columns added.

    Raises
    ------
    ValueError
        If data is not sorted by date.

    Notes
    -----
    - Each lag feature uses pandas `.shift(lag)`.
    - No future data is accessed.
    - Fully leakage-safe.
    - Respects config-driven lag values.
    """

    if not config.get("features", {}).get("lag_features", {}).get("enabled", False):
        return df

    if "data_schema" not in config or "sales" not in config["data_schema"]:
        raise ValueError("Missing 'data_schema.sales' configuration.")

    df = df.copy()

    date_col = config["data_schema"]["sales"]["date_column"]
    target_col = config["data_schema"]["sales"]["target_column"]
    lags = config["features"]["lag_features"]["lags"]

    if not isinstance(lags, list) or not all(isinstance(l, int) and l > 0 for l in lags):
        raise ValueError("Lag values must be a list of positive integers.")

    _validate_sorted(df, date_col)

    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found for lag features.")

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """
    Add rolling mean and rolling standard deviation features.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed and time-sorted dataset.
    config : Dict
        Full configuration dictionary. Uses:
        - data_schema.sales.date_column
        - data_schema.sales.target_column
        - features.rolling_features.windows

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling feature columns added.

    Raises
    ------
    ValueError
        If data is not sorted by date.

    Notes
    -----
    - Rolling features are computed using shift(1) before rolling.
    - This ensures current-day sales are NOT included in rolling window.
    - Fully leakage-safe.
    - Respects config-driven window sizes.
    """

    if not config.get("features", {}).get("rolling_features", {}).get("enabled", False):
        return df

    if "data_schema" not in config or "sales" not in config["data_schema"]:
        raise ValueError("Missing 'data_schema.sales' configuration.")

    df = df.copy()

    date_col = config["data_schema"]["sales"]["date_column"]
    target_col = config["data_schema"]["sales"]["target_column"]
    windows = config["features"]["rolling_features"]["windows"]

    if not isinstance(windows, list) or not all(isinstance(w, int) and w > 0 for w in windows):
        raise ValueError("Rolling windows must be a list of positive integers.")

    _validate_sorted(df, date_col)

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
