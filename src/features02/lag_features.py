# src/features02/lag_features.py

"""
Lag and Rolling Feature Engineering Module
==========================================

Purpose
-------
Provides deterministic, leakage-safe time-series feature generation
for lag-based and rolling statistical features within the IDS pipeline.

This module is part of the Mode B sequential artifact chain and operates
strictly on ingestion-cleaned datasets.

Architectural Guarantees
------------------------
- No raw data loading
- No schema mutation
- No target leakage
- No future value access
- No implicit sorting dependency on upstream layers
- Deterministic feature naming
- Config-driven behavior only

Time-Series Safety Enforcement
------------------------------
All feature functions internally:

1. Enforce datetime conversion on the date column
2. Sort the dataset ascending by date
3. Reset index after sorting
4. Use shift() operations to prevent future data access

This guarantees:
    feature(t) uses only information available at time t-1 or earlier.

Leakage Prevention Logic
------------------------
Lag features:
    y_lag_k(t) = y(t - k)

Rolling features:
    rolling_mean_k(t) = mean( y(t-1), y(t-2), ..., y(t-k) )

Note:
Rolling features are computed on shifted data (shift(1))
to avoid incorporating current observation y(t).

Impact of Maximum Lag
---------------------
If maximum lag = L,
then the first L rows will contain NaN for lag features.

These rows must be handled at the pipeline level
(typically dropped after feature generation).

Failure Conditions
------------------
- Date column missing
- Target column missing
- Invalid lag configuration
- Invalid rolling window configuration
- Non-positive integer lag/window values

This module does NOT:
---------------------
- Drop rows
- Fill missing values
- Persist artifacts
- Perform validation beyond feature scope
"""

from typing import Dict
import pandas as pd


def _prepare_time_order(
    df: pd.DataFrame,
    date_col: str
) -> pd.DataFrame:
    """
    Enforce strict chronological ordering.

    Responsibilities
    ----------------
    - Validate presence of date column
    - Cast date column to datetime
    - Sort ascending by date
    - Reset index to ensure deterministic row positions

    This eliminates dependence on upstream ordering and
    guarantees time-series correctness.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    date_col : str
        Name of datetime column

    Returns
    -------
    pd.DataFrame
        Chronologically sorted dataframe

    Raises
    ------
    ValueError
        If date column does not exist
        If datetime parsing fails
    """

    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found for time ordering.")

    df = df.copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="raise")

    df = df.sort_values(by=date_col).reset_index(drop=True)

    return df


def add_lag_features(
    df: pd.DataFrame,
    config: Dict,
    date_col: str
) -> pd.DataFrame:
    """
    Generate lag-based target features.

    Mathematical Definition
    ------------------------
    For each lag k:

        y_lag_k(t) = y(t - k)

    Where:
        y = target column
        k = lag window
        t = current time index

    Leakage Safety
    --------------
    - Uses shift(k)
    - No forward access
    - Data sorted internally
    - Deterministic feature naming

    Config Contract
    ---------------
    config["features"]["lag_features"]["enabled"]
    config["features"]["lag_features"]["lags"]

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset from ingestion
    config : Dict
        Global configuration dictionary
    date_col : str
        Datetime column name

    Returns
    -------
    pd.DataFrame
        DataFrame with additional lag columns

    Raises
    ------
    ValueError
        If lag configuration invalid
        If target column missing
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
    Generate rolling statistical features (mean and standard deviation).

    Mathematical Definition
    ------------------------
    For each window w:

        rolling_mean_w(t) =
            mean( y(t-1), y(t-2), ..., y(t-w) )

        rolling_std_w(t) =
            std( y(t-1), y(t-2), ..., y(t-w) )

    Important:
        shift(1) is applied before rolling to prevent
        using current value y(t) inside rolling window.

    Leakage Safety
    --------------
    - Internal chronological enforcement
    - shift(1) prior to rolling
    - No access to current or future values

    Config Contract
    ---------------
    config["features"]["rolling_features"]["enabled"]
    config["features"]["rolling_features"]["windows"]

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset
    config : Dict
        Global configuration
    date_col : str
        Datetime column name

    Returns
    -------
    pd.DataFrame
        DataFrame with additional rolling feature columns

    Raises
    ------
    ValueError
        If window configuration invalid
        If target column missing
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