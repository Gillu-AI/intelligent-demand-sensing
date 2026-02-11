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

    if not config["features"]["lag_features"]["enabled"]:
        return df

    df = df.copy()

    date_col = config["data_schema"]["sales"]["date_column"]
    target_col = config["data_schema"]["sales"]["target_column"]
    lags = config["features"]["lag_features"]["lags"]

    _validate_sorted(df, date_col)

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

    if not config["features"]["rolling_features"]["enabled"]:
        return df

    df = df.copy()

    date_col = config["data_schema"]["sales"]["date_column"]
    target_col = config["data_schema"]["sales"]["target_column"]
    windows = config["features"]["rolling_features"]["windows"]

    _validate_sorted(df, date_col)

    shifted_series = df[target_col].shift(1)

    for window in windows:
        df[f"{target_col}_rolling_mean_{window}"] = (
            shifted_series.rolling(window=window).mean()
        )

        df[f"{target_col}_rolling_std_{window}"] = (
            shifted_series.rolling(window=window).std()
        )

    return df
