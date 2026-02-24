# src/features02/calendar_features.py

from typing import Dict
import pandas as pd


def merge_calendar_features(
    df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """
    Merge holiday and festival signals into the main dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Main sales dataset (must contain date column).
    calendar_df : pd.DataFrame
        Calendar dataset containing holiday/festival information.
    config : Dict
        Configuration dictionary containing column mappings:
            - date_col
            - calendar_date_col
            - holiday_col
            - festival_col
            - festival_name_col

    Returns
    -------
    pd.DataFrame
        DataFrame with holiday and festival columns merged.

    Raises
    ------
    ValueError
        If required columns are missing in either dataframe.

    Notes
    -----
    - Fully config-driven (no hardcoded column names).
    - Performs left join on date column.
    - Enforces datetime type before merge.
    - Missing holiday/festival flags are defaulted to 0.
    - Does not modify original input DataFrames.
    - No data leakage risk (pure calendar merge).
    """

    df = df.copy()
    calendar_df = calendar_df.copy()

    sales_schema = config["data_schema"]["sales"]
    calendar_schema = config["data_schema"]["calendar"]

    date_col = sales_schema["date_column"]
    calendar_date_col = calendar_schema["date_column"]
    holiday_col = calendar_schema["holiday_column"]
    festival_col = calendar_schema["festival_column"]
    festival_name_col = calendar_schema["festival_name_column"]

    if date_col not in df.columns:
        raise ValueError(f"{date_col} not found in main dataframe")

    if calendar_date_col not in calendar_df.columns:
        raise ValueError(f"{calendar_date_col} not found in calendar dataframe")

    # --------------------------------------------------
    # Enforce datetime type for safe merge
    # --------------------------------------------------

    df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    calendar_df[calendar_date_col] = pd.to_datetime(
        calendar_df[calendar_date_col],
        errors="raise"
    )

    # Rename calendar date column to match sales date column
    if calendar_date_col != date_col:
        calendar_df = calendar_df.rename(
            columns={calendar_date_col: date_col}
        )

    required_cols = [
        date_col,
        holiday_col,
        festival_col,
        festival_name_col
    ]

    for col in required_cols:
        if col not in calendar_df.columns:
            raise ValueError(f"{col} missing in calendar dataframe")

    df = df.merge(
        calendar_df[required_cols],
        on=date_col,
        how="left"
    )

    # Fill missing values safely
    df[holiday_col] = df[holiday_col].fillna(0).astype(int)
    df[festival_col] = df[festival_col].fillna(0).astype(int)

    return df
