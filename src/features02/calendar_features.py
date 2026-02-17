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
    - Missing holiday/festival flags are defaulted to False.
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

    calendar_df = calendar_df.rename(columns={
        calendar_date_col: date_col
    })

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

    df[holiday_col] = df[holiday_col].fillna(False).astype(int)
    df[festival_col] = df[festival_col].fillna(False).astype(int)

    return df


def merge_promo_features(
    df: pd.DataFrame,
    promo_df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """
    Merge promotion signals into the main dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Main sales dataset (must contain date column).
    promo_df : pd.DataFrame
        Promotion dataset containing promotion flags and discount details.
    config : Dict
        Configuration dictionary containing column mappings:
            - date_col
            - promo_date_col
            - promo_flag_col
            - discount_col

    Returns
    -------
    pd.DataFrame
        DataFrame with promotion columns merged.

    Raises
    ------
    ValueError
        If required columns are missing in either dataframe.

    Notes
    -----
    - Fully config-driven (no hardcoded column names).
    - Performs left join on date column.
    - Missing promotion flags default to False.
    - Missing discount values default to 0.0.
    - No leakage risk (promotion schedule is pre-known).
    """

    df = df.copy()
    promo_df = promo_df.copy()

    sales_schema = config["data_schema"]["sales"]
    promo_schema = config["data_schema"].get("promotions", {})
    if not promo_schema:
        raise ValueError("Missing 'data_schema.promotions' configuration.")

    date_col = sales_schema["date_column"]
    promo_date_col = promo_schema["date_column"]
    promo_flag_col = promo_schema["promo_flag_column"]
    discount_col = promo_schema["discount_column"]


    if date_col not in df.columns:
        raise ValueError(f"{date_col} not found in main dataframe")

    if promo_date_col not in promo_df.columns:
        raise ValueError(f"{promo_date_col} not found in promo dataframe")

    promo_df = promo_df.rename(columns={
        promo_date_col: date_col
    })

    required_cols = [date_col, promo_flag_col, discount_col]

    for col in required_cols:
        if col not in promo_df.columns:
            raise ValueError(f"{col} missing in promo dataframe")

    df = df.merge(
        promo_df[required_cols],
        on=date_col,
        how="left"
    )

    df[promo_flag_col] = df[promo_flag_col].fillna(False).astype(int)
    df[discount_col] = df[discount_col].fillna(0.0)

    return df
