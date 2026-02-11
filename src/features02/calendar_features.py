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

    date_col = config["date_col"]
    calendar_date_col = config["calendar_date_col"]
    holiday_col = config["holiday_col"]
    festival_col = config["festival_col"]
    festival_name_col = config["festival_name_col"]

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

    df[holiday_col] = df[holiday_col].fillna(False)
    df[festival_col] = df[festival_col].fillna(False)

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

    date_col = config["date_col"]
    promo_date_col = config["promo_date_col"]
    promo_flag_col = config["promo_flag_col"]
    discount_col = config["discount_col"]

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

    df[promo_flag_col] = df[promo_flag_col].fillna(False)
    df[discount_col] = df[discount_col].fillna(0.0)

    return df
