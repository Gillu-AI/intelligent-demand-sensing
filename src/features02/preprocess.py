# src/features02/preprocessing.py

from typing import List
import pandas as pd


def validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    """
    Ensure required columns exist in dataframe.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def enforce_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Ensure date column is datetime type.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    return df


def sort_by_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Sort dataframe by date column.
    """
    return df.sort_values(by=date_col).reset_index(drop=True)


def remove_negative_sales(
    df: pd.DataFrame,
    sales_col: str
) -> pd.DataFrame:
    """
    Remove rows where sales is negative.
    """
    return df[df[sales_col] >= 0].reset_index(drop=True)


def remove_duplicate_dates(
    df: pd.DataFrame,
    date_col: str
) -> pd.DataFrame:
    """
    Remove duplicate dates keeping the first occurrence.
    """
    return df.drop_duplicates(subset=[date_col], keep="first").reset_index(drop=True)


def preprocess_base_dataframe(
    df: pd.DataFrame,
    date_col: str = "date",
    sales_col: str = "total_sales",
    required_cols: List[str] | None = None,
) -> pd.DataFrame:
    """
    Master preprocessing function for feature layer.
    """
    required_cols = required_cols or [date_col, sales_col]

    validate_required_columns(df, required_cols)

    df = enforce_datetime(df, date_col)
    df = remove_duplicate_dates(df, date_col)
    df = remove_negative_sales(df, sales_col)
    df = sort_by_date(df, date_col)

    return df
