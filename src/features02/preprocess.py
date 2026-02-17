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
    if date_col not in df.columns:
        raise ValueError(
            f"Column '{date_col}' not found in dataframe during datetime enforcement."
        )

    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="raise")

    return df

def sort_by_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Sort dataframe by date column.
    """
    if date_col not in df.columns:
        raise ValueError(
            f"Column '{date_col}' not found in dataframe during sorting."
        )

    return df.sort_values(by=date_col).reset_index(drop=True)

def remove_negative_sales(
    df: pd.DataFrame,
    sales_col: str
) -> pd.DataFrame:
    """
    Remove rows where sales is negative.
    """
    if sales_col not in df.columns:
        raise ValueError(
            f"Column '{sales_col}' not found in dataframe during negative sales removal."
        )

    return df[df[sales_col] >= 0].reset_index(drop=True)


def remove_duplicate_dates(
    df: pd.DataFrame,
    date_col: str
) -> pd.DataFrame:
    """
    Remove duplicate dates keeping the first occurrence.
    """
    if date_col not in df.columns:
        raise ValueError(
            f"Column '{date_col}' not found in dataframe during duplicate removal."
        )

    return df.drop_duplicates(subset=[date_col], keep="first").reset_index(drop=True)


def preprocess_base_dataframe(
    df: pd.DataFrame,
    date_col: str,
    sales_col: str,
    required_cols: List[str] | None = None,
) -> pd.DataFrame:
    """
    Master preprocessing function for feature layer.
    """
    
    if required_cols is None:
        raise ValueError(
            "required_cols must be explicitly provided to preprocess_base_dataframe."
        )


    validate_required_columns(df, required_cols)

    df = enforce_datetime(df, date_col)
    df = remove_duplicate_dates(df, date_col)
    df = remove_negative_sales(df, sales_col)
    df = sort_by_date(df, date_col)

    return df
