# src/utils/data_profiler.py

"""
Data Profiler Utility
=====================

Purpose:
--------
Generic dataset profiling engine for:
- Missing value detection
- Duplicate detection
- Data type identification
- Cardinality analysis
- Candidate primary key detection

Reusable across:
- Regression projects
- Classification projects
- Time series projects
- Multi-table systems

This module DOES NOT modify data.
It only detects and reports.
"""

from typing import Dict, List
import pandas as pd


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detect high-level column data types.

    Returns:
    --------
    Dict[str, str]
        Mapping of column name to detected type:
        - "numeric"
        - "categorical"
        - "boolean"
        - "datetime"
        - "unknown"
    """

    type_map = {}

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            type_map[col] = "datetime"
        elif pd.api.types.is_bool_dtype(df[col]):
            type_map[col] = "boolean"
        elif pd.api.types.is_numeric_dtype(df[col]):
            type_map[col] = "numeric"
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            type_map[col] = "categorical"
        else:
            type_map[col] = "unknown"

    return type_map


def profile_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate missing value summary.

    Returns:
    --------
    DataFrame with:
        - missing_count
        - missing_percentage
    """

    total_rows = len(df)

    missing_count = df.isna().sum()
    missing_percentage = (missing_count / total_rows) * 100

    summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": missing_percentage
    })

    return summary.sort_values(by="missing_percentage", ascending=False)


def detect_exact_duplicates(df: pd.DataFrame) -> int:
    """
    Detect exact duplicate rows.

    Returns:
    --------
    int
        Number of duplicate rows.
    """

    return df.duplicated().sum()


def detect_duplicate_by_keys(df: pd.DataFrame, keys: List[str]) -> int:
    """
    Detect duplicates based on specific key columns.

    Parameters:
    -----------
    keys : list of column names
        Candidate primary key columns.

    Returns:
    --------
    int
        Number of duplicate key occurrences.

    Example:
    --------
    For time series:
        keys = ["date"]

    For transactional data:
        keys = ["order_id"]

    For composite key:
        keys = ["order_id", "product_id"]
    """

    return df.duplicated(subset=keys).sum()


def suggest_candidate_keys(df: pd.DataFrame) -> List[str]:
    """
    Suggest candidate unique key columns.

    Logic:
    ------
    Columns where:
        number of unique values == number of rows

    WARNING:
    --------
    This is only a suggestion.
    Business validation required.
    """

    candidates = []

    total_rows = len(df)

    for col in df.columns:
        if df[col].nunique(dropna=False) == total_rows:
            candidates.append(col)

    return candidates


def profile_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze unique value counts for each column.

    Useful for:
    - Detecting high cardinality categorical columns
    - Detecting potential classification targets
    """

    cardinality = {
        col: df[col].nunique(dropna=True)
        for col in df.columns
    }

    return pd.DataFrame.from_dict(
        cardinality,
        orient="index",
        columns=["unique_values"]
    ).sort_values(by="unique_values", ascending=False)
