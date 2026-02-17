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


# ==========================================================
# Column Type Detection
# ==========================================================

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

    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    type_map = {}

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            type_map[col] = "datetime"
        elif pd.api.types.is_bool_dtype(df[col]):
            type_map[col] = "boolean"
        elif pd.api.types.is_numeric_dtype(df[col]):
            type_map[col] = "numeric"
        elif (
            pd.api.types.is_object_dtype(df[col]) or
            pd.api.types.is_categorical_dtype(df[col])
        ):
            type_map[col] = "categorical"
        else:
            type_map[col] = "unknown"

    return type_map


# ==========================================================
# Missing Value Profiling
# ==========================================================

def profile_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate missing value summary.

    Returns:
    --------
    DataFrame with:
        - missing_count
        - missing_percentage
    """

    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    total_rows = len(df)

    if total_rows == 0:
        return pd.DataFrame(
            columns=["missing_count", "missing_percentage"]
        )

    missing_count = df.isna().sum()
    missing_percentage = (missing_count / total_rows) * 100

    summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": missing_percentage
    })

    return summary.sort_values(
        by="missing_percentage",
        ascending=False
    )


# ==========================================================
# Duplicate Detection
# ==========================================================

def detect_exact_duplicates(df: pd.DataFrame) -> int:
    """
    Detect exact duplicate rows.

    Returns:
    --------
    int
        Number of duplicate rows.
    """

    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    return int(df.duplicated().sum())


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
    """

    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    if not keys:
        raise ValueError("Keys list cannot be empty.")

    missing_keys = [k for k in keys if k not in df.columns]
    if missing_keys:
        raise ValueError(
            f"Duplicate key check failed. "
            f"Columns not found: {missing_keys}"
        )

    return int(df.duplicated(subset=keys).sum())


# ==========================================================
# Candidate Key Suggestion
# ==========================================================

def suggest_candidate_keys(df: pd.DataFrame) -> List[str]:
    """
    Suggest candidate unique key columns.

    Logic:
    ------
    Columns where:
        number of unique values == number of rows

    WARNING:
    --------
    This performs a full uniqueness scan.
    Business validation required.
    """

    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    candidates = []
    total_rows = len(df)

    if total_rows == 0:
        return candidates

    for col in df.columns:
        if df[col].nunique(dropna=False) == total_rows:
            candidates.append(col)

    return candidates


# ==========================================================
# Cardinality Analysis
# ==========================================================

def profile_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze unique value counts for each column.

    Useful for:
    - Detecting high cardinality categorical columns
    - Detecting potential classification targets

    NOTE:
    -----
    Performs full unique value scan.
    """

    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    cardinality = {
        col: df[col].nunique(dropna=True)
        for col in df.columns
    }

    return (
        pd.DataFrame.from_dict(
            cardinality,
            orient="index",
            columns=["unique_values"]
        )
        .sort_values(by="unique_values", ascending=False)
    )
