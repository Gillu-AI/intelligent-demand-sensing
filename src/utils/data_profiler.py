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

from typing import Dict, List, Any
import pandas as pd


# ==========================================================
# Column Type Detection
# ==========================================================

def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detect high-level column data types.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

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
            isinstance(df[col].dtype, pd.CategoricalDtype)
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
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    total_rows = len(df)

    if total_rows == 0:
        return pd.DataFrame(
            columns=["missing_count", "missing_percentage"]
        )

    missing_count = df.isna().sum()
    missing_percentage = ((missing_count / total_rows) * 100).round(4)

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
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    return int(df.duplicated().sum())


def detect_duplicate_by_keys(df: pd.DataFrame, keys: List[str]) -> int:
    """
    Detect duplicates based on specific key columns.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if not isinstance(keys, list) or not keys:
        raise ValueError("Keys must be a non-empty list of column names.")

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
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

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
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

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


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate consolidated profiling summary for a DataFrame.

    Returns
    -------
    Dict[str, Any]
        {
            "row_count": int,
            "column_count": int,
            "column_types": Dict[str, str],
            "missing_summary": pd.DataFrame,
            "duplicate_rows": int,
            "cardinality": pd.DataFrame,
            "candidate_keys": List[str]
        }
    """

    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    summary = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "column_types": detect_column_types(df),
        "missing_summary": profile_missing_values(df),
        "duplicate_rows": detect_exact_duplicates(df),
        "cardinality": profile_cardinality(df),
        "candidate_keys": suggest_candidate_keys(df),
    }

    return summary
