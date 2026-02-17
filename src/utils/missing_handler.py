# src/utils/missing_handler.py

"""
Missing Value & Duplicate Handling Engine
=========================================

Purpose:
--------
Config-driven cleaning engine that handles:

1. Duplicate removal (dataset-specific)
2. Missing value treatment (type-aware)
3. Optional per-column override strategies
4. Missing indicator creation

Reusable for:
- Regression
- Classification
- Time Series
- Multi-table systems

IMPORTANT:
----------
This module modifies the DataFrame.
Use after profiling step.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


# ==========================================================
# DUPLICATE HANDLING
# ==========================================================

def handle_duplicates(
    df: pd.DataFrame,
    unique_key: Optional[List[str]] = None,
    strategy: str = "keep_first"
) -> pd.DataFrame:
    """
    Remove duplicates based on specified unique key.

    Parameters
    ----------
    unique_key : List[str]
        Column(s) defining uniqueness.

        Examples:
        ---------
        ["date"]                    # Time series
        ["order_id"]                # Orders table
        ["order_id", "product_id"]  # Composite key

    strategy : str
        Options:
        --------
        "keep_first"  -> Keep first occurrence
        "keep_last"   -> Keep last occurrence
        "error"       -> Raise error if duplicates exist
        "none"        -> Do not remove duplicates

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame.
    """

    if unique_key is None:
        return df

    duplicate_count = df.duplicated(subset=unique_key).sum()

    if duplicate_count == 0:
        return df

    if strategy == "error":
        raise ValueError(f"Duplicate keys detected: {duplicate_count}")

    if strategy == "keep_first":
        return df.drop_duplicates(subset=unique_key, keep="first")

    if strategy == "keep_last":
        return df.drop_duplicates(subset=unique_key, keep="last")

    if strategy == "none":
        return df

    raise ValueError(f"Invalid duplicate strategy: {strategy}")


# ==========================================================
# MISSING VALUE HANDLING
# ==========================================================

def _impute_numeric(series: pd.Series, strategy: str) -> pd.Series:
    """
    Numeric imputation strategies.

    Options:
    --------
    "median"       -> Robust to outliers
                     Good default for skewed distributions.

    "mean"         -> Good if distribution symmetric.

    "interpolate"  -> Best for time series / continuous signals.

    "ffill"        -> Forward fill (time dependent data).

    "bfill"        -> Backward fill.

    "zero"         -> Fill with 0 (ONLY if business logic allows).

    "none"         -> Leave as-is.

    WARNING:
    --------
    Numeric imputation must consider business meaning.
    """

    if strategy == "median":
        return series.fillna(series.median())

    if strategy == "mean":
        return series.fillna(series.mean())

    if strategy == "interpolate":
        return series.interpolate(method="linear")

    if strategy == "ffill":
        return series.fillna(method="ffill")

    if strategy == "bfill":
        return series.fillna(method="bfill")

    if strategy == "zero":
        return series.fillna(0)

    if strategy == "none":
        return series

    raise ValueError(f"Invalid numeric strategy: {strategy}")


def _impute_categorical(series: pd.Series, strategy: str) -> pd.Series:
    """
    Categorical imputation strategies.

    Options:
    --------
    "unknown"  -> Add 'Unknown' category.
                  Safest production option.

    "mode"     -> Fill with most frequent category.
                  Good if few categories.

    "none"     -> Leave as-is.
    """

    if strategy == "unknown":
        return series.fillna("Unknown")

    if strategy == "mode":
        return series.fillna(series.mode().iloc[0])

    if strategy == "none":
        return series

    raise ValueError(f"Invalid categorical strategy: {strategy}")


def _impute_boolean(series: pd.Series, strategy: str) -> pd.Series:
    """
    Boolean imputation strategies.

    Options:
    --------
    "mode"    -> Fill with most frequent boolean.
    "false"   -> Force False.
    "true"    -> Force True.
    "none"    -> Leave as-is.
    """

    if strategy == "mode":
        return series.fillna(series.mode().iloc[0])

    if strategy == "false":
        return series.fillna(False)

    if strategy == "true":
        return series.fillna(True)

    if strategy == "none":
        return series

    raise ValueError(f"Invalid boolean strategy: {strategy}")


def handle_missing_values(
    df: pd.DataFrame,
    config: Dict,
    dataset_name: str
) -> pd.DataFrame:
    """
    Apply config-driven missing value handling.

    Config Example:
    ---------------
    data_cleaning:
      sales:
        missing:
          numeric: "interpolate"
          categorical: "unknown"
          boolean: "mode"
          datetime: "none"

        per_column:
          total_sales: "ffill"
          price: "median"

    Parameters
    ----------
    config : Dict
        Full configuration dictionary.

    dataset_name : str
        Dataset identifier.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """

    df = df.copy()

    cleaning_config = config.get("data_cleaning", {}).get(dataset_name, {})
    missing_config = cleaning_config.get("missing", {})
    per_column_override = cleaning_config.get("per_column", {})

    for col in df.columns:

        if df[col].isna().sum() == 0:
            continue

        # Per-column override takes priority
        if col in per_column_override:
            df[col] = _impute_numeric(df[col], per_column_override[col])
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            strategy = missing_config.get("numeric", "median")
            df[col] = _impute_numeric(df[col], strategy)

        elif pd.api.types.is_bool_dtype(df[col]):
            strategy = missing_config.get("boolean", "mode")
            df[col] = _impute_boolean(df[col], strategy)

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            strategy = missing_config.get("datetime", "none")
            if strategy == "ffill":
                df[col] = df[col].fillna(method="ffill")

        else:
            strategy = missing_config.get("categorical", "unknown")
            df[col] = _impute_categorical(df[col], strategy)

    return df