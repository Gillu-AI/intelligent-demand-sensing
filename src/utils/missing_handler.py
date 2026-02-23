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
    """

    if unique_key is None:
        return df

    if not isinstance(unique_key, list):
        raise ValueError("unique_key must be a list of column names.")

    missing_cols = set(unique_key) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Duplicate handling failed. Columns not found: {sorted(missing_cols)}"
        )

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
    """

    if strategy == "unknown":
        return series.fillna("Unknown")

    if strategy == "mode":
        if series.mode().empty:
            return series
        return series.fillna(series.mode().iloc[0])

    if strategy == "none":
        return series

    raise ValueError(f"Invalid categorical strategy: {strategy}")


def _impute_boolean(series: pd.Series, strategy: str) -> pd.Series:
    """
    Boolean imputation strategies.
    """

    if strategy == "mode":
        if series.mode().empty:
            return series
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
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if dataset_name not in config.get("data_cleaning", {}):
        raise ValueError(
            f"Missing data_cleaning configuration for dataset '{dataset_name}'."
        )

    cleaning_config = config["data_cleaning"][dataset_name]

    if not isinstance(cleaning_config, dict):
        raise ValueError(
            f"Invalid data_cleaning configuration for dataset '{dataset_name}'."
        )

    missing_config = cleaning_config.get("missing", {})
    per_column_override = cleaning_config.get("per_column", {})

    if not isinstance(missing_config, dict):
        raise ValueError(
            f"Invalid missing configuration for dataset '{dataset_name}'."
        )

    if not isinstance(per_column_override, dict):
        raise ValueError(
            f"Invalid per_column configuration for dataset '{dataset_name}'."
        )

    df = df.copy()

    for col in df.columns:

        if df[col].isna().sum() == 0:
            continue

        if col in per_column_override:

            strategy = per_column_override[col]

            if not isinstance(strategy, str):
                raise ValueError(
                    f"Invalid strategy for column '{col}'. Must be string."
                )

            if pd.api.types.is_bool_dtype(df[col]):
                df[col] = _impute_boolean(df[col], strategy)

            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = _impute_numeric(df[col], strategy)

            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                if strategy == "ffill":
                    df[col] = df[col].fillna(method="ffill")
                elif strategy == "bfill":
                    df[col] = df[col].fillna(method="bfill")
                elif strategy == "none":
                    pass
                else:
                    raise ValueError(
                        f"Invalid datetime strategy '{strategy}' "
                        f"for column '{col}'."
                    )
            else:
                df[col] = _impute_categorical(df[col], strategy)

            continue

        if pd.api.types.is_bool_dtype(df[col]):
            if "boolean" not in missing_config:
                raise ValueError(
                    f"Missing boolean strategy in config for dataset '{dataset_name}'."
                )
            strategy = missing_config["boolean"]
            df[col] = _impute_boolean(df[col], strategy)

        elif pd.api.types.is_numeric_dtype(df[col]):
            if "numeric" not in missing_config:
                raise ValueError(
                    f"Missing numeric strategy in config for dataset '{dataset_name}'."
                )
            strategy = missing_config["numeric"]
            df[col] = _impute_numeric(df[col], strategy)

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            if "datetime" not in missing_config:
                raise ValueError(
                    f"Missing datetime strategy in config for dataset '{dataset_name}'."
                )
            strategy = missing_config["datetime"]

            if strategy == "ffill":
                df[col] = df[col].fillna(method="ffill")

            elif strategy == "bfill":
                df[col] = df[col].fillna(method="bfill")

            elif strategy == "none":
                pass

            else:
                raise ValueError(
                    f"Invalid datetime strategy '{strategy}' "
                    f"for column '{col}'."
                )

        else:
            if "categorical" not in missing_config:
                raise ValueError(
                    f"Missing categorical strategy in config for dataset '{dataset_name}'."
                )
            strategy = missing_config["categorical"]
            df[col] = _impute_categorical(df[col], strategy)

    return df
