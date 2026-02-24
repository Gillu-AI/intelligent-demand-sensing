# src/utils/missing_handler.py

"""
Missing Value & Duplicate Handling Engine
=========================================

Enterprise-Grade Cleaning Engine

Responsibilities
----------------
1. Duplicate removal (config-driven)
2. Schema-aware numeric casting
3. Critical column enforcement
4. Type-aware missing value handling

Design Principles
-----------------
- Fully config-driven
- Deterministic execution order
- Strict schema protection
- No logging inside utility layer
- Safe DataFrame mutation (no chained assignment)
"""

from typing import Dict, List, Optional
import pandas as pd


# ==========================================================
# DUPLICATE HANDLING
# ==========================================================

def handle_duplicates(
    df: pd.DataFrame,
    unique_key: Optional[List[str]] = None,
    strategy: str = "keep_first"
) -> pd.DataFrame:

    if unique_key is None:
        return df.copy()

    if not isinstance(unique_key, list):
        raise ValueError("unique_key must be a list of column names.")

    missing_cols = set(unique_key) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Duplicate handling failed. Columns not found: {sorted(missing_cols)}"
        )

    if strategy == "error":
        if df.duplicated(subset=unique_key).any():
            raise ValueError("Duplicate keys detected.")

    if strategy == "keep_first":
        return df.drop_duplicates(subset=unique_key, keep="first").copy()

    if strategy == "keep_last":
        return df.drop_duplicates(subset=unique_key, keep="last").copy()

    if strategy == "none":
        return df.copy()

    raise ValueError(f"Invalid duplicate strategy: {strategy}")


# ==========================================================
# SCHEMA NUMERIC CASTING
# ==========================================================

def enforce_schema_numeric_cast(
    df: pd.DataFrame,
    config: Dict,
    dataset_name: str
) -> pd.DataFrame:
    """
    Cast only schema-required numeric columns to numeric dtype.
    Date columns are explicitly excluded.
    """

    df = df.copy()

    schema_cfg = config.get("data_schema", {}).get(dataset_name, {})
    required_cols = schema_cfg.get("required_columns", [])
    date_col = schema_cfg.get("date_column")

    for col in required_cols:

        if col not in df.columns:
            continue

        if col == date_col:
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ==========================================================
# CRITICAL COLUMN ENFORCEMENT
# ==========================================================

def enforce_critical_columns(
    df: pd.DataFrame,
    config: Dict,
    dataset_name: str
) -> pd.DataFrame:

    df = df.copy()

    cleaning_cfg = config["data_cleaning"][dataset_name]
    critical_cols = cleaning_cfg.get("critical_columns", [])

    if not isinstance(critical_cols, list):
        raise ValueError(
            f"'critical_columns' must be list in dataset '{dataset_name}'."
        )

    missing_cols = set(critical_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Critical columns not found: {sorted(missing_cols)}"
        )

    if critical_cols:
        df = df.dropna(subset=critical_cols).copy()

    return df


# ==========================================================
# IMPUTATION HELPERS
# ==========================================================

def _impute_numeric(series: pd.Series, strategy: str) -> pd.Series:

    if strategy == "median":
        return series.fillna(series.median())

    if strategy == "mean":
        return series.fillna(series.mean())

    if strategy == "interpolate":
        return series.interpolate(method="linear")

    if strategy == "ffill":
        return series.ffill()

    if strategy == "bfill":
        return series.bfill()

    if strategy == "zero":
        return series.fillna(0)

    if strategy == "none":
        return series

    raise ValueError(f"Invalid numeric strategy: {strategy}")


def _impute_categorical(series: pd.Series, strategy: str) -> pd.Series:

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


# ==========================================================
# MISSING VALUE HANDLING
# ==========================================================

def handle_missing_values(
    df: pd.DataFrame,
    config: Dict,
    dataset_name: str
) -> pd.DataFrame:

    df = df.copy()

    cleaning_cfg = config["data_cleaning"][dataset_name]
    missing_cfg = cleaning_cfg.get("missing", {})
    per_column_override = cleaning_cfg.get("per_column", {})

    schema_cfg = config.get("data_schema", {}).get(dataset_name, {})
    required_cols = schema_cfg.get("required_columns", [])
    date_col = schema_cfg.get("date_column")

    for col in df.columns:

        if df[col].isna().sum() == 0:
            continue

        # Per-column override takes priority
        if col in per_column_override:
            strategy = per_column_override[col]

        else:
            # Schema-based enforcement
            if col in required_cols and col != date_col:
                strategy = missing_cfg.get("numeric", "none")

            elif col == date_col:
                strategy = missing_cfg.get("datetime", "none")

            elif pd.api.types.is_bool_dtype(df[col]):
                strategy = missing_cfg.get("boolean", "none")

            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                strategy = missing_cfg.get("datetime", "none")

            else:
                strategy = missing_cfg.get("categorical", "none")

        # --- Apply strategy based on schema priority ---

        if col in required_cols and col != date_col:
            df.loc[:, col] = _impute_numeric(
                pd.to_numeric(df[col], errors="coerce"),
                strategy
            )

        elif col == date_col:
            if strategy == "ffill":
                df.loc[:, col] = df[col].ffill()
            elif strategy == "bfill":
                df.loc[:, col] = df[col].bfill()
            elif strategy == "none":
                pass
            else:
                raise ValueError(
                    f"Invalid datetime strategy '{strategy}' for '{col}'."
                )

        elif pd.api.types.is_bool_dtype(df[col]):
            df.loc[:, col] = _impute_boolean(df[col], strategy)

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            if strategy == "ffill":
                df.loc[:, col] = df[col].ffill()
            elif strategy == "bfill":
                df.loc[:, col] = df[col].bfill()
            elif strategy == "none":
                pass
            else:
                raise ValueError(
                    f"Invalid datetime strategy '{strategy}' for '{col}'."
                )

        else:
            df.loc[:, col] = _impute_categorical(df[col], strategy)

    return df


# ==========================================================
# MAIN CLEAN FUNCTION
# ==========================================================

def clean_dataframe(
    df: pd.DataFrame,
    config: Dict,
    dataset_name: str
) -> pd.DataFrame:

    if dataset_name not in config.get("data_cleaning", {}):
        raise ValueError(
            f"No cleaning configuration for dataset '{dataset_name}'."
        )

    df = df.copy()

    cleaning_cfg = config["data_cleaning"][dataset_name]
    duplicate_cfg = cleaning_cfg.get("duplicate", {})

    df = handle_duplicates(
        df=df,
        unique_key=duplicate_cfg.get("unique_key"),
        strategy=duplicate_cfg.get("strategy", "keep_first")
    )

    df = enforce_schema_numeric_cast(df, config, dataset_name)

    df = enforce_critical_columns(df, config, dataset_name)

    df = handle_missing_values(df, config, dataset_name)

    return df