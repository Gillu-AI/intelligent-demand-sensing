# src/pipelines/run_features.py

"""
Feature Engineering Pipeline
============================

Enterprise-grade feature orchestration for IDS (Mode B execution).

Purpose
-------
This module consumes ingestion-cleaned artifacts and produces a
deterministic model-ready dataset through config-driven feature
engineering while preserving strict time-series integrity.

Responsibilities
----------------
- Consume cleaned_sales.parquet only
- Optionally merge cleaned_calendar.parquet
- Apply time, lag, rolling, aggregation, and bucketing features
- Drop only structural lag-window rows
- Apply domain-aware missing value handling
- Prevent unintended numeric corruption
- Log structural and column diagnostics
- Persist deterministic model_ready.parquet

Strict Guarantees
-----------------
- No raw data loading
- No column drops
- No additional row drops beyond structural lag window
- No blanket numeric imputation
- No lag or rolling feature imputation
- Fail-fast on unexpected numeric NaN
"""

import os
import pandas as pd
from tabulate import tabulate

from utils.config_loader import load_config
from utils.logger import get_logger
from utils.helpers import ensure_directory, validate_dataframe_not_empty

from features02.time_features import add_time_features
from features02.lag_features import add_lag_features, add_rolling_features
from features02.calendar_features import merge_calendar_features
from features02.aggregation_features import add_avg_sales_same_weekday
from features02.demand_bucketing import add_demand_bucketing


# ==========================================================
# Diagnostics Utilities
# ==========================================================

def _log_shape_change(logger, df_before, df_after, step_name):
    """
    Log structural dataset changes between steps.
    """

    rb, cb = df_before.shape
    ra, ca = df_after.shape

    row_delta = ra - rb
    col_delta = ca - cb
    pct_change = ((ra - rb) / rb * 100) if rb > 0 else 0

    logger.info(
        f"\n[STEP: {step_name}]"
        f"\nRows   : {rb} → {ra}"
        f"\nRow Δ  : {row_delta}"
        f"\nRow %  : {pct_change:.2f}%"
        f"\nCols   : {cb} → {ca}"
        f"\nCol Δ  : {col_delta}"
        "\n--------------------------------------------------"
    )

    if ra == 0:
        raise RuntimeError(f"Dataset collapsed after step: {step_name}")

    if ca == 0:
        raise RuntimeError(f"All columns removed after step: {step_name}")


def _log_column_diagnostics(logger, df, stage):
    """
    Log column-level diagnostics including missing counts and uniqueness.
    """

    total_rows = len(df)
    duplicate_rows = df.duplicated().sum()

    summary = []

    for col in df.columns:
        missing = df[col].isna().sum()
        missing_pct = (missing / total_rows * 100) if total_rows > 0 else 0
        unique = df[col].nunique(dropna=False)
        constant = unique <= 1

        summary.append([
            col,
            str(df[col].dtype),
            missing,
            f"{missing_pct:.2f}%",
            unique,
            constant
        ])

    table = tabulate(
        summary,
        headers=["Column", "DType", "Missing", "Missing %", "Unique", "Constant"],
        tablefmt="grid"
    )

    logger.info(f"\n========== COLUMN DIAGNOSTICS: {stage} ==========")
    logger.info(f"\nTotal Rows     : {total_rows}")
    logger.info(f"Duplicate Rows : {duplicate_rows}")
    logger.info("\n" + table)
    logger.info("==================================================\n")


# ==========================================================
# Main Pipeline
# ==========================================================

def run_features():
    """
    Execute full feature engineering pipeline under Mode B.
    """

    config = load_config("config/config.yaml")
    logger = get_logger(config)

    logger.info("Starting feature engineering pipeline (Mode B).")

    processed_path = config["paths"]["data"]["processed"]
    sales_file = os.path.join(processed_path, "cleaned_sales.parquet")

    if not os.path.exists(sales_file):
        raise FileNotFoundError("cleaned_sales.parquet not found.")

    df = pd.read_parquet(sales_file)
    validate_dataframe_not_empty(df, "cleaned_sales")

    _log_column_diagnostics(logger, df, "initial_cleaned_sales")

    date_col = config["data_schema"]["sales"]["date_column"]
    df[date_col] = pd.to_datetime(df[date_col], errors="raise")

    # --------------------------------------------------
    # Calendar Load
    # --------------------------------------------------

    calendar_df = None

    if config["features"]["calendar_features"]["enabled"]:
        calendar_file = os.path.join(
            processed_path,
            "cleaned_calendar.parquet"
        )

        if not os.path.exists(calendar_file):
            raise FileNotFoundError("cleaned_calendar.parquet missing.")

        calendar_df = pd.read_parquet(calendar_file)
        validate_dataframe_not_empty(calendar_df, "cleaned_calendar")

    # --------------------------------------------------
    # Feature Engineering Steps
    # --------------------------------------------------

    if config["features"]["time_features"]["enabled"]:
        before = df.copy()
        df = add_time_features(df, date_col)
        _log_shape_change(logger, before, df, "time_features")

    if config["features"]["lag_features"]["enabled"]:
        before = df.copy()
        df = add_lag_features(df, config, date_col)
        _log_shape_change(logger, before, df, "lag_features")

    if config["features"]["rolling_features"]["enabled"]:
        before = df.copy()
        df = add_rolling_features(df, config, date_col)
        _log_shape_change(logger, before, df, "rolling_features")

    if calendar_df is not None:
        before = df.copy()
        df = merge_calendar_features(df, calendar_df, config)
        _log_shape_change(logger, before, df, "calendar_merge")

    if config["features"]["aggregation_features"]["enabled"]:
        before = df.copy()
        df = add_avg_sales_same_weekday(df, config)
        _log_shape_change(logger, before, df, "aggregation_features")

    if config["features"].get("demand_bucketing", {}).get("method"):
        before = df.copy()
        df = add_demand_bucketing(df, config)
        _log_shape_change(logger, before, df, "demand_bucketing")

    # --------------------------------------------------
    # Drop Structural Lag Rows Only
    # --------------------------------------------------

    if config["features"]["lag_features"]["enabled"]:
        lags = config["features"]["lag_features"]["lags"]
        if lags:
            max_lag = max(lags)
            before = df.copy()
            df = df.iloc[max_lag:].reset_index(drop=True)
            _log_shape_change(logger, before, df, "drop_max_lag_rows")

    # --------------------------------------------------
    # Domain-Aware Missing Handling
    # --------------------------------------------------

    before = df.copy()
    target_col = config["data_schema"]["sales"]["target_column"]

    df = df.dropna(subset=[target_col])

    if "is_festival" in df.columns:
        df["is_festival"] = df["is_festival"].fillna(0)

    if "festival_name" in df.columns:
        if isinstance(df["festival_name"].dtype, pd.CategoricalDtype):
            if "NoFestival" not in df["festival_name"].cat.categories:
                df["festival_name"] = df["festival_name"].cat.add_categories(["NoFestival"])
        df["festival_name"] = df["festival_name"].fillna("NoFestival")

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:
        if col == "festival_name":
            continue

        if isinstance(df[col].dtype, pd.CategoricalDtype):
            if "Unknown" not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(["Unknown"])

        df[col] = df[col].fillna("Unknown")

    numeric_missing = df.select_dtypes(include=["number"]).isna().sum().sum()

    if numeric_missing > 0:
        raise ValueError(
            "Numeric missing values detected after feature engineering. "
            "Lag or rolling features contain NaN beyond structural window."
        )

    df = df.reset_index(drop=True)

    _log_shape_change(logger, before, df, "controlled_na_handling")
    _log_column_diagnostics(logger, df, "final_model_ready")

    # --------------------------------------------------
    # Persist Deterministic Output
    # --------------------------------------------------

    ensure_directory(processed_path)

    output_file = os.path.join(
        processed_path,
        "model_ready.parquet"
    )

    df = df.sort_values(by=date_col).reset_index(drop=True)

    df.to_parquet(output_file, index=False)

    logger.info(f"Final model_ready shape: {df.shape}")
    logger.info(f"Saved model-ready dataset to: {output_file}")
    logger.info("Feature engineering completed successfully.")


if __name__ == "__main__":
    run_features()