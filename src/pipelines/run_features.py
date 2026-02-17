# src/pipelines/run_features.py

"""
Feature Engineering Pipeline
============================

Orchestrates feature generation steps and saves
final model-ready dataset.
"""

import os
import pandas as pd
from typing import Dict

from utils.config_loader import load_config
from utils.logger import get_logger
from utils.helpers import ensure_directory
from features02.time_features import add_time_features
from features02.lag_features import add_lag_features, add_rolling_features
from features02.calendar_features import merge_calendar_features
from features02.aggregation_features import add_same_weekday_rolling
from features02.demand_bucketing import add_demand_buckets


# ==========================================================
# Main Feature Pipeline
# ==========================================================

def run_features():
    """
    Execute full feature engineering workflow.
    """

    config = load_config()
    logger = get_logger(config)

    logger.info("Starting feature engineering pipeline.")

    # --------------------------------------------------
    # 1. Load Cleaned Dataset
    # --------------------------------------------------
    processed_path = config["paths"]["data"]["processed"]

    cleaned_file = os.path.join(processed_path, "cleaned_sales.parquet")

    if not os.path.exists(cleaned_file):
        raise FileNotFoundError(
            "cleaned_sales.parquet not found. "
            "Run ingestion pipeline first."
        )

    df = pd.read_parquet(cleaned_file)

    date_col = config["data_schema"]["sales"]["date_column"]
    target_col = config["data_schema"]["sales"]["target_column"]

    logger.info(f"Loaded cleaned dataset with shape: {df.shape}")

    # --- Industrial Guard: enforce datetime & sorting ---
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)

    # --------------------------------------------------
    # 2. Time Features
    # --------------------------------------------------
    if config["features"]["time_features"]["enabled"]:
        logger.info("Applying time features.")
        df = add_time_features(df, config)

    # --------------------------------------------------
    # 3. Lag Features
    # --------------------------------------------------
    if config["features"]["lag_features"]["enabled"]:
        lags = config["features"]["lag_features"]["lags"]
        logger.info(f"Applying lag features: {lags}")
        df = add_lag_features(df, target_col, lags, date_col)

    # --------------------------------------------------
    # 4. Rolling Features
    # --------------------------------------------------
    if config["features"]["rolling_features"]["enabled"]:
        windows = config["features"]["rolling_features"]["windows"]
        logger.info(f"Applying rolling features: {windows}")
        df = add_rolling_features(df, target_col, windows, date_col)

    # --------------------------------------------------
    # 5. Calendar Merge
    # --------------------------------------------------
    if config["features"]["calendar_features"]["enabled"]:

        calendar_path = os.path.join(
            config["paths"]["data"]["raw"],
            config["ingestion"]["datasets"]["calendar"]["file"]
        )

        if os.path.exists(calendar_path):

            logger.info("Merging calendar features.")

            calendar_df = pd.read_csv(calendar_path)

            calendar_df[
                config["data_schema"]["calendar"]["date_column"]
            ] = pd.to_datetime(
                calendar_df[
                    config["data_schema"]["calendar"]["date_column"]
                ]
            )

            calendar_config = {
                "date_col": date_col,
                "calendar_date_col": config["data_schema"]["calendar"]["date_column"],
                "holiday_col": config["data_schema"]["calendar"]["holiday_column"],
                "festival_col": config["data_schema"]["calendar"]["festival_column"],
                "festival_name_col": config["data_schema"]["calendar"]["festival_name_column"],
            }

            df = merge_calendar_features(df, calendar_df, calendar_config)

    # --------------------------------------------------
    # 6. Aggregation Features
    # --------------------------------------------------
    if config["features"]["aggregation_features"]["enabled"]:
        window = config["features"]["aggregation_features"]["same_weekday_window"]
        logger.info(f"Applying same weekday aggregation with window: {window}")
        df = add_same_weekday_rolling(df, target_col, date_col, window)

    # --------------------------------------------------
    # 7. Demand Bucketing
    # --------------------------------------------------
    if config["features"].get("demand_bucketing", {}).get("method"):
        logger.info("Applying demand bucketing.")
        df = add_demand_buckets(df, config)

    # --------------------------------------------------
    # 8. Drop Insufficient Lag History
    # --------------------------------------------------
    if config["features"]["lag_features"]["enabled"]:
        max_lag = max(config["features"]["lag_features"]["lags"])
        logger.info(f"Dropping first {max_lag} rows due to lag history.")
        df = df.iloc[max_lag:].reset_index(drop=True)

    # --------------------------------------------------
    # 9. Final Validation (Fail Fast)
    # --------------------------------------------------
    if df.isna().sum().sum() > 0:
        raise ValueError(
            "model_ready.parquet contains missing values. "
            "Check feature generation pipeline."
        )

    logger.info("Feature validation passed. No missing values detected.")

    # --------------------------------------------------
    # 10. Save Model-Ready Dataset
    # --------------------------------------------------
    ensure_directory(processed_path)

    output_file = os.path.join(processed_path, "model_ready.parquet")

    df.to_parquet(output_file, index=False)

    logger.info("Feature engineering completed successfully.")
    logger.info(f"Saved model-ready dataset to: {output_file}")


if __name__ == "__main__":
    run_features()
