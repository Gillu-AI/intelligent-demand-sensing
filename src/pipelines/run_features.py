# src/pipelines/run_features.py

"""
Feature Engineering Pipeline
============================

Orchestrates feature generation steps and saves
final model-ready dataset.
"""

import os
from datetime import datetime
import pandas as pd

from utils.config_loader import load_config
from utils.logger import get_logger
from utils.helpers import ensure_directory
from utils.helpers import validate_dataframe_not_empty

from features02.time_features import add_time_features
from features02.lag_features import add_lag_features, add_rolling_features
from features02.calendar_features import merge_calendar_features
from features02.aggregation_features import add_avg_sales_same_weekday
from features02.demand_bucketing import add_demand_bucketing


# ==========================================================
# Main Feature Pipeline
# ==========================================================

def run_features():
    """
    Execute full feature engineering workflow.
    """

    config = load_config("config/config.yaml")
    logger = get_logger(config)

    logger.info("Starting feature engineering pipeline.")

    # --------------------------------------------------
    # Validate Required Config Sections
    # --------------------------------------------------

    required_sections = ["paths", "data_schema", "features"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing '{section}' configuration.")

    processed_path = config["paths"]["data"]["processed"]

    if not os.path.exists(processed_path):
        raise FileNotFoundError("Processed data directory not found.")

    # --------------------------------------------------
    # 1. Load Latest Cleaned Dataset
    # --------------------------------------------------

    cleaned_file = os.path.join(processed_path, "cleaned_sales.parquet")
    if not os.path.exists(cleaned_file):
        raise FileNotFoundError(
            "cleaned_sales.parquet not found. Run ingestion pipeline first."
        )

    df = pd.read_parquet(cleaned_file)

    validate_dataframe_not_empty(df, "cleaned_sales")

    logger.info("Loaded cleaned dataset:cleaned_sales.parquet")
    logger.info(f"Dataset shape: {df.shape}")

    date_col = config["data_schema"]["sales"]["date_column"]

    df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    df = df.sort_values(by=date_col).reset_index(drop=True)

    # --------------------------------------------------
    # 2. Time Features
    # --------------------------------------------------

    if config["features"]["time_features"]["enabled"]:

        time_feature_list = config["features"]["time_features"]

        logger.info("Applying time features.")
        df = add_time_features(
            df=df,
            date_col=date_col,
            features=time_feature_list
        )

    # --------------------------------------------------
    # 3. Lag Features
    # --------------------------------------------------

    if config["features"]["lag_features"]["enabled"]:

        logger.info("Applying lag features.")
        df = add_lag_features(df, config)

    # --------------------------------------------------
    # 4. Rolling Features
    # --------------------------------------------------

    if config["features"]["rolling_features"]["enabled"]:

        logger.info("Applying rolling features.")
        df = add_rolling_features(df, config)

    # --------------------------------------------------
    # 5. Calendar Merge
    # --------------------------------------------------

    if config["features"]["calendar_features"]["enabled"]:

        calendar_dataset_cfg = config["ingestion"]["datasets"].get("calendar")

        if not calendar_dataset_cfg:
            raise ValueError("Calendar dataset not configured under ingestion.datasets.")

        calendar_path = os.path.join(
            config["paths"]["data"]["raw"],
            calendar_dataset_cfg["file"]
        )

        if not os.path.exists(calendar_path):
            raise FileNotFoundError("Configured calendar file not found.")

        calendar_df = pd.read_csv(calendar_path)

        calendar_df[
            config["data_schema"]["calendar"]["date_column"]
        ] = pd.to_datetime(
            calendar_df[
                config["data_schema"]["calendar"]["date_column"]
            ],
            errors="raise"
        )

        logger.info("Merging calendar features.")
        df = merge_calendar_features(df, calendar_df, config)

    # --------------------------------------------------
    # 6. Aggregation Features
    # --------------------------------------------------

    if config["features"]["aggregation_features"]["enabled"]:

        logger.info("Applying same weekday aggregation.")
        df = add_avg_sales_same_weekday(df, config)

    # --------------------------------------------------
    # 7. Demand Bucketing
    # --------------------------------------------------

    if config["features"].get("demand_bucketing", {}).get("method"):

        logger.info("Applying demand bucketing.")
        df = add_demand_bucketing(df, config)

    # --------------------------------------------------
    # 8. Drop Insufficient Lag History
    # --------------------------------------------------

    if config["features"]["lag_features"]["enabled"]:

        lags = config["features"]["lag_features"]["lags"]

        if lags:
            max_lag = max(lags)
            logger.info(f"Dropping first {max_lag} rows due to lag history.")
            df = df.iloc[max_lag:].reset_index(drop=True)

    # --------------------------------------------------
    # 9. Final Validation
    # --------------------------------------------------

    if df.isna().sum().sum() > 0:
        raise ValueError(
            "model_ready dataset contains missing values. "
            "Check feature generation pipeline."
        )

    logger.info("Feature validation passed. No missing values detected.")

    # --------------------------------------------------
    # 10. Save Model-Ready Dataset (Versioned)
    # --------------------------------------------------

    ensure_directory(processed_path)

    output_file = os.path.join(
        processed_path, 
        "model_ready.parquet"
    )

    df.to_parquet(output_file, index=False)

    logger.info("Feature engineering completed successfully.")
    logger.info(f"Saved model-ready dataset to: {output_file}")


if __name__ == "__main__":
    run_features()
