# src/pipelines/run_features.py

"""
Feature Engineering Pipeline
============================

Orchestrates feature generation steps and saves
final model-ready dataset.

DESIGN PRINCIPLES
-----------------
- Fully config-driven execution
- No hardcoded column names
- Consumes processed outputs from ingestion
- No raw reloading
- Fail-fast validation
- Deterministic output
- Leakage-safe transformations
"""

import os
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

    Execution Flow
    --------------
    1. Load configuration
    2. Load cleaned datasets from ingestion layer
    3. Apply time features
    4. Apply lag features
    5. Apply rolling features
    6. Merge calendar signals
    7. Apply aggregation features
    8. Apply demand bucketing
    9. Drop initial lag rows
    10. Validate final dataset
    11. Save model-ready dataset
    """

    # --------------------------------------------------
    # 1. Load Configuration
    # --------------------------------------------------

    config = load_config("config/config.yaml")
    logger = get_logger(config)

    logger.info("Starting feature engineering pipeline.")

    required_sections = ["paths", "data_schema", "features"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing '{section}' configuration.")

    processed_path = config["paths"]["data"]["processed"]

    if not os.path.exists(processed_path):
        raise FileNotFoundError("Processed data directory not found.")

    # --------------------------------------------------
    # 2. Load Cleaned Sales Dataset
    # --------------------------------------------------

    sales_file = os.path.join(processed_path, "cleaned_sales.parquet")

    if not os.path.exists(sales_file):
        raise FileNotFoundError(
            "cleaned_sales.parquet not found. Run ingestion pipeline first."
        )

    df = pd.read_parquet(sales_file)

    validate_dataframe_not_empty(df, "cleaned_sales")

    logger.info("Loaded cleaned dataset: cleaned_sales.parquet")
    logger.info(f"Dataset shape: {df.shape}")

    date_col = config["data_schema"]["sales"]["date_column"]
    df[date_col] = pd.to_datetime(df[date_col], errors="raise")

    # --------------------------------------------------
    # 3. Load Cleaned Calendar Dataset
    # --------------------------------------------------

    if config["features"]["calendar_features"]["enabled"]:

        calendar_file = os.path.join(
            processed_path,
            "cleaned_calendar.parquet"
        )

        if not os.path.exists(calendar_file):
            raise FileNotFoundError(
                "cleaned_calendar.parquet not found. "
                "Run ingestion pipeline first."
            )

        calendar_df = pd.read_parquet(calendar_file)

        validate_dataframe_not_empty(calendar_df, "cleaned_calendar")

        logger.info("Loaded cleaned calendar dataset.")

    else:
        calendar_df = None

    # --------------------------------------------------
    # 4. Time Features
    # --------------------------------------------------

    if config["features"]["time_features"]["enabled"]:

        logger.info("Applying time features.")

        df = add_time_features(
            df=df,
            date_col=date_col,
            features=None
        )

    # --------------------------------------------------
    # 5. Lag Features
    # --------------------------------------------------

    if config["features"]["lag_features"]["enabled"]:

        logger.info("Applying lag features.")

        df = add_lag_features(
            df=df,
            config=config,
            date_col=date_col
        )

    # --------------------------------------------------
    # 6. Rolling Features
    # --------------------------------------------------

    if config["features"]["rolling_features"]["enabled"]:

        logger.info("Applying rolling features.")

        df = add_rolling_features(
            df=df,
            config=config,
            date_col=date_col
        )

    # --------------------------------------------------
    # 7. Calendar Merge
    # --------------------------------------------------

    if calendar_df is not None:

        logger.info("Merging calendar features.")

        df = merge_calendar_features(
            df=df,
            calendar_df=calendar_df,
            config=config
        )

    # --------------------------------------------------
    # 8. Aggregation Features
    # --------------------------------------------------

    if config["features"]["aggregation_features"]["enabled"]:

        logger.info("Applying same weekday aggregation.")

        df = add_avg_sales_same_weekday(
            df=df,
            config=config
        )

    # --------------------------------------------------
    # 9. Demand Bucketing
    # --------------------------------------------------

    if config["features"].get("demand_bucketing", {}).get("method"):

        logger.info("Applying demand bucketing.")

        df = add_demand_bucketing(
            df=df,
            config=config
        )

    # --------------------------------------------------
    # 10. Drop Insufficient Lag History
    # --------------------------------------------------

    if config["features"]["lag_features"]["enabled"]:

        lags = config["features"]["lag_features"]["lags"]

        if lags:
            max_lag = max(lags)
            logger.info(f"Dropping first {max_lag} rows due to lag history.")
            df = df.iloc[max_lag:].reset_index(drop=True)

    # --------------------------------------------------
    # 11. Final Validation
    # --------------------------------------------------

    df = df.dropna().reset_index(drop=True)

    if df.isna().sum().sum() > 0:
        raise ValueError(
            "model_ready dataset contains missing values."
        )

    logger.info("Feature validation passed. No missing values detected.")

    # --------------------------------------------------
    # 12. Save Model-Ready Dataset
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
