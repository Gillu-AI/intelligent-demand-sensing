# src/pipelines/run_ingestion.py

"""
Ingestion Pipeline Orchestrator
===============================

Responsible for:

1. Loading raw datasets using universal_loader
2. Selecting demand source dataset
3. Applying duplicate handling
4. Applying missing value treatment
5. Saving cleaned_sales.parquet
6. Saving cleaned_calendar.parquet (if enabled)

Design Principles:
------------------
- No schema logic here (handled in universal_loader)
- No feature logic here
- Fully config-driven
- Fail-fast validation
- Industrial logging enabled
- Deterministic artifact generation
"""

import os
from datetime import datetime
import pandas as pd

from utils.config_loader import load_config
from utils.logger import get_logger
from utils.helpers import ensure_directory
from ingestion01.universal_loader import load_all_datasets
from utils.data_profiler import profile_dataframe
from utils.missing_handler import clean_dataframe


# ==========================================================
# Main Ingestion Pipeline
# ==========================================================

def run_ingestion():
    """
    Execute ingestion + cleaning workflow.

    Produces:
        - cleaned_sales.parquet
        - cleaned_calendar.parquet (if calendar enabled)
    """

    config = load_config("config/config.yaml")
    logger = get_logger(config)

    logger.info("========== INGESTION PIPELINE STARTED ==========")

    try:

        # ------------------------------------------------------
        # Validate Required Config Sections
        # ------------------------------------------------------

        required_paths = [
            ("paths", None),
            ("paths", "data"),
            ("paths", "output"),
        ]

        for parent, child in required_paths:
            if parent not in config:
                raise ValueError(f"Missing '{parent}' configuration.")
            if child and child not in config[parent]:
                raise ValueError(f"Missing '{parent}.{child}' configuration.")

        processed_dir = config["paths"]["data"]["processed"]
        ensure_directory(processed_dir)

        # ------------------------------------------------------
        # 1. Load Datasets via Universal Loader
        # ------------------------------------------------------

        datasets = load_all_datasets(config, config["paths"])

        demand_source = config["ingestion"]["demand_source"]

        if demand_source not in datasets:
            raise ValueError(
                f"Demand source '{demand_source}' not found in loaded datasets."
            )

        sales_df = datasets[demand_source].copy()

        if sales_df.empty:
            raise ValueError(f"Loaded dataset '{demand_source}' is empty.")

        logger.info(f"Loaded dataset: {demand_source}")
        logger.info(f"Raw dataset shape: {sales_df.shape}")

        # ------------------------------------------------------
        # 2. Profile Sales Dataset
        # ------------------------------------------------------

        logger.info("Profiling sales dataset...")

        profile_report = profile_dataframe(sales_df)

        profile_output_dir = config["paths"]["output"]["reports"]
        ensure_directory(profile_output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        summary_df = pd.DataFrame([{
            "row_count": profile_report["row_count"],
            "column_count": profile_report["column_count"],
            "duplicate_rows": profile_report["duplicate_rows"],
            "candidate_keys": ", ".join(profile_report["candidate_keys"]),
        }])

        summary_df.to_csv(
            os.path.join(profile_output_dir, f"data_profile_summary_{timestamp}.csv"),
            index=False
        )

        profile_report["missing_summary"].to_csv(
            os.path.join(profile_output_dir, f"data_profile_missing_{timestamp}.csv")
        )

        profile_report["cardinality"].to_csv(
            os.path.join(profile_output_dir, f"data_profile_cardinality_{timestamp}.csv")
        )

        logger.info(f"Data profiling reports saved to: {profile_output_dir}")

        # ------------------------------------------------------
        # 3. Clean Sales Dataset
        # ------------------------------------------------------

        logger.info("Applying duplicate and missing value handling (sales)...")

        sales_cleaned = clean_dataframe(
            df=sales_df,
            config=config,
            dataset_name=demand_source
        )

        if sales_cleaned.empty:
            raise ValueError("Cleaned sales dataset is empty after processing.")

        date_col = config["data_schema"]["sales"]["date_column"]

        sales_cleaned[date_col] = pd.to_datetime(
            sales_cleaned[date_col],
            errors="raise"
        )

        sales_cleaned = (
            sales_cleaned
            .sort_values(by=date_col)
            .reset_index(drop=True)
        )

        # ------------------------------------------------------
        # 4. Save Cleaned Sales Dataset
        # ------------------------------------------------------

        sales_output_path = os.path.join(
            processed_dir,
            "cleaned_sales.parquet"
        )

        sales_cleaned.to_parquet(sales_output_path, index=False)

        logger.info(f"Cleaned sales dataset saved to: {sales_output_path}")

        # ------------------------------------------------------
        # 5. Clean & Save Calendar Dataset (If Present)
        # ------------------------------------------------------

        if "calendar" in datasets:

            logger.info("Processing calendar dataset...")

            calendar_df = datasets["calendar"].copy()

            if calendar_df.empty:
                raise ValueError("Calendar dataset is empty.")

            calendar_date_col = config["data_schema"]["calendar"]["date_column"]

            calendar_df[calendar_date_col] = pd.to_datetime(
                calendar_df[calendar_date_col],
                errors="raise"
            )

            calendar_df = (
                calendar_df
                .sort_values(by=calendar_date_col)
                .reset_index(drop=True)
            )

            calendar_output_path = os.path.join(
                processed_dir,
                "cleaned_calendar.parquet"
            )

            calendar_df.to_parquet(calendar_output_path, index=False)

            logger.info(f"Cleaned calendar dataset saved to: {calendar_output_path}")

        logger.info("========== INGESTION PIPELINE COMPLETED ==========")

    except Exception:
        logger.exception("Ingestion pipeline failed.")
        raise


if __name__ == "__main__":
    run_ingestion()
