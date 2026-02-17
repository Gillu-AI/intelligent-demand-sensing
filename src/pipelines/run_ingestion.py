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

Design Principles:
------------------
- No schema logic here (handled in universal_loader)
- No feature logic here
- Fully config-driven
- Fail-fast validation
- Industrial logging enabled
- Audit-ready profiling output
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

    Steps:
    ------
    - Load datasets via universal loader
    - Profile raw data
    - Apply duplicate & missing handling
    - Validate output
    - Persist cleaned dataset

    This function is safe for production execution.
    """

    config = load_config()
    logger = get_logger(config)

    logger.info("========== INGESTION PIPELINE STARTED ==========")

    try:

        # ------------------------------------------------------
        # Validate Required Config Sections
        # ------------------------------------------------------

        if "paths" not in config:
            raise ValueError("Missing 'paths' configuration.")

        if "data" not in config["paths"]:
            raise ValueError("Missing 'paths.data' configuration.")

        if "processed" not in config["paths"]["data"]:
            raise ValueError("Missing 'paths.data.processed' configuration.")

        if "output" not in config["paths"]:
            raise ValueError("Missing 'paths.output' configuration.")

        if "reports" not in config["paths"]["output"]:
            raise ValueError("Missing 'paths.output.reports' configuration.")

        # ------------------------------------------------------
        # 1. Load Datasets via Universal Loader
        # ------------------------------------------------------

        paths_cfg = config["paths"]

        datasets = load_all_datasets(config, paths_cfg)

        demand_source = config["ingestion"]["demand_source"]

        if demand_source not in datasets:
            raise ValueError(
                f"Demand source '{demand_source}' not found in loaded datasets."
            )

        df = datasets[demand_source].copy()

        if df.empty:
            raise ValueError(
                f"Loaded dataset '{demand_source}' is empty."
            )

        logger.info(f"Loaded dataset: {demand_source}")
        logger.info(f"Raw dataset shape: {df.shape}")

        # ------------------------------------------------------
        # 2. Profile Data (Audit Trail)
        # ------------------------------------------------------

        logger.info("Profiling dataset...")

        profile_report = profile_dataframe(df)

        profile_output_dir = config["paths"]["output"]["reports"]
        ensure_directory(profile_output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        profile_output_path = os.path.join(
            profile_output_dir,
            f"data_profile_report_{timestamp}.csv"
        )

        profile_report.to_csv(profile_output_path, index=False)

        logger.info(f"Data profiling report saved to: {profile_output_path}")

        # ------------------------------------------------------
        # 3. Apply Cleaning (Duplicates + Missing)
        # ------------------------------------------------------

        logger.info("Applying duplicate and missing value handling...")

        df_cleaned = clean_dataframe(
            df=df,
            config=config,
            dataset_name=demand_source
        )

        logger.info(f"Cleaned dataset shape: {df_cleaned.shape}")

        # ------------------------------------------------------
        # 4. Final Validation
        # ------------------------------------------------------

        if df_cleaned.empty:
            raise ValueError(
                "Cleaned dataframe is empty after ingestion. "
                "Check duplicate/missing handling configuration."
            )

        date_col = config["data_schema"]["sales"]["date_column"]

        if date_col not in df_cleaned.columns:
            raise ValueError(
                f"Date column '{date_col}' missing after cleaning."
            )

        df_cleaned[date_col] = pd.to_datetime(
            df_cleaned[date_col],
            errors="raise"
        )

        df_cleaned = (
            df_cleaned
            .sort_values(by=date_col)
            .reset_index(drop=True)
        )

        logger.info("Date column validated and dataset sorted chronologically.")

        # ------------------------------------------------------
        # 5. Save Cleaned Dataset
        # ------------------------------------------------------

        processed_dir = config["paths"]["data"]["processed"]
        ensure_directory(processed_dir)

        output_path = os.path.join(
            processed_dir,
            f"cleaned_sales_{timestamp}.parquet"
        )

        df_cleaned.to_parquet(output_path, index=False)

        logger.info("Ingestion completed successfully.")
        logger.info(f"Cleaned dataset saved to: {output_path}")
        logger.info("========== INGESTION PIPELINE COMPLETED ==========")

    except Exception:
        logger.exception("Ingestion pipeline failed.")
        raise
