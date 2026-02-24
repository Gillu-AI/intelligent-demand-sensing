# src/pipelines/run_ingestion.py

"""
Ingestion Pipeline Orchestrator
===============================

Module Purpose
--------------
Orchestrates ingestion workflow for IDS.

Responsibilities
----------------
1. Load raw datasets using universal_loader
2. Select configured demand source
3. Perform governance profiling snapshot
4. Apply staged duplicate and critical-column enforcement
5. Apply schema numeric casting
6. Apply missing value handling
7. Perform non-overlapping row removal attribution
8. Enforce strict date integrity
9. Persist deterministic cleaned artifacts

Row Removal Attribution (Option B)
-----------------------------------
Row removals are applied sequentially:

Stage 1  - Duplicate Removal
Stage 2  - Critical Column Enforcement (date, target)
Stage 3  - Schema Numeric Casting (no removal)
Stage 4  - Missing Handling (non-critical)

Each stage removes rows only from the remaining dataset.
No overlap is allowed.
Total Removed must equal sum of stage removals.
Failure to reconcile causes pipeline failure.

Execution Contract
------------------
Input:
    Raw datasets via ingestion configuration

Output:
    data/processed/cleaned_sales.parquet
    data/processed/cleaned_calendar.parquet (if enabled)

Failure Conditions
------------------
- Demand source missing
- Dataset empty
- Date invalid after cleaning
- Removal reconciliation mismatch
- Parquet save failure
"""

import os
from datetime import datetime
import pandas as pd
from tabulate import tabulate

from utils.config_loader import load_config
from utils.logger import get_logger
from utils.helpers import ensure_directory
from ingestion01.universal_loader import load_all_datasets
from utils.data_profiler import profile_dataframe
from utils.missing_handler import (
    handle_duplicates,
    enforce_schema_numeric_cast,
    enforce_critical_columns,
    handle_missing_values
)


def run_ingestion():

    config = load_config("config/config.yaml")
    logger = get_logger(config)

    logger.info("========== INGESTION PIPELINE STARTED ==========")

    try:

        # ------------------------------------------------------
        # Directory Setup
        # ------------------------------------------------------

        processed_dir = config["paths"]["data"]["processed"]
        ensure_directory(processed_dir)

        # ------------------------------------------------------
        # Load Datasets
        # ------------------------------------------------------

        datasets = load_all_datasets(config, config["paths"])
        demand_source = config["ingestion"]["demand_source"]

        if demand_source not in datasets:
            raise ValueError(f"Demand source '{demand_source}' not found.")

        sales_df = datasets[demand_source].copy()

        if sales_df.empty:
            raise ValueError(f"Loaded dataset '{demand_source}' is empty.")

        logger.info(f"Loaded dataset: {demand_source}")
        logger.info(f"Raw dataset shape: {sales_df.shape}")

        raw_row_count = len(sales_df)

        # ------------------------------------------------------
        # Profiling Snapshot
        # ------------------------------------------------------

        logger.info("Profiling sales dataset...")
        profile_report = profile_dataframe(sales_df)

        profile_output_dir = config["paths"]["output"]["reports"]
        ensure_directory(profile_output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        pd.DataFrame([{
            "row_count": profile_report["row_count"],
            "column_count": profile_report["column_count"],
            "duplicate_rows": profile_report["duplicate_rows"],
        }]).to_csv(
            os.path.join(profile_output_dir, f"data_profile_summary_{timestamp}.csv"),
            index=False
        )

        logger.info(f"Data profiling reports saved to: {profile_output_dir}")

        # ------------------------------------------------------
        # Staged Cleaning with Non-Overlapping Attribution
        # ------------------------------------------------------

        logger.info("Applying staged cleaning with reconciliation-safe attribution...")

        original_df = sales_df.copy()
        current_df = original_df.copy()

        removal_tracker = {}
        removed_indices = set()

        date_col = config["data_schema"]["sales"]["date_column"]
        target_col = config["data_schema"]["sales"]["target_column"]

        # ---------------- Stage 1: Duplicate Removal ----------------

        df_after_duplicates = handle_duplicates(
            df=current_df,
            unique_key=config["data_cleaning"][demand_source]["duplicate"].get("unique_key"),
            strategy=config["data_cleaning"][demand_source]["duplicate"].get("strategy", "keep_first")
        )

        dup_removed = set(current_df.index) - set(df_after_duplicates.index)
        removal_tracker["Duplicate Rows"] = len(dup_removed)

        removed_indices.update(dup_removed)
        current_df = df_after_duplicates.copy()

        # ---------------- Stage 2: Critical Column Enforcement ----------------

        df_after_critical = enforce_critical_columns(
            df=current_df,
            config=config,
            dataset_name=demand_source
        )

        critical_removed = set(current_df.index) - set(df_after_critical.index)

        missing_date = []
        missing_target = []

        for idx in critical_removed:
            row = current_df.loc[idx]
            if pd.isna(row[date_col]):
                missing_date.append(idx)
            elif pd.isna(row[target_col]):
                missing_target.append(idx)

        removal_tracker["Missing Date"] = len(missing_date)
        removal_tracker["Missing Target"] = len(missing_target)

        removed_indices.update(critical_removed)
        current_df = df_after_critical.copy()

        # ---------------- Stage 3: Schema Numeric Casting ----------------

        current_df = enforce_schema_numeric_cast(
            df=current_df,
            config=config,
            dataset_name=demand_source
        )

        # ---------------- Stage 4: Missing Value Handling ----------------

        df_after_missing = handle_missing_values(
            df=current_df,
            config=config,
            dataset_name=demand_source
        )

        other_removed = set(current_df.index) - set(df_after_missing.index)
        removal_tracker["Other Cleaning Rules"] = len(other_removed)

        removed_indices.update(other_removed)
        sales_cleaned = df_after_missing.copy()

        # ------------------------------------------------------
        # Reconciliation Validation
        # ------------------------------------------------------

        total_removed = len(removed_indices)

        sum_of_parts = sum(removal_tracker.values())

        if total_removed != sum_of_parts:
            raise ValueError(
                f"Removal reconciliation failed. "
                f"Total removed ({total_removed}) "
                f"!= Sum of stage removals ({sum_of_parts})."
            )

        summary_rows = [
            ("Duplicate Rows", removal_tracker.get("Duplicate Rows", 0)),
            ("Missing Date", removal_tracker.get("Missing Date", 0)),
            ("Missing Target", removal_tracker.get("Missing Target", 0)),
            ("Other Cleaning Rules", removal_tracker.get("Other Cleaning Rules", 0)),
        ]

        summary_table = tabulate(
            summary_rows,
            headers=["Reason", "Rows Removed"],
            tablefmt="grid"
        )

        logger.info("\n========== INGESTION ROW REMOVAL SUMMARY ==========")
        logger.info("\n" + summary_table)
        logger.info(f"\nTotal Rows Removed: {total_removed}")
        logger.info("====================================================\n")

        if sales_cleaned.empty:
            raise ValueError("Cleaned sales dataset is empty after processing.")

        # ------------------------------------------------------
        # Date Integrity Enforcement
        # ------------------------------------------------------

        sales_cleaned[date_col] = pd.to_datetime(
            sales_cleaned[date_col],
            errors="raise"
        )

        if sales_cleaned[date_col].isna().sum() > 0:
            raise ValueError("Date column contains missing values after cleaning.")

        sales_cleaned = (
            sales_cleaned
            .sort_values(by=date_col)
            .reset_index(drop=True)
        )

        # ------------------------------------------------------
        # Persist Cleaned Sales
        # ------------------------------------------------------

        sales_output_path = os.path.join(
            processed_dir,
            "cleaned_sales.parquet"
        )

        sales_cleaned.to_parquet(sales_output_path, index=False)

        logger.info(f"Cleaned sales dataset saved to: {sales_output_path}")

        # ------------------------------------------------------
        # Calendar Processing (If Present)
        # ------------------------------------------------------

        if "calendar" in datasets:

            logger.info("Processing calendar dataset...")

            calendar_df = datasets["calendar"].copy()
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