# src/pipelines/run_inventory.py

"""
Inventory Pipeline Orchestrator
================================

Enterprise-grade inventory planning workflow for IDS.

This pipeline converts recursive demand forecast into:

1. Daily forecast output
2. Weekly summary (week-end only rows)
3. Monthly summary + growth %
4. Versioned inventory planning output
5. Optional LLM scenario simulation layer

Design Principles:
------------------
- Fully config-driven
- Strict logging (no print statements)
- Production-safe versioning
- Fail-fast validation
- Clean separation of daily vs aggregated outputs
- LLM invoked only if enabled
"""

import os
import joblib
import pandas as pd
from datetime import datetime

from utils.config_loader import load_config
from utils.logger import get_logger
from utils.helpers import ensure_directory

from modeling03.forecasting_utils import recursive_forecast
from inventory04.inventory_planning import generate_inventory_plan


# ==========================================================
# Main Inventory Pipeline
# ==========================================================

def run_inventory():
    """
    Execute full inventory planning workflow.
    """

    config = load_config("config/config.yaml")
    logger = get_logger(config)

    logger.info("========== INVENTORY PIPELINE STARTED ==========")

    if not config.get("forecasting", {}).get("enabled", False):
        raise ValueError("Forecasting is disabled in config.")

    execution_mode = config.get("execution", {}).get("mode", "dev")

    # ------------------------------------------------------
    # 1. Load Best Model
    # ------------------------------------------------------

    model_path = os.path.join(
        config["paths"]["output"]["model"],
        "best_model_latest.pkl"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "best_model_latest.pkl not found. "
            "Run training pipeline first."
        )

    model = joblib.load(model_path)
    logger.info("Loaded best model successfully.")

    # ------------------------------------------------------
    # 2. Load Model-Ready Dataset
    # ------------------------------------------------------

    data_path = os.path.join(
        config["paths"]["data"]["processed"],
        "model_ready.parquet"
    )

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "model_ready.parquet not found. "
            "Run feature pipeline first."
        )

    df = pd.read_parquet(data_path)

    sales_schema = config["data_schema"]["sales"]
    date_col = sales_schema["date_column"]
    target_col = sales_schema["target_column"]

    if date_col not in df.columns:
        raise ValueError(f"{date_col} missing in model_ready dataset.")

    if target_col not in df.columns:
        raise ValueError(f"{target_col} missing in model_ready dataset.")

    df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    df = df.sort_values(by=date_col).reset_index(drop=True)

    logger.info(f"Loaded historical dataset with shape: {df.shape}")

    # ------------------------------------------------------
    # 3. Historical Demand
    # ------------------------------------------------------

    historical_demand = df[target_col]

    # ------------------------------------------------------
    # 4. Recursive Future Forecast
    # ------------------------------------------------------

    logger.info("Generating recursive future forecast.")
    calendar_df = pd.DataFrame()

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

        calendar_date_col = config["data_schema"]["calendar"]["date_column"]

        calendar_df[calendar_date_col] = pd.to_datetime(
            calendar_df[calendar_date_col],
            errors="raise"
        )

    future_df = recursive_forecast(
        model=model,
        historical_df=df,
        calendar_df=calendar_df,
        config=config
    )

    if future_df.empty:
        raise ValueError("Recursive forecast returned empty dataframe.")

    logger.info(f"Generated forecast with shape: {future_df.shape}")

    # ------------------------------------------------------
    # 5. Weekly Summary (Week-End Only)
    # ------------------------------------------------------

    if config["forecasting"]["output"]["include_weekly_summary"]:

        weekly_summary = (
            future_df
            .resample("W", on=date_col)[target_col]
            .sum()
            .reset_index()
            .rename(columns={target_col: "weekly_total_sales"})
        )

        future_df = future_df.merge(
            weekly_summary,
            on=date_col,
            how="left"
        )

        logger.info("Weekly summary applied.")

    # ------------------------------------------------------
    # 6. Monthly Summary + Growth %
    # ------------------------------------------------------

    monthly_summary = None

    if config["forecasting"]["output"]["include_monthly_summary"]:

        monthly_summary = (
            future_df
            .resample("M", on=date_col)[target_col]
            .sum()
            .reset_index()
            .rename(columns={
                date_col: "month",
                target_col: "monthly_total_sales"
            })
        )

        monthly_summary["monthly_growth_pct"] = (
            monthly_summary["monthly_total_sales"]
            .pct_change() * 100
        ).round(2)

        logger.info("Monthly summary calculated.")

    # ------------------------------------------------------
    # 7. Save Forecast Output
    # ------------------------------------------------------

    forecast_output_dir = config["paths"]["output"]["forecasts"]
    ensure_directory(forecast_output_dir)

    start_str = config["forecasting"]["start_date"]
    end_str = config["forecasting"]["end_date"]

    start_fmt = pd.to_datetime(start_str).strftime("%b%Y")
    end_fmt = pd.to_datetime(end_str).strftime("%b%Y")

    forecast_filename = f"{start_fmt}-{end_fmt}-futuresales.csv"
    forecast_path = os.path.join(
        forecast_output_dir,
        forecast_filename
    )

    future_df.to_csv(forecast_path, index=False)
    logger.info(f"Forecast saved to: {forecast_path}")

    if monthly_summary is not None:

        monthly_filename = f"{start_fmt}-{end_fmt}-monthly_summary.csv"
        monthly_path = os.path.join(
            forecast_output_dir,
            monthly_filename
        )

        monthly_summary.to_csv(monthly_path, index=False)
        logger.info(f"Monthly summary saved to: {monthly_path}")

    # ------------------------------------------------------
    # 8. Inventory Planning
    # ------------------------------------------------------

    inventory_cfg = config["inventory"]

    if execution_mode == "prod" and inventory_cfg.get("simulated_current_inventory") is not None:
        raise ValueError(
            "simulated_current_inventory is not allowed in production mode. "
            "Configure real inventory source."
        )

    current_inventory = inventory_cfg.get("simulated_current_inventory")

    if current_inventory is None:
        raise ValueError("Current inventory value not configured.")

    inventory_result = generate_inventory_plan(
        historical_demand=historical_demand,
        forecast_series=future_df[target_col],
        current_inventory=current_inventory,
        config=config
    )

    # Add inventory reference to result (for visualization consistency)
    inventory_result["current_inventory"] = current_inventory

    inventory_output_dir = config["paths"]["output"]["inventory"]
    ensure_directory(inventory_output_dir)

    versioned_path = os.path.join(
        inventory_output_dir,
        "inventory_plan.csv"
    )

    latest_path = os.path.join(
        inventory_output_dir,
        "inventory_plan_latest.csv"
    )

    pd.DataFrame([inventory_result]).to_csv(versioned_path, index=False)
    pd.DataFrame([inventory_result]).to_csv(latest_path, index=False)

    logger.info("Inventory planning completed successfully.")

    # ------------------------------------------------------
    # 9. LLM Scenario Layer
    # ------------------------------------------------------

    if config.get("llm", {}).get("enabled", False):

        logger.info("Launching LLM scenario CLI.")

        from llm05.scenario_cli import run_cli

        run_cli(
            baseline_forecast_path=forecast_path
        )

    logger.info("========== INVENTORY PIPELINE COMPLETED ==========")
