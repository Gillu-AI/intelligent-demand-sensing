# src/pipelines/run_visualization.py

"""
Visualization Pipeline Orchestrator
====================================

Enterprise-grade visualization workflow for IDS.

Responsible for:
----------------
1. Loading model comparison metrics
2. Loading feature importance (if available)
3. Loading forecast output
4. Loading inventory output
5. Calling visualization modules

Design Principles:
------------------
- No plotting logic here
- Fully config-driven
- Fail-fast validation
- Logger-only (no print)
- Safe for production execution
"""

import os
import pandas as pd

from utils.config_loader import load_config
from utils.logger import get_logger
from utils.helpers import ensure_directory, validate_dataframe_not_empty

from visualization06.performance_plots import plot_model_performance
from visualization06.forecast_plots import plot_forecast
from visualization06.inventory_plots import plot_inventory_metrics
from visualization06.model_plot import (
    plot_model_comparison,
    plot_feature_importance
)


# ==========================================================
# Visualization Orchestrator
# ==========================================================

def run_visualization():
    """
    Execute full visualization workflow.

    Loads outputs generated from:
    - Training
    - Forecasting
    - Inventory

    Then generates visual artifacts.
    """

    config = load_config()
    logger = get_logger(config)

    logger.info("Starting visualization pipeline.")

    plots_dir = config["paths"]["output"]["plots"]
    ensure_directory(plots_dir)

    # ======================================================
    # 1. Model Comparison
    # ======================================================

    metrics_dir = config["paths"]["output"]["metrics"]

    if not os.path.exists(metrics_dir):
        logger.warning("Metrics directory not found. Skipping model comparison plot.")
    else:
        metrics_files = sorted(
            [
                f for f in os.listdir(metrics_dir)
                if f.startswith("model_comparison_") and f.endswith(".csv")
            ]
        )

        if metrics_files:
            latest_metrics = os.path.join(metrics_dir, metrics_files[-1])
            metrics_df = pd.read_csv(latest_metrics, index_col=0)

            validate_dataframe_not_empty(metrics_df, "metrics_df")

            plot_model_comparison(metrics_df, config, logger)
            plot_model_performance(metrics_df, config, logger)

            logger.info("Model performance visualizations generated.")
        else:
            logger.warning("No model comparison CSV found. Skipping.")

    # ======================================================
    # 2. Feature Importance (Optional)
    # ======================================================

    model_dir = config["paths"]["output"]["model"]

    if os.path.exists(model_dir):

        fi_files = sorted(
            [
                f for f in os.listdir(model_dir)
                if f.startswith("feature_importance_") and f.endswith(".csv")
            ]
        )

        if fi_files:
            latest_fi = os.path.join(model_dir, fi_files[-1])
            fi_df = pd.read_csv(latest_fi)

            validate_dataframe_not_empty(fi_df, "feature_importance_df")

            plot_feature_importance(fi_df, config, logger)
            logger.info("Feature importance visualization generated.")
        else:
            logger.info("No feature importance file found. Skipping.")
    else:
        logger.warning("Model directory not found. Skipping feature importance.")

    # ======================================================
    # 3. Forecast Visualization
    # ======================================================

    forecast_dir = config["paths"]["output"]["forecasts"]

    if os.path.exists(forecast_dir):

        forecast_files = sorted(
            [
                f for f in os.listdir(forecast_dir)
                if f.endswith("-futuresales.csv")
            ]
        )

        if forecast_files:
            latest_forecast = os.path.join(
                forecast_dir,
                forecast_files[-1]
            )

            forecast_df = pd.read_csv(latest_forecast)

            validate_dataframe_not_empty(forecast_df, "forecast_df")

            plot_forecast(forecast_df, config, logger)
            logger.info("Forecast visualization generated.")
        else:
            logger.warning("No forecast file found. Skipping forecast plot.")
    else:
        logger.warning("Forecast directory not found. Skipping.")

    # ======================================================
    # 4. Inventory Visualization
    # ======================================================

    inventory_dir = config["paths"]["output"]["inventory"]

    if os.path.exists(inventory_dir):

        inv_files = sorted(
            [
                f for f in os.listdir(inventory_dir)
                if f.startswith("inventory_plan_")
                and f.endswith(".csv")
            ]
        )

        if inv_files:
            latest_inv = os.path.join(
                inventory_dir,
                inv_files[-1]
            )

            inv_df = pd.read_csv(latest_inv)

            validate_dataframe_not_empty(inv_df, "inventory_df")

            plot_inventory_metrics(inv_df, config, logger)
            logger.info("Inventory visualization generated.")
        else:
            logger.warning("No inventory file found. Skipping inventory plot.")
    else:
        logger.warning("Inventory directory not found. Skipping.")

    logger.info("Visualization pipeline completed successfully.")


if __name__ == "__main__":
    run_visualization()
