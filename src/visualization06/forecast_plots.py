# src/visualization06/forecast_plots.py

"""
Forecast Visualization Module
=============================

Responsible for visualizing forecast outputs.

Plots:
------
1. Daily forecast line chart
2. Weekly totals overlay (if available)
3. Monthly totals overlay (if available)
4. 3-month rolling average markers (if available)

Design Principles:
------------------
- Fully config-driven paths
- No hardcoded filenames
- Fail-fast validation
- Pure visualization logic only
- Reusable across projects
"""

import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

from utils.helpers import validate_dataframe_not_empty
from utils.helpers import ensure_directory


# ==========================================================
# Main Forecast Plot
# ==========================================================

def plot_forecast(forecast_df: pd.DataFrame, config: Dict, logger) -> None:
    """
    Generate forecast visualization chart.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        DataFrame containing future forecast output.
    config : Dict
        Project configuration dictionary.
    logger : logging.Logger
        Project logger.
    """

    # ------------------------------------------------------
    # Visualization Toggle Validation
    # ------------------------------------------------------

    if not config.get("visualization", {}).get("enabled", False):
        logger.info("Visualization disabled via config. Skipping forecast plot.")
        return

    validate_dataframe_not_empty(forecast_df, "forecast_df")

    date_col = config["data_schema"]["sales"]["date_column"]
    target_col = config["data_schema"]["sales"]["target_column"]

    if date_col not in forecast_df.columns:
        raise ValueError(f"{date_col} not found in forecast dataframe.")

    if target_col not in forecast_df.columns:
        raise ValueError(f"{target_col} not found in forecast dataframe.")

    forecast_df = forecast_df.copy()
    forecast_df[date_col] = pd.to_datetime(forecast_df[date_col])

    output_dir = config["paths"]["output"]["plots"]
    ensure_directory(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure(figsize=(14, 7))

    # ------------------------------------------------------
    # Daily Forecast Line
    # ------------------------------------------------------
    plt.plot(
        forecast_df[date_col],
        forecast_df[target_col],
        label="Daily Forecast"
    )

    # ------------------------------------------------------
    # Weekly Overlay (if present)
    # ------------------------------------------------------
    if "weekly_total_sales" in forecast_df.columns:
        weekly_points = forecast_df.dropna(
            subset=["weekly_total_sales"]
        )

        plt.scatter(
            weekly_points[date_col],
            weekly_points["weekly_total_sales"],
            label="Weekly Total",
            marker="o"
        )

    # ------------------------------------------------------
    # Monthly Overlay (if present)
    # ------------------------------------------------------
    if "monthly_total_sales" in forecast_df.columns:
        monthly_points = forecast_df.dropna(
            subset=["monthly_total_sales"]
        )

        plt.scatter(
            monthly_points[date_col],
            monthly_points["monthly_total_sales"],
            label="Monthly Total",
            marker="s"
        )

    # ------------------------------------------------------
    # 3-Month Average Markers (if present)
    # ------------------------------------------------------
    if "three_month_avg_sales" in forecast_df.columns:
        avg_points = forecast_df.dropna(
            subset=["three_month_avg_sales"]
        )

        plt.scatter(
            avg_points[date_col],
            avg_points["three_month_avg_sales"],
            label="3-Month Avg",
            marker="D"
        )

    plt.title("Demand Forecast Visualization")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        f"forecast_visualization_{timestamp}.png"
    )

    plt.savefig(output_path)
    plt.close()

    logger.info(f"Forecast visualization saved: {output_path}")
