# src/visualization06/performance_plots.py

"""
Performance Visualization Module
================================

Responsible for visualizing model evaluation results.

Plots:
------
1. MAPE comparison bar chart (Primary metric)
2. MAE / RMSE comparison chart (Optional)
3. R2 comparison (if available)

Design Principles:
------------------
- Fully config-driven paths
- No hardcoded filenames
- Fail-fast validation
- No business logic
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
# Plot: MAPE Comparison
# ==========================================================

def plot_mape_comparison(metrics_df: pd.DataFrame,
                         config: Dict,
                         logger) -> None:
    """
    Generate bar chart comparing MAPE across models.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame containing model evaluation metrics.
    config : Dict
        Project configuration dictionary.
    logger : logging.Logger
        Project logger.
    """

    # ------------------------------------------------------
    # Visualization Toggle
    # ------------------------------------------------------

    if not config.get("visualization", {}).get("enabled", False):
        logger.info("Visualization disabled via config. Skipping MAPE plot.")
        return

    validate_dataframe_not_empty(metrics_df, "metrics_df")

    if "MAPE (%)" not in metrics_df.columns:
        raise ValueError("MAPE (%) column not found in metrics dataframe.")

    if "paths" not in config or "output" not in config["paths"]:
        raise ValueError("Missing paths.output configuration.")

    output_dir = config["paths"]["output"]["plots"]
    ensure_directory(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure(figsize=(10, 6))

    metrics_df["MAPE (%)"].sort_values().plot(
        kind="bar",
        title="Model Comparison - MAPE (%)"
    )

    plt.ylabel("MAPE (%)")
    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        f"performance_mape_{timestamp}.png"
    )

    plt.savefig(output_path)
    plt.close()

    logger.info(f"MAPE performance plot saved: {output_path}")


# ==========================================================
# Plot: MAE & RMSE Comparison
# ==========================================================

def plot_error_comparison(metrics_df: pd.DataFrame,
                          config: Dict,
                          logger) -> None:
    """
    Generate grouped bar chart comparing MAE & RMSE.

    Parameters
    ----------
    metrics_df : pd.DataFrame
    config : Dict
    logger : logging.Logger
    """

    # ------------------------------------------------------
    # Visualization Toggle
    # ------------------------------------------------------

    if not config.get("visualization", {}).get("enabled", False):
        logger.info("Visualization disabled via config. Skipping error comparison plot.")
        return

    validate_dataframe_not_empty(metrics_df, "metrics_df")

    required_cols = ["MAE", "RMSE"]
    for col in required_cols:
        if col not in metrics_df.columns:
            logger.warning(f"{col} not found in metrics dataframe. Skipping error comparison.")
            return

    output_dir = config["paths"]["output"]["plots"]
    ensure_directory(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure(figsize=(12, 6))

    metrics_df[required_cols].plot(kind="bar")
    plt.title("Model Comparison - MAE & RMSE")
    plt.ylabel("Error")
    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        f"performance_error_{timestamp}.png"
    )

    plt.savefig(output_path)
    plt.close()

    logger.info(f"Error comparison plot saved: {output_path}")
