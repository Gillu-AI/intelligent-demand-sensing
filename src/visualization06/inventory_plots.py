# src/visualization06/inventory_plots.py

"""
Inventory Visualization Module
==============================

Responsible for visualizing inventory planning output.

Plots:
------
1. Forecast demand line
2. Current inventory reference line
3. Safety stock line
4. Reorder point line

Design Principles:
------------------
- Pure visualization only
- No business logic
- Fully config-driven
- Fail-fast validation
- Reusable across projects
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

from utils.helpers import validate_dataframe_not_empty
from utils.helpers import ensure_directory


# ==========================================================
# Inventory Planning Plot
# ==========================================================

def plot_inventory_plan(
    forecast_df: pd.DataFrame,
    inventory_result: Dict,
    config: Dict,
    logger
) -> None:
    """
    Visualize inventory planning metrics.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Forecast dataframe (future horizon).
    inventory_result : Dict
        Output from generate_inventory_plan().
    config : Dict
        Project configuration dictionary.
    logger : logging.Logger
        Logger instance.
    """

    validate_dataframe_not_empty(forecast_df, "forecast_df")

    date_col = config["data_schema"]["sales"]["date_column"]
    target_col = config["data_schema"]["sales"]["target_column"]

    if date_col not in forecast_df.columns:
        raise ValueError(f"{date_col} not found in forecast dataframe.")

    if target_col not in forecast_df.columns:
        raise ValueError(f"{target_col} not found in forecast dataframe.")

    required_keys = [
        "safety_stock",
        "reorder_point",
        "current_inventory"
    ]

    for key in required_keys:
        if key not in inventory_result:
            raise ValueError(f"{key} missing in inventory_result.")

    forecast_df[date_col] = pd.to_datetime(forecast_df[date_col])

    output_dir = config["paths"]["output"]["plots"]
    ensure_directory(output_dir)

    plt.figure(figsize=(14, 7))

    # ------------------------------------------------------
    # Forecast Demand Line
    # ------------------------------------------------------
    plt.plot(
        forecast_df[date_col],
        forecast_df[target_col],
        label="Forecast Demand"
    )

    # ------------------------------------------------------
    # Current Inventory (Horizontal Line)
    # ------------------------------------------------------
    plt.axhline(
        y=inventory_result["current_inventory"],
        linestyle="--",
        label="Current Inventory"
    )

    # ------------------------------------------------------
    # Safety Stock Line
    # ------------------------------------------------------
    plt.axhline(
        y=inventory_result["safety_stock"],
        linestyle=":",
        label="Safety Stock"
    )

    # ------------------------------------------------------
    # Reorder Point Line
    # ------------------------------------------------------
    plt.axhline(
        y=inventory_result["reorder_point"],
        linestyle="-.",
        label="Reorder Point"
    )

    plt.title("Inventory Planning Visualization")
    plt.xlabel("Date")
    plt.ylabel("Units")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        "inventory_visualization.png"
    )

    plt.savefig(output_path)
    plt.close()

    logger.info(f"Inventory visualization saved: {output_path}")
