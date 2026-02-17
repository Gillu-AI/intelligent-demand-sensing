# src/visualization06/model_plot.py

"""
Model Visualization Module
==========================

Responsible for visualizing model training outputs.

Plots:
------
1. Model comparison bar chart (MAPE primary)
2. Feature importance bar chart (if available)

Design Principles:
------------------
- Pure visualization (no modeling logic)
- Config-driven paths
- Fail-fast validation
- Reusable across projects
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

from utils.helpers import ensure_directory
from utils.helpers import validate_dataframe_not_empty


# ==========================================================
# Model Comparison Plot
# ==========================================================

def plot_model_comparison(
    metrics_df: pd.DataFrame,
    config: Dict,
    logger
) -> None:
    """
    Plot model comparison using MAPE (%).

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Model comparison dataframe.
    config : Dict
        Configuration dictionary.
    logger : logging.Logger
        Logger instance.
    """

    validate_dataframe_not_empty(metrics_df, "metrics_df")

    if "MAPE (%)" not in metrics_df.columns:
        raise ValueError("MAPE (%) column missing in metrics dataframe.")

    output_dir = config["paths"]["output"]["plots"]
    ensure_directory(output_dir)

    plt.figure(figsize=(12, 6))

    sorted_df = metrics_df.sort_values(by="MAPE (%)")

    plt.bar(
        sorted_df.index,
        sorted_df["MAPE (%)"]
    )

    plt.xticks(rotation=45)
    plt.ylabel("MAPE (%)")
    plt.title("Model Comparison (Lower is Better)")
    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        "model_comparison.png"
    )

    plt.savefig(output_path)
    plt.close()

    logger.info(f"Model comparison plot saved: {output_path}")


# ==========================================================
# Feature Importance Plot
# ==========================================================

def plot_feature_importance(
    feature_importance_df: pd.DataFrame,
    config: Dict,
    logger,
    top_n: int = 20
) -> None:
    """
    Plot feature importance (top N).

    Parameters
    ----------
    feature_importance_df : pd.DataFrame
        Feature importance dataframe with columns:
        ['feature', 'importance']
    config : Dict
        Configuration dictionary.
    logger : logging.Logger
        Logger instance.
    top_n : int
        Number of top features to plot.
    """

    validate_dataframe_not_empty(
        feature_importance_df,
        "feature_importance_df"
    )

    required_cols = ["feature", "importance"]
    for col in required_cols:
        if col not in feature_importance_df.columns:
            raise ValueError(f"{col} missing in feature importance dataframe.")

    output_dir = config["paths"]["output"]["plots"]
    ensure_directory(output_dir)

    top_features = (
        feature_importance_df
        .sort_values(by="importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(12, 8))

    plt.barh(
        top_features["feature"],
        top_features["importance"]
    )

    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importance")
    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        "feature_importance.png"
    )

    plt.savefig(output_path)
    plt.close()

    logger.info(f"Feature importance plot saved: {output_path}")
