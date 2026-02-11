# src/features02/demand_bucketing.py

from typing import Dict
import pandas as pd


def add_demand_bucketing(
    df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """
    Create demand_class feature based on configured bucketing method.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing target column.
    config : Dict
        Configuration dictionary. Uses:
            - data_schema.sales.target_column
            - features.demand_bucketing.method
            - features.demand_bucketing.labels

    Returns
    -------
    pd.DataFrame
        DataFrame with 'demand_class' column added.

    Raises
    ------
    ValueError
        If target column missing or unsupported method provided.

    Notes
    -----
    - Default method: 'quantile'
    - Uses pandas qcut for equal-frequency bins.
    - Does not introduce leakage (based on historical data only).
    - Intended to be executed AFTER lag features.
    """

    df = df.copy()

    target_col = config["data_schema"]["sales"]["target_column"]
    method = config["features"]["demand_bucketing"]["method"]
    labels = config["features"]["demand_bucketing"]["labels"]

    if target_col not in df.columns:
        raise ValueError(f"{target_col} not found in DataFrame")

    if method != "quantile":
        raise ValueError(f"Unsupported bucketing method: {method}")

    if len(labels) < 2:
        raise ValueError("At least two labels required for bucketing.")

    df["demand_class"] = pd.qcut(
        df[target_col],
        q=len(labels),
        labels=labels,
        duplicates="drop"
    )

    return df
