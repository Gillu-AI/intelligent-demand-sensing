# src/features02/aggregation_features.py

from typing import Dict
import pandas as pd


def add_avg_sales_same_weekday(
    df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """
    Add average sales for same weekday over past N weeks.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed, sorted dataset with time features.
    config : Dict
        Configuration dictionary. Uses:
        - data_schema.sales.date_column
        - data_schema.sales.target_column
        - features.aggregation_features.same_weekday_window

    Returns
    -------
    pd.DataFrame
        DataFrame with avg_sales_same_weekday feature added.

    Notes
    -----
    - Uses shift(1) before grouping.
    - Prevents current-day leakage.
    - Requires day_of_week feature to exist.
    """

    # --------------------------------------------------
    # Validate feature config structure
    # --------------------------------------------------
    if "features" not in config:
        raise ValueError("Missing 'features' section in configuration.")

    aggregation_cfg = config["features"].get("aggregation_features")
    if not isinstance(aggregation_cfg, dict):
        raise ValueError(
            "Missing or invalid 'features.aggregation_features' configuration."
        )

    if not aggregation_cfg.get("enabled", False):
        return df

    df = df.copy()

    # --------------------------------------------------
    # Validate required schema config
    # --------------------------------------------------
    if "data_schema" not in config or "sales" not in config["data_schema"]:
        raise ValueError("Missing 'data_schema.sales' configuration.")

    sales_schema = config["data_schema"]["sales"]

    date_col = sales_schema["date_column"]
    target_col = sales_schema["target_column"]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    if "day_of_week" not in df.columns:
        raise ValueError("day_of_week column required before aggregation.")

    # --------------------------------------------------
    # Validate rolling window
    # --------------------------------------------------
    window = aggregation_cfg.get("same_weekday_window")

    if not isinstance(window, int) or window <= 0:
        raise ValueError(
            "'same_weekday_window' must be a positive integer."
        )

    # --------------------------------------------------
    # Leakage-safe rolling calculation
    # --------------------------------------------------
    shifted_col = f"{target_col}_shifted"

    df[shifted_col] = df[target_col].shift(1)

    df["avg_sales_same_weekday"] = (
        df.groupby("day_of_week")[shifted_col]
        .transform(lambda x: x.rolling(window).mean())
    )

    df.drop(columns=[shifted_col], inplace=True)

    return df


def add_previous_week_demand_class(
    df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """
    Add previous week's demand class label.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing demand bucket labels.
    config : Dict
        Configuration dictionary.

    Returns
    -------
    pd.DataFrame
        DataFrame with demand_class_prev_week feature added.

    Notes
    -----
    - Uses shift(7) to access previous week.
    - Requires demand_class column.
    - Leakage-safe by construction.
    """

    # --------------------------------------------------
    # Validate feature config structure
    # --------------------------------------------------
    if "features" not in config:
        raise ValueError("Missing 'features' section in configuration.")

    aggregation_cfg = config["features"].get("aggregation_features")
    if not isinstance(aggregation_cfg, dict):
        raise ValueError(
            "Missing or invalid 'features.aggregation_features' configuration."
        )

    if not aggregation_cfg.get("enabled", False):
        return df

    df = df.copy()

    if "demand_class" not in df.columns:
        raise ValueError(
            "demand_class column required for previous week aggregation."
        )

    df["demand_class_prev_week"] = df["demand_class"].shift(7)

    return df
