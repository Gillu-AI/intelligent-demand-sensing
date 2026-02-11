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
        - features.aggregation.same_weekday_window

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

    if not config["features"].get("aggregation_features", {}).get("enabled", False):
        return df

    df = df.copy()

    date_col = config["data_schema"]["sales"]["date_column"]
    target_col = config["data_schema"]["sales"]["target_column"]
    window = config["features"]["aggregation_features"]["same_weekday_window"]

    if "day_of_week" not in df.columns:
        raise ValueError("day_of_week column required before aggregation.")

    df[target_col + "_shifted"] = df[target_col].shift(1)

    df["avg_sales_same_weekday"] = (
        df.groupby("day_of_week")[target_col + "_shifted"]
        .transform(lambda x: x.rolling(window).mean())
    )

    df.drop(columns=[target_col + "_shifted"], inplace=True)

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

    if not config["features"].get("aggregation_features", {}).get("enabled", False):
        return df

    df = df.copy()

    if "demand_class" not in df.columns:
        raise ValueError("demand_class column required for previous week aggregation.")

    df["demand_class_prev_week"] = df["demand_class"].shift(7)

    return df
