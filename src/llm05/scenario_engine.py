# src/llm05/scenario_engine.py

"""
Scenario Engine
===============

Applies structured scenario transformations
to baseline forecast dataframe.

Responsibilities:
-----------------
- Preserve original baseline forecast
- Apply controlled demand adjustments
- Support full and filtered scenarios
- Compute delta metrics
- Produce deterministic output schema

Output Columns:
---------------
date
base_forecast
scenario_forecast
delta_units
delta_pct

Design Principles:
------------------
- No in-place mutation of baseline values
- Fully defensive validation
- Deterministic rounding
- Filter-aware adjustments
- Fail-fast behavior
"""

import pandas as pd
from typing import Dict


# =========================================================
# CORE SCENARIO APPLICATION
# =========================================================

def apply_scenario(
    baseline_df: pd.DataFrame,
    parsed: Dict,
    date_col: str,
    target_col: str
) -> pd.DataFrame:
    """
    Apply structured scenario modification to baseline forecast.

    Parameters
    ----------
    baseline_df : pd.DataFrame
        Baseline forecast dataframe.

    parsed : Dict
        Parsed scenario dictionary.
        Possible keys:
            - filter_type
            - adjustment_pct
            - lead_time_days
            - service_level

    date_col : str
        Name of date column.

    target_col : str
        Name of baseline forecast column.

    Returns
    -------
    pd.DataFrame
        Scenario-adjusted dataframe including:
            base_forecast
            scenario_forecast
            delta_units
            delta_pct
    """

    # -----------------------------------------------------
    # Defensive Validation
    # -----------------------------------------------------

    if not isinstance(baseline_df, pd.DataFrame):
        raise ValueError("baseline_df must be a pandas DataFrame.")

    if baseline_df.empty:
        raise ValueError("baseline_df cannot be empty.")

    if not isinstance(parsed, dict):
        raise ValueError("Parsed scenario must be a dictionary.")

    if date_col not in baseline_df.columns:
        raise ValueError(f"Date column '{date_col}' not found.")

    if target_col not in baseline_df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    adjustment_pct = parsed.get("adjustment_pct")

    if adjustment_pct is None:
        raise ValueError("Scenario must include 'adjustment_pct'.")

    if not isinstance(adjustment_pct, (int, float)):
        raise ValueError("adjustment_pct must be numeric.")

    # -----------------------------------------------------
    # Prepare Working Copy
    # -----------------------------------------------------

    df = baseline_df.copy()

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(f"Column '{target_col}' must be numeric.")

    df = df.rename(columns={target_col: "base_forecast"})

    df["scenario_forecast"] = df["base_forecast"].astype(float)

    adjustment_factor = 1 + (adjustment_pct / 100)

    filter_type = parsed.get("filter_type")

    # -----------------------------------------------------
    # Apply Scenario Logic
    # -----------------------------------------------------

    if filter_type is None:
        # Full scenario (no filter)
        df["scenario_forecast"] = df["scenario_forecast"] * adjustment_factor

    elif filter_type == "festival":

        if "is_festival" not in baseline_df.columns:
            raise ValueError(
                "Column 'is_festival' required for festival scenario."
            )

        mask = baseline_df["is_festival"] == 1
        df.loc[mask, "scenario_forecast"] = (
            df.loc[mask, "scenario_forecast"] * adjustment_factor
        )

    elif filter_type == "weekend":

        if "is_weekend" not in baseline_df.columns:
            raise ValueError(
                "Column 'is_weekend' required for weekend scenario."
            )

        mask = baseline_df["is_weekend"] == 1
        df.loc[mask, "scenario_forecast"] = (
            df.loc[mask, "scenario_forecast"] * adjustment_factor
        )

    else:
        raise ValueError(
            f"Unsupported filter_type '{filter_type}'."
        )

    # -----------------------------------------------------
    # Compute Delta Metrics
    # -----------------------------------------------------

    df["delta_units"] = (
        df["scenario_forecast"] - df["base_forecast"]
    )

    df["delta_pct"] = (
        df["delta_units"]
        .div(df["base_forecast"].replace(0, pd.NA))
        .fillna(0)
        * 100
    )

    # -----------------------------------------------------
    # Rounding (Deterministic)
    # -----------------------------------------------------

    df["scenario_forecast"] = df["scenario_forecast"].round(2)
    df["delta_units"] = df["delta_units"].round(2)
    df["delta_pct"] = df["delta_pct"].round(2)

    # -----------------------------------------------------
    # Deterministic Column Ordering
    # -----------------------------------------------------

    ordered_cols = [
        date_col,
        "base_forecast",
        "scenario_forecast",
        "delta_units",
        "delta_pct"
    ]

    remaining_cols = [
        col for col in df.columns
        if col not in ordered_cols
    ]

    df = df[ordered_cols + remaining_cols]

    return df
