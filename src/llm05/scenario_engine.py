# src/llm05/scenario_engine.py

"""
Scenario Engine
===============

Applies structured scenario transformations
to baseline forecast dataframe.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List


def apply_scenario(
    baseline_df: pd.DataFrame,
    parsed: Dict,
    date_col: str,
    target_col: str
) -> pd.DataFrame:
    """
    Apply scenario modifications to forecast.
    """

    if not isinstance(parsed, dict):
        raise ValueError("Parsed scenario must be a dictionary.")

    required_keys = {"filter_type", "adjustment_pct"}
    missing = required_keys - parsed.keys()
    if missing:
        raise ValueError(f"Missing scenario keys: {sorted(missing)}")

    if not isinstance(parsed["adjustment_pct"], (int, float)):
        raise ValueError("adjustment_pct must be numeric.")

    df = baseline_df.copy()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe.")

    adjustment_factor = 1 + parsed["adjustment_pct"] / 100

    if parsed["filter_type"] == "festival":

        if "is_festival" not in df.columns:
            raise ValueError("Column 'is_festival' not found for festival scenario.")

        mask = df["is_festival"] == 1
        df.loc[mask, target_col] *= adjustment_factor

    elif parsed["filter_type"] == "weekend":

        if "is_weekend" not in df.columns:
            raise ValueError("Column 'is_weekend' not found for weekend scenario.")

        mask = df["is_weekend"] == 1
        df.loc[mask, target_col] *= adjustment_factor

    else:
        raise ValueError(
            f"Unsupported filter_type '{parsed['filter_type']}'."
        )

    df[target_col] = df[target_col].round(2)

    return df


def generate_scenario_id(existing_ids: List[str]) -> str:
    """
    Generate incremental scenario ID in format SCN_XXX.
    """

    if not existing_ids:
        return "SCN_001"

    numeric_ids = []

    for sid in existing_ids:
        if sid.startswith("SCN_"):
            try:
                numeric_ids.append(int(sid.split("_")[1]))
            except Exception:
                continue

    next_id = max(numeric_ids, default=0) + 1

    return f"SCN_{str(next_id).zfill(3)}"
