# src/llm05/scenario_engine.py

"""
Scenario Engine
===============

Applies structured scenario transformations
to baseline forecast dataframe.
"""

import pandas as pd
from datetime import datetime
from typing import Dict


def apply_scenario(
    baseline_df: pd.DataFrame,
    parsed: Dict,
    date_col: str,
    target_col: str
) -> pd.DataFrame:
    """
    Apply scenario modifications to forecast.
    """

    df = baseline_df.copy()

    if parsed["filter_type"] == "festival":
        if "is_festival" in df.columns:
            mask = df["is_festival"] == 1
            df.loc[mask, target_col] *= (1 + parsed["adjustment_pct"] / 100)

    if parsed["filter_type"] == "weekend":
        if "is_weekend" in df.columns:
            mask = df["is_weekend"] == 1
            df.loc[mask, target_col] *= (1 + parsed["adjustment_pct"] / 100)

    df[target_col] = df[target_col].round(2)

    return df


def generate_scenario_id(existing_ids):
    base = len(existing_ids) + 1
    return f"SCN_{str(base).zfill(3)}"
