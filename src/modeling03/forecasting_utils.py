# src/modeling03/forecasting_utils.py

"""
Recursive Forecasting Utilities
===============================

Implements efficient multi-step recursive forecasting.

Key Characteristics:
-------------------
- Fully config-driven
- Computes features only for last row
- No full dataframe recomputation
- No leakage
- Production scalable
"""

import pandas as pd
import numpy as np
from typing import Dict


# ==========================================================
# Generate Future Date Range
# ==========================================================

def generate_future_dates(config: Dict) -> pd.DatetimeIndex:
    """
    Generate future date range from config.

    Raises validation error if invalid range.
    """

    if "forecasting" not in config:
        raise ValueError("Missing 'forecasting' configuration.")

    forecasting_cfg = config["forecasting"]

    required_keys = {"start_date", "end_date", "frequency"}
    missing = required_keys - forecasting_cfg.keys()

    if missing:
        raise ValueError(f"Missing forecasting keys: {sorted(missing)}")

    start_date = pd.to_datetime(forecasting_cfg["start_date"])
    end_date = pd.to_datetime(forecasting_cfg["end_date"])
    freq = forecasting_cfg["frequency"]

    if end_date <= start_date:
        raise ValueError("forecasting.end_date must be after start_date")

    return pd.date_range(start=start_date, end=end_date, freq=freq)


# ==========================================================
# Recursive Forecast Engine
# ==========================================================

def recursive_forecast(
    model,
    historical_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """
    Perform industrial-grade recursive multi-step forecasting.
    """

    # --------------------------------------------------
    # Defensive Config Validation
    # --------------------------------------------------

    if "data_schema" not in config or "sales" not in config["data_schema"]:
        raise ValueError("Missing 'data_schema.sales' configuration.")

    if "features" not in config:
        raise ValueError("Missing 'features' configuration.")

    sales_schema = config["data_schema"]["sales"]

    required_schema_keys = {"date_column", "target_column"}
    missing_schema = required_schema_keys - sales_schema.keys()

    if missing_schema:
        raise ValueError(
            f"Missing required sales schema keys: {sorted(missing_schema)}"
        )

    date_col = sales_schema["date_column"]
    target_col = sales_schema["target_column"]

    if date_col not in historical_df.columns:
        raise ValueError(f"'{date_col}' not found in historical dataset.")

    if target_col not in historical_df.columns:
        raise ValueError(f"'{target_col}' not found in historical dataset.")

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(historical_df[date_col]):
        raise ValueError(
            f"'{date_col}' must be datetime type before forecasting."
        )

    features_cfg = config["features"]

    lag_enabled = features_cfg.get("lag_features", {}).get("enabled", False)
    rolling_enabled = features_cfg.get("rolling_features", {}).get("enabled", False)
    agg_enabled = features_cfg.get("aggregation_features", {}).get("enabled", False)
    time_enabled = features_cfg.get("time_features", {}).get("enabled", False)

    lags = features_cfg.get("lag_features", {}).get("lags", [])
    windows = features_cfg.get("rolling_features", {}).get("windows", [])
    same_weekday_window = features_cfg.get(
        "aggregation_features", {}
    ).get("same_weekday_window", 0)

    df = historical_df.copy()

    future_dates = generate_future_dates(config)

    results = []

    for future_date in future_dates:

        # --------------------------------------------------
        # 1. Append new row
        # --------------------------------------------------

        new_row = {date_col: future_date}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        last_idx = df.index[-1]

        # --------------------------------------------------
        # 2. Time Features (respect enable flag)
        # --------------------------------------------------

        if time_enabled:

            time_feature_list = features_cfg.get("time_features", {}).get(
                "features", []
            )

            if "day_of_week" in time_feature_list:
                df.loc[last_idx, "day_of_week"] = future_date.weekday()

            if "month" in time_feature_list:
                df.loc[last_idx, "month"] = future_date.month

            if "quarter" in time_feature_list:
                df.loc[last_idx, "quarter"] = future_date.quarter

            if "is_weekend" in time_feature_list:
                df.loc[last_idx, "is_weekend"] = int(future_date.weekday() >= 5)

            if "is_month_start" in time_feature_list:
                df.loc[last_idx, "is_month_start"] = int(future_date.is_month_start)

            if "is_month_end" in time_feature_list:
                df.loc[last_idx, "is_month_end"] = int(future_date.is_month_end)

        # --------------------------------------------------
        # 3. Calendar Merge (safe handling)
        # --------------------------------------------------

        if isinstance(calendar_df, pd.DataFrame) and not calendar_df.empty:

            if date_col not in calendar_df.columns:
                raise ValueError(
                    f"Calendar dataframe must contain '{date_col}' column."
                )

            cal_row = calendar_df[calendar_df[date_col] == future_date]

            if not cal_row.empty:
                for col in cal_row.columns:
                    if col != date_col:
                        df.loc[last_idx, col] = cal_row.iloc[0][col]

        # --------------------------------------------------
        # 4. Lag Features
        # --------------------------------------------------

        if lag_enabled:

            for lag in lags:
                col_name = f"{target_col}_lag_{lag}"

                if len(df) > lag:
                    df.loc[last_idx, col_name] = df[target_col].iloc[-lag-1]
                else:
                    df.loc[last_idx, col_name] = np.nan

        # --------------------------------------------------
        # 5. Rolling Features
        # --------------------------------------------------

        if rolling_enabled:

            for window in windows:
                mean_col = f"{target_col}_rolling_mean_{window}"
                std_col = f"{target_col}_rolling_std_{window}"

                if len(df) > window:
                    window_slice = df[target_col].iloc[-window-1:-1]
                    df.loc[last_idx, mean_col] = window_slice.mean()
                    df.loc[last_idx, std_col] = window_slice.std()
                else:
                    df.loc[last_idx, mean_col] = np.nan
                    df.loc[last_idx, std_col] = np.nan

        # --------------------------------------------------
        # 6. Same Weekday Aggregation
        # --------------------------------------------------

        if agg_enabled and same_weekday_window > 0:

            weekday = future_date.weekday()

            same_weekday_values = df[
                df[date_col].dt.weekday == weekday
            ][target_col].dropna().tail(same_weekday_window)

            col_name = "avg_sales_same_weekday"

            if len(same_weekday_values) > 0:
                df.loc[last_idx, col_name] = same_weekday_values.mean()
            else:
                df.loc[last_idx, col_name] = np.nan

        # --------------------------------------------------
        # 7. Prepare Model Input (numeric safe)
        # --------------------------------------------------

        feature_row = df.iloc[last_idx:last_idx+1].drop(columns=[date_col])

        feature_row = feature_row.select_dtypes(include=["number"])

        if feature_row.empty:
            raise ValueError(
                "No numeric features available for model prediction."
            )

        prediction = model.predict(feature_row)[0]

        # --------------------------------------------------
        # 8. Insert Prediction
        # --------------------------------------------------

        df.loc[last_idx, target_col] = prediction

        results.append({
            date_col: future_date,
            target_col: prediction
        })

    return pd.DataFrame(results)
