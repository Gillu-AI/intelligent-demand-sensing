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

    start_date = pd.to_datetime(config["forecasting"]["start_date"])
    end_date = pd.to_datetime(config["forecasting"]["end_date"])
    freq = config["forecasting"]["frequency"]

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

    Parameters
    ----------
    model : trained model
    historical_df : pd.DataFrame
        Fully feature-engineered historical dataset
    calendar_df : pd.DataFrame
        Calendar dataset
    config : Dict

    Returns
    -------
    pd.DataFrame
        Future predictions with features
    """

    df = historical_df.copy()

    date_col = config["data_schema"]["sales"]["date_column"]
    target_col = config["data_schema"]["sales"]["target_column"]

    lags = config["features"]["lag_features"]["lags"]
    windows = config["features"]["rolling_features"]["windows"]
    same_weekday_window = config["features"]["aggregation_features"]["same_weekday_window"]

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
        # 2. Time Features (for last row only)
        # --------------------------------------------------

        df.loc[last_idx, "day_of_week"] = future_date.weekday()
        df.loc[last_idx, "month"] = future_date.month
        df.loc[last_idx, "quarter"] = future_date.quarter
        df.loc[last_idx, "is_weekend"] = int(future_date.weekday() >= 5)
        df.loc[last_idx, "is_month_start"] = int(future_date.is_month_start)
        df.loc[last_idx, "is_month_end"] = int(future_date.is_month_end)

        # --------------------------------------------------
        # 3. Calendar Merge (single row)
        # --------------------------------------------------

        if not calendar_df.empty:
            cal_row = calendar_df[
                calendar_df[date_col] == future_date
            ]

            if not cal_row.empty:
                for col in cal_row.columns:
                    if col != date_col:
                        df.loc[last_idx, col] = cal_row.iloc[0][col]

        # --------------------------------------------------
        # 4. Lag Features
        # --------------------------------------------------

        for lag in lags:
            col_name = f"lag_{lag}"

            if len(df) > lag:
                df.loc[last_idx, col_name] = df[target_col].iloc[-lag-1]
            else:
                df.loc[last_idx, col_name] = np.nan

        # --------------------------------------------------
        # 5. Rolling Features
        # --------------------------------------------------

        for window in windows:
            col_name = f"rolling_{window}"

            if len(df) > window:
                df.loc[last_idx, col_name] = \
                    df[target_col].iloc[-window-1:-1].mean()
            else:
                df.loc[last_idx, col_name] = np.nan

        # --------------------------------------------------
        # 6. Same Weekday Aggregation
        # --------------------------------------------------

        weekday = future_date.weekday()

        same_weekday_values = df[
            df[date_col].dt.weekday == weekday
        ][target_col].dropna().tail(same_weekday_window)

        if len(same_weekday_values) > 0:
            df.loc[last_idx, "same_weekday_avg"] = \
                same_weekday_values.mean()
        else:
            df.loc[last_idx, "same_weekday_avg"] = np.nan

        # --------------------------------------------------
        # 7. Prepare Model Input
        # --------------------------------------------------

        feature_row = df.iloc[last_idx:last_idx+1].drop(
            columns=[date_col]
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
