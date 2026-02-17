# src/inventory04/safety_stock.py

"""
Safety Stock Calculation Module
================================

Implements industry-standard safety stock calculation.

Formula:
--------
SafetyStock = Z × σ_d × √LeadTime

Where:
- Z = service level factor
- σ_d = demand standard deviation
- LeadTime = lead time in days

This module is reusable across any demand planning project.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_safety_stock(
    demand_series,
    service_level: float,
    lead_time_days: int
):
    """
    Calculate safety stock based on demand variability.

    Parameters
    ----------
    demand_series : pd.Series
        Historical demand values.
    service_level : float
        Desired service level (e.g., 0.95).
    lead_time_days : int
        Supplier lead time in days.

    Returns
    -------
    float
        Safety stock quantity.
    """

    # --------------------------------------------------
    # Defensive Validation
    # --------------------------------------------------

    if not isinstance(demand_series, pd.Series):
        raise ValueError("demand_series must be a pandas Series.")

    if len(demand_series) < 2:
        raise ValueError("Not enough data to calculate variability.")

    if not (0 < service_level < 1):
        raise ValueError("service_level must be between 0 and 1.")

    if not isinstance(lead_time_days, (int, float)) or lead_time_days <= 0:
        raise ValueError("lead_time_days must be a positive number.")

    # --------------------------------------------------
    # Core Calculation
    # --------------------------------------------------

    # Z-score for service level
    z_value = norm.ppf(service_level)

    # Sample standard deviation (ddof=1 for industry correctness)
    demand_std = np.std(demand_series, ddof=1)

    safety_stock = z_value * demand_std * np.sqrt(lead_time_days)

    return max(0, round(float(safety_stock), 2))
