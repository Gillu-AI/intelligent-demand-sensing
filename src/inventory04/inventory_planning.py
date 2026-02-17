# src/inventory04/inventory_planning.py

"""
Inventory Planning Module
==========================

Transforms forecast output into actionable inventory decisions:

- Safety Stock
- Reorder Point
- Recommended Order Quantity
"""

import pandas as pd
from typing import Dict
from .safety_stock import calculate_safety_stock


def generate_inventory_plan(
    historical_demand: pd.Series,
    forecast_series: pd.Series,
    current_inventory: float,
    config: Dict
) -> Dict:
    """
    Generate inventory planning metrics.

    Parameters
    ----------
    historical_demand : pd.Series
        Past demand used for variability calculation.
    forecast_series : pd.Series
        Future predicted demand.
    current_inventory : float
        Current available stock.
    config : Dict
        Project configuration.

    Returns
    -------
    Dict
        Dictionary containing:
        - safety_stock
        - reorder_point
        - recommended_order_qty
    """

    # --------------------------------------------------
    # Defensive Input Validation
    # --------------------------------------------------

    if not isinstance(historical_demand, pd.Series):
        raise ValueError("historical_demand must be a pandas Series.")

    if not isinstance(forecast_series, pd.Series):
        raise ValueError("forecast_series must be a pandas Series.")

    if forecast_series.empty:
        raise ValueError("forecast_series cannot be empty.")

    if not isinstance(current_inventory, (int, float)):
        raise ValueError("current_inventory must be numeric.")

    if "inventory" not in config:
        raise ValueError("Missing 'inventory' configuration.")

    inventory_cfg = config["inventory"]

    required_keys = {"service_level", "lead_time_days"}
    missing = required_keys - inventory_cfg.keys()

    if missing:
        raise ValueError(
            f"Missing inventory config keys: {sorted(missing)}"
        )

    service_level = inventory_cfg["service_level"]
    lead_time = inventory_cfg["lead_time_days"]

    if not (0 < service_level < 1):
        raise ValueError("service_level must be between 0 and 1.")

    if not isinstance(lead_time, (int, float)) or lead_time <= 0:
        raise ValueError("lead_time_days must be a positive number.")

    # --------------------------------------------------
    # Core Calculations
    # --------------------------------------------------

    avg_daily_demand = forecast_series.mean()

    safety_stock = calculate_safety_stock(
        historical_demand,
        service_level,
        lead_time
    )

    reorder_point = (avg_daily_demand * lead_time) + safety_stock

    order_qty = max(0, reorder_point - current_inventory)

    return {
        "safety_stock": round(float(safety_stock), 2),
        "reorder_point": round(float(reorder_point), 2),
        "recommended_order_qty": round(float(order_qty), 2),
    }
