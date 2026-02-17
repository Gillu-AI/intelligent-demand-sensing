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

    service_level = config["inventory"]["service_level"]
    lead_time = config["inventory"]["lead_time_days"]

    # Average daily demand from forecast window
    avg_daily_demand = forecast_series.mean()

    safety_stock = calculate_safety_stock(
        historical_demand,
        service_level,
        lead_time
    )

    reorder_point = (avg_daily_demand * lead_time) + safety_stock

    order_qty = max(0, reorder_point - current_inventory)

    return {
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2),
        "recommended_order_qty": round(order_qty, 2),
    }
