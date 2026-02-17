# src/llm05/scenario_guardrails.py

"""
Scenario Guardrails
===================

Prevents unsafe or invalid scenario execution.
"""

from typing import Dict


def validate_scenario(parsed: Dict):
    """
    Enforce business and logical constraints.
    """

    if parsed.get("adjustment_pct") and abs(parsed["adjustment_pct"]) > 200:
        raise ValueError("Adjustment percentage too large.")

    if parsed.get("filter_type") not in [None, "festival", "weekend"]:
        raise ValueError("Unsupported filter type.")

    if parsed.get("lead_time_days") and parsed["lead_time_days"] < 0:
        raise ValueError("Lead time cannot be negative.")

    return True
