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

    if not isinstance(parsed, dict):
        raise ValueError("Parsed scenario must be a dictionary.")

    # --------------------------------------------------
    # Adjustment Percentage Validation
    # --------------------------------------------------

    if "adjustment_pct" in parsed:

        if not isinstance(parsed["adjustment_pct"], (int, float)):
            raise ValueError("adjustment_pct must be numeric.")

        if abs(parsed["adjustment_pct"]) > 200:
            raise ValueError("Adjustment percentage too large.")

    # --------------------------------------------------
    # Filter Type Validation
    # --------------------------------------------------

    if "filter_type" in parsed:

        allowed_filters = {"festival", "weekend"}

        if parsed["filter_type"] is not None and \
           parsed["filter_type"] not in allowed_filters:

            raise ValueError("Unsupported filter type.")

    # --------------------------------------------------
    # Lead Time Validation (if used in future)
    # --------------------------------------------------

    if "lead_time_days" in parsed:

        if not isinstance(parsed["lead_time_days"], (int, float)):
            raise ValueError("lead_time_days must be numeric.")

        if parsed["lead_time_days"] < 0:
            raise ValueError("Lead time cannot be negative.")

    return True
