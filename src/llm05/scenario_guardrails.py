# src/llm05/scenario_guardrails.py

"""
Scenario Guardrails
===================

Enforces strict business and logical constraints
before scenario execution.

Purpose:
--------
Prevent unsafe, unrealistic, or governance-violating
scenario transformations.

Strict Rules:
-------------
- Demand adjustment: ±50%
- Lead time: ≤ 90 days
- Service level: 0.80–0.99

Behavior:
---------
- Fail-fast validation
- No partial execution
- Clear, actionable error messages
- Fully defensive input checking
"""

from typing import Dict


# =========================================================
# SCENARIO VALIDATION
# =========================================================

def validate_scenario(parsed: Dict) -> bool:
    """
    Validate parsed scenario dictionary against
    industrial guardrail constraints.

    Parameters
    ----------
    parsed : Dict
        Parsed structured scenario dictionary.

    Returns
    -------
    bool
        True if validation passes.

    Raises
    ------
    ValueError
        If any guardrail is violated.
    """

    if not isinstance(parsed, dict):
        raise ValueError("Parsed scenario must be a dictionary.")

    # =====================================================
    # Demand Adjustment Guardrail
    # =====================================================

    if "adjustment_pct" in parsed:

        adjustment = parsed["adjustment_pct"]

        if not isinstance(adjustment, (int, float)):
            raise ValueError("Demand adjustment must be numeric.")

        if abs(adjustment) > 50:
            raise ValueError(
                "Demand adjustment exceeds ±50% limit."
            )

    # =====================================================
    # Filter Type Validation
    # =====================================================

    if "filter_type" in parsed:

        allowed_filters = {"festival", "weekend"}

        filter_type = parsed["filter_type"]

        if filter_type is not None and filter_type not in allowed_filters:
            raise ValueError(
                f"Unsupported filter type '{filter_type}'. "
                f"Allowed values: {sorted(allowed_filters)}"
            )

    # =====================================================
    # Lead Time Guardrail
    # =====================================================

    if "lead_time_days" in parsed:

        lead_time = parsed["lead_time_days"]

        if not isinstance(lead_time, (int, float)):
            raise ValueError("Lead time must be numeric.")

        if lead_time < 0:
            raise ValueError("Lead time cannot be negative.")

        if lead_time > 90:
            raise ValueError(
                "Lead time exceeds maximum allowed limit of 90 days."
            )

    # =====================================================
    # Service Level Guardrail
    # =====================================================

    if "service_level" in parsed:

        service_level = parsed["service_level"]

        if not isinstance(service_level, (int, float)):
            raise ValueError("Service level must be numeric.")

        if service_level < 0.80 or service_level > 0.99:
            raise ValueError(
                "Service level must be between 0.80 and 0.99."
            )

    return True
