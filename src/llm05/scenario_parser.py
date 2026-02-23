# src/llm05/scenario_parser.py

"""
Scenario Parser
===============

Converts natural language scenario prompts into
structured, machine-readable scenario instructions.

Responsibilities:
-----------------
- Extract demand adjustment percentage
- Detect filter type (festival / weekend)
- Extract lead time modifications
- Extract service level modifications
- Enforce unambiguous filter usage
- Fail-fast if no actionable instruction found

This module does NOT apply logic.
It only parses and structures.
"""

import re
from typing import Dict


# =========================================================
# PROMPT PARSER
# =========================================================

def parse_prompt(prompt: str) -> Dict:
    """
    Parse user prompt into structured scenario dictionary.

    Parameters
    ----------
    prompt : str
        Natural language scenario instruction.

    Returns
    -------
    Dict
        Structured scenario parameters.

    Raises
    ------
    ValueError
        If prompt is invalid or contains ambiguity.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")

    prompt_lower = prompt.lower()

    parsed = {
        "adjustment_pct": None,
        "filter_type": None,
        "lead_time_days": None,
        "service_level": None
    }

    # =====================================================
    # Percentage Extraction (supports negative values)
    # =====================================================

    pct_match = re.search(r"(-?\d+)\s*%", prompt_lower)
    if pct_match:
        parsed["adjustment_pct"] = int(pct_match.group(1))

    # =====================================================
    # Filter Detection (mutually exclusive)
    # =====================================================

    festival_flag = "festival" in prompt_lower
    weekend_flag = "weekend" in prompt_lower

    if festival_flag and weekend_flag:
        raise ValueError(
            "Ambiguous filter detected. Specify either "
            "'festival' or 'weekend', not both."
        )

    if festival_flag:
        parsed["filter_type"] = "festival"

    if weekend_flag:
        parsed["filter_type"] = "weekend"

    # =====================================================
    # Lead Time Extraction
    # =====================================================

    lead_match = re.search(r"(lead\s*time|lag)[^\d]*(-?\d+)", prompt_lower)
    if lead_match:
        parsed["lead_time_days"] = int(lead_match.group(2))

    # =====================================================
    # Service Level Extraction
    # =====================================================

    service_match = re.search(
        r"(service\s*level)[^\d]*(0\.\d+|\d?\.\d+)",
        prompt_lower
    )

    if service_match:
        parsed["service_level"] = float(service_match.group(2))

    # =====================================================
    # Ensure At Least One Actionable Instruction
    # =====================================================

    actionable_fields = [
        parsed["adjustment_pct"],
        parsed["lead_time_days"],
        parsed["service_level"]
    ]

    if all(value is None for value in actionable_fields):
        raise ValueError(
            "No actionable instruction detected in prompt."
        )

    return parsed
