# src/llm05/scenario_parser.py

"""
Scenario Parser
===============

Converts user prompt into structured scenario instructions.
"""

import re
from typing import Dict


def parse_prompt(prompt: str) -> Dict:
    """
    Extract structured scenario elements.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")

    prompt_lower = prompt.lower()

    parsed = {
        "adjustment_pct": None,
        "filter_type": None,
        "lead_time_days": None
    }

    # --------------------------------------------------
    # Percentage Extraction (supports negative)
    # --------------------------------------------------

    pct_match = re.search(r"(-?\d+)\s*%", prompt_lower)
    if pct_match:
        parsed["adjustment_pct"] = int(pct_match.group(1))

    # --------------------------------------------------
    # Filter Type Detection (mutually exclusive)
    # --------------------------------------------------

    festival_flag = "festival" in prompt_lower
    weekend_flag = "weekend" in prompt_lower

    if festival_flag and weekend_flag:
        raise ValueError(
            "Ambiguous filter: specify either 'festival' or 'weekend', not both."
        )

    if festival_flag:
        parsed["filter_type"] = "festival"

    if weekend_flag:
        parsed["filter_type"] = "weekend"

    # --------------------------------------------------
    # Lead Time / Lag Extraction
    # --------------------------------------------------

    lag_match = re.search(r"(lead\s*time|lag)[^\d]*(-?\d+)", prompt_lower)
    if lag_match:
        parsed["lead_time_days"] = int(lag_match.group(2))

    return parsed
