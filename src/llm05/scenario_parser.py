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

    parsed = {
        "adjustment_pct": None,
        "filter_type": None,
        "lead_time_days": None
    }

    pct_match = re.search(r"(\d+)%", prompt)
    if pct_match:
        parsed["adjustment_pct"] = int(pct_match.group(1))

    if "festival" in prompt.lower():
        parsed["filter_type"] = "festival"

    if "weekend" in prompt.lower():
        parsed["filter_type"] = "weekend"

    lag_match = re.search(r"lag.*?(\d+)", prompt.lower())
    if lag_match:
        parsed["lead_time_days"] = int(lag_match.group(1))

    return parsed
