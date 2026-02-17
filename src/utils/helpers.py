# src/utils/helpers.py

"""
Reusable Helper Utilities
==========================

Small, generic utility functions used across pipelines.

Design Principles:
------------------
- Fully reusable across projects
- No project-specific assumptions
- No hardcoded paths
- Lightweight and dependency-safe
- Safe for production use

These helpers are intentionally minimal and generic.
"""

import os
import pandas as pd
from datetime import datetime
from typing import Optional


# ==========================================================
# Filesystem Utilities
# ==========================================================

def ensure_directory(path: str) -> None:
    """
    Ensure that a directory exists.
    """

    if not isinstance(path, str) or not path.strip():
        raise ValueError("Directory path must be a non-empty string.")

    os.makedirs(path, exist_ok=True)


# ==========================================================
# Filename Utilities
# ==========================================================

def build_period_filename(start_date: str, end_date: str, suffix: str) -> str:
    """
    Create a standardized filename using a date range.
    """

    if not isinstance(suffix, str) or not suffix.strip():
        raise ValueError("suffix must be a non-empty string.")

    try:
        start_fmt = pd.to_datetime(start_date).strftime("%b%Y")
        end_fmt = pd.to_datetime(end_date).strftime("%b%Y")
    except Exception as e:
        raise ValueError(
            f"Invalid date format provided to build_period_filename: {e}"
        )

    return f"{start_fmt}-{end_fmt}-{suffix}"


def generate_timestamp(fmt: str = "%Y%m%d_%H%M") -> str:
    """
    Generate formatted timestamp string.
    """

    if not isinstance(fmt, str) or not fmt.strip():
        raise ValueError("Timestamp format must be a non-empty string.")

    return datetime.now().strftime(fmt)


# ==========================================================
# Validation Utilities
# ==========================================================

def validate_dataframe_not_empty(df: pd.DataFrame, name: str) -> None:
    """
    Raise an error if a dataframe is empty.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")

    if df.empty:
        raise ValueError(f"{name} dataframe is empty.")
