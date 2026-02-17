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

    Parameters
    ----------
    path : str
        Directory path to create if missing.

    Notes
    -----
    - Safe to call multiple times.
    - Does nothing if directory already exists.
    - Used in pipelines before writing artifacts.
    """
    os.makedirs(path, exist_ok=True)


# ==========================================================
# Filename Utilities
# ==========================================================

def build_period_filename(start_date: str, end_date: str, suffix: str) -> str:
    """
    Create a standardized filename using a date range.

    Example
    -------
    build_period_filename("2025-03-01", "2025-12-31", "futuresales.csv")

    Returns:
        Mar2025-Dec2025-futuresales.csv

    Parameters
    ----------
    start_date : str
        Forecast start date (ISO format recommended).
    end_date : str
        Forecast end date.
    suffix : str
        File suffix including extension.

    Returns
    -------
    str
        Generated filename string.
    """

    start_fmt = pd.to_datetime(start_date).strftime("%b%Y")
    end_fmt = pd.to_datetime(end_date).strftime("%b%Y")

    return f"{start_fmt}-{end_fmt}-{suffix}"


def generate_timestamp(fmt: str = "%Y%m%d_%H%M") -> str:
    """
    Generate formatted timestamp string.

    Default format:
        YYYYMMDD_HHMM

    Used for:
    - Model versioning
    - Inventory versioning
    - Scenario outputs
    - Experiment tracking

    Parameters
    ----------
    fmt : str, optional
        Datetime format string.

    Returns
    -------
    str
        Formatted timestamp.
    """
    return datetime.now().strftime(fmt)


# ==========================================================
# Validation Utilities
# ==========================================================

def validate_dataframe_not_empty(df: pd.DataFrame, name: str) -> None:
    """
    Raise an error if a dataframe is empty.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    name : str
        Logical name of dataframe (used in error message).

    Raises
    ------
    ValueError
        If dataframe is empty.

    Purpose
    -------
    Fail-fast validation to avoid silent downstream errors.
    """
    if df.empty:
        raise ValueError(f"{name} dataframe is empty.")
