# src/ingestion01/csv_ingestion.py

"""
CSV ingestion module for the IDS project.

Responsibilities
----------------
- Read CSV files using strictly config-driven behavior
- Perform defensive validation of ingestion inputs
- Return raw data without applying schema or transformations

Design Principles
-----------------
- No hardcoded defaults
- No schema enforcement
- No feature logic
- Fail fast with clear, actionable errors

This module is intentionally dumb and reusable.
"""

from pathlib import Path
from typing import Dict
import logging

import pandas as pd


logger = logging.getLogger(__name__)


def ingest(
    dataset_name: str,
    dataset_cfg: Dict,
    global_cfg: Dict,
    paths_cfg: Dict,
) -> pd.DataFrame:
    """
    Ingest a CSV dataset based on configuration provided in config.yaml.

    Parameters
    ----------
    dataset_name : str
        Logical name of the dataset (used for logging and error context)
    dataset_cfg : dict
        Dataset-specific ingestion configuration (file, csv block, etc.)
    global_cfg : dict
        Global ingestion configuration (header, skip_rows, na_values)
    paths_cfg : dict
        Resolved project paths configuration

    Returns
    -------
    pandas.DataFrame
        Raw DataFrame loaded from the CSV file

    Raises
    ------
    ValueError
        If required configuration keys are missing
    FileNotFoundError
        If the CSV file does not exist
    TypeError
        If ingestion does not return a pandas DataFrame
    """

    # --------------------------------------------------
    # Resolve file path
    # --------------------------------------------------
    raw_dir = Path(paths_cfg["data"]["raw"])
    file_name = dataset_cfg.get("file")

    if not isinstance(file_name, str) or not file_name:
        raise ValueError(
            f"[CSV INGESTION] 'file' must be a non-empty string for dataset '{dataset_name}'"
        ) 

    file_path = raw_dir / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"[CSV INGESTION] File not found for dataset '{dataset_name}': {file_path}"
        )

    # --------------------------------------------------
    # Global ingestion config (MANDATORY)
    # --------------------------------------------------
    
    required_global_keys = {"header", "skip_rows", "na_values"}
    missing_keys = required_global_keys - global_cfg.keys()

    if missing_keys:
        raise ValueError(
            f"[CSV INGESTION] Missing global ingestion keys for '{dataset_name}': "
            f"{sorted(missing_keys)}"
        )

        header_flag = global_cfg["header"]

        if not isinstance(header_flag, bool):
            raise ValueError(
                f"[CSV INGESTION] 'header' must be boolean for dataset '{dataset_name}'"
            )

        skip_rows = global_cfg["skip_rows"]
        na_values = global_cfg["na_values"]

        header = 0 if header_flag else None

    # --------------------------------------------------
    # CSV-specific config (MANDATORY)
    # --------------------------------------------------
    if "csv" not in dataset_cfg:
        raise ValueError(
            f"[CSV INGESTION] Missing 'csv' config block for dataset '{dataset_name}'"
        )

    csv_cfg = dataset_cfg["csv"]

    if "delimiter" not in csv_cfg:
        raise ValueError(
            f"[CSV INGESTION] Missing 'delimiter' in csv config for '{dataset_name}'"
        )

    if "encoding" not in csv_cfg:
        raise ValueError(
            f"[CSV INGESTION] Missing 'encoding' in csv config for '{dataset_name}'"
        )

    delimiter = csv_cfg["delimiter"]
    encoding = csv_cfg["encoding"]

    if not isinstance(delimiter, str):
        raise ValueError(
            f"[CSV INGESTION] 'delimiter' must be string for dataset '{dataset_name}'"
        )

    if not isinstance(encoding, str):
        raise ValueError(
            f"[CSV INGESTION] 'encoding' must be string for dataset '{dataset_name}'"
        )

    # --------------------------------------------------
    # Read CSV
    # --------------------------------------------------
    logger.info(
        f"[CSV INGESTION] Reading dataset '{dataset_name}' from {file_path}"
    )

    df = pd.read_csv(
        file_path,
        sep=delimiter,
        encoding=encoding,
        header=header,
        skiprows=skip_rows,
        na_values=na_values,
    )

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"[CSV INGESTION] Expected pandas DataFrame for dataset "
            f"'{dataset_name}', got {type(df).__name__}"
        )

    logger.info(
        f"[CSV INGESTION] Successfully loaded dataset '{dataset_name}' "
        f"with shape {df.shape}"
    )

    return df
