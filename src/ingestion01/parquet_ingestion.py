"""
Parquet ingestion module for the IDS project.

ROLE
----
Source-specific ingestion layer responsible for reading Parquet files
in a strictly config-driven, production-safe manner.

RESPONSIBILITIES
----------------
- Resolve Parquet file paths from project configuration
- Read Parquet data using the configured engine
- Perform defensive validation of ingestion outputs
- Emit structured ingestion logs for observability
- Return raw data without schema enforcement or transformations

WHAT THIS MODULE DOES NOT DO
----------------------------
- Does NOT apply schema validation
- Does NOT perform feature engineering
- Does NOT assume defaults
- Does NOT modify data semantics

DESIGN PRINCIPLES
-----------------
- Config-driven only
- Fail fast with clear, actionable errors
- Reusable across projects
- Consistent with universal ingestion contract

This module is intentionally dumb and isolated.
"""

from pathlib import Path
from typing import Dict, Any
import logging

import pandas as pd


logger = logging.getLogger(__name__)


def ingest(
    dataset_name: str,
    dataset_cfg: Dict[str, Any],
    global_cfg: Dict[str, Any],
    paths_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Ingest a Parquet dataset based strictly on configuration
    provided in config.yaml.

    Parameters
    ----------
    dataset_name : str
        Logical dataset identifier (used for logging and error context)
    dataset_cfg : dict
        Dataset-specific ingestion configuration
    global_cfg : dict
        Global ingestion configuration (unused but required by contract)
    paths_cfg : dict
        Resolved project paths configuration

    Returns
    -------
    pandas.DataFrame
        Raw DataFrame loaded from the Parquet file

    Raises
    ------
    ValueError
        If required configuration keys are missing
    FileNotFoundError
        If the Parquet file does not exist
    TypeError
        If ingestion output violates the expected contract
    """

    # --------------------------------------------------
    # Resolve file path
    # --------------------------------------------------
    if "file" not in dataset_cfg:
        raise ValueError(
            f"[PARQUET INGESTION] Missing 'file' for dataset '{dataset_name}'"
        )

    raw_dir = Path(paths_cfg["data"]["raw"])
    file_path = raw_dir / dataset_cfg["file"]

    if not file_path.exists():
        raise FileNotFoundError(
            f"[PARQUET INGESTION] File not found for dataset '{dataset_name}': {file_path}"
        )

    parquet_cfg = dataset_cfg.get("parquet", {})
    engine = parquet_cfg.get("engine")

    logger.info(
        f"[PARQUET INGESTION] Reading dataset '{dataset_name}' from {file_path}"
    )

    # --------------------------------------------------
    # Read Parquet
    # --------------------------------------------------
    df = pd.read_parquet(
        file_path,
        engine=engine,
    )

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"[PARQUET INGESTION] Expected pandas DataFrame for dataset "
            f"'{dataset_name}', got {type(df).__name__}"
        )

    logger.info(
        f"[PARQUET INGESTION] Successfully loaded dataset '{dataset_name}' "
        f"with shape {df.shape}"
    )

    return df
