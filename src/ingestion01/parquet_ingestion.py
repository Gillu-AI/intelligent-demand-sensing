# src/ingestion01/parquet_ingestion.py

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

    # --------------------------------------------------
    # Resolve file path
    # --------------------------------------------------
    file_name = dataset_cfg.get("file")

    if not isinstance(file_name, str) or not file_name:
        raise ValueError(
            f"[PARQUET INGESTION] 'file' must be a non-empty string "
            f"for dataset '{dataset_name}'"
        )

    raw_dir = Path(paths_cfg["data"]["raw"])
    file_path = raw_dir / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"[PARQUET INGESTION] File not found for dataset "
            f"'{dataset_name}': {file_path}"
        )

    # --------------------------------------------------
    # Parquet-specific config (MANDATORY)
    # --------------------------------------------------
    if "parquet" not in dataset_cfg:
        raise ValueError(
            f"[PARQUET INGESTION] Missing 'parquet' config block "
            f"for dataset '{dataset_name}'"
        )

    parquet_cfg = dataset_cfg["parquet"]

    if "engine" not in parquet_cfg:
        raise ValueError(
            f"[PARQUET INGESTION] Missing 'engine' in parquet config "
            f"for dataset '{dataset_name}'"
        )

    engine = parquet_cfg["engine"]

    if not isinstance(engine, str) or not engine:
        raise ValueError(
            f"[PARQUET INGESTION] 'engine' must be a non-empty string "
            f"for dataset '{dataset_name}'"
        )

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
        f"[PARQUET INGESTION] Successfully loaded dataset "
            f"'{dataset_name}' with shape {df.shape}"
    )

    return df