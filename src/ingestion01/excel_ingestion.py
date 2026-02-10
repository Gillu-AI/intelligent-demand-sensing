"""
Excel ingestion module for the IDS project.

ROLE
----
Source-specific ingestion layer responsible for reading Excel files
in a strictly config-driven, production-safe manner.

RESPONSIBILITIES
----------------
- Resolve Excel file paths from project configuration
- Support exactly ONE ingestion mode per dataset:
    * Single sheet (`sheet_name`)
    * Multiple selected sheets (`sheet_names`)
    * All sheets (`load_all_sheets: true`)
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
from typing import Dict, Union
import logging

import pandas as pd


logger = logging.getLogger(__name__)


def ingest(
    dataset_name: str,
    dataset_cfg: Dict,
    global_cfg: Dict,
    paths_cfg: Dict,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Ingest an Excel dataset based strictly on configuration provided
    in config.yaml.

    Exactly ONE sheet selection mode must be configured:
    - `sheet_name`: load a single sheet
    - `sheet_names`: load selected multiple sheets
    - `load_all_sheets: true`: load all sheets

    Parameters
    ----------
    dataset_name : str
        Logical dataset identifier (used for logging and error context)
    dataset_cfg : dict
        Dataset-specific ingestion configuration
    global_cfg : dict
        Global ingestion configuration (header, skip_rows, na_values)
    paths_cfg : dict
        Resolved project paths configuration

    Returns
    -------
    pandas.DataFrame
        When a single sheet is loaded
    Dict[str, pandas.DataFrame]
        When multiple sheets are loaded

    Raises
    ------
    ValueError
        If ingestion configuration is invalid
    FileNotFoundError
        If the Excel file does not exist
    TypeError
        If the returned object violates the ingestion contract
    """

    # --------------------------------------------------
    # Resolve file path
    # --------------------------------------------------
    raw_dir = Path(paths_cfg["data"]["raw"])
    file_name = dataset_cfg.get("file")

    if not file_name:
        raise ValueError(
            f"[EXCEL INGESTION] Missing 'file' for dataset '{dataset_name}'"
        )

    file_path = raw_dir / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"[EXCEL INGESTION] File not found for dataset '{dataset_name}': {file_path}"
        )

    # --------------------------------------------------
    # Global ingestion config (MANDATORY)
    # --------------------------------------------------
    header_flag = global_cfg["header"]
    skip_rows = global_cfg["skip_rows"]
    na_values = global_cfg["na_values"]

    header = 0 if header_flag else None

    # --------------------------------------------------
    # Excel-specific config (MANDATORY)
    # --------------------------------------------------
    if "excel" not in dataset_cfg:
        raise ValueError(
            f"[EXCEL INGESTION] Missing 'excel' config block for dataset '{dataset_name}'"
        )

    excel_cfg = dataset_cfg["excel"]

    has_sheet_name = "sheet_name" in excel_cfg
    has_sheet_names = "sheet_names" in excel_cfg
    has_load_all = excel_cfg.get("load_all_sheets", False) is True

    mode_count = sum([has_sheet_name, has_sheet_names, has_load_all])

    if mode_count == 0:
        raise ValueError(
            f"[EXCEL INGESTION] No sheet selection defined for dataset '{dataset_name}'. "
            f"Specify ONE of: sheet_name | sheet_names | load_all_sheets"
        )

    if mode_count > 1:
        raise ValueError(
            f"[EXCEL INGESTION] Multiple sheet selection modes defined for dataset "
            f"'{dataset_name}'. Only ONE is allowed."
        )

    logger.info(
        f"[EXCEL INGESTION] Reading dataset '{dataset_name}' from {file_path}"
    )

    # --------------------------------------------------
    # Read Excel — single sheet
    # --------------------------------------------------
    if has_sheet_name:
        sheet = excel_cfg["sheet_name"]

        df = pd.read_excel(
            file_path,
            sheet_name=sheet,
            header=header,
            skiprows=skip_rows,
            na_values=na_values,
        )

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"[EXCEL INGESTION] Expected pandas DataFrame for dataset "
                f"'{dataset_name}', got {type(df).__name__}"
            )

        logger.info(
            f"[EXCEL INGESTION] Successfully loaded dataset '{dataset_name}' "
            f"(mode=single_sheet, sheet='{sheet}', shape={df.shape})"
        )

        return df

    # --------------------------------------------------
    # Read Excel — selected multiple sheets
    # --------------------------------------------------
    if has_sheet_names:
        sheets = excel_cfg["sheet_names"]

        if not isinstance(sheets, list) or not sheets:
            raise ValueError(
                f"[EXCEL INGESTION] 'sheet_names' must be a non-empty list "
                f"for dataset '{dataset_name}'"
            )

        dfs = pd.read_excel(
            file_path,
            sheet_name=sheets,
            header=header,
            skiprows=skip_rows,
            na_values=na_values,
        )

        if not isinstance(dfs, dict) or not all(
            isinstance(v, pd.DataFrame) for v in dfs.values()
        ):
            raise TypeError(
                f"[EXCEL INGESTION] Expected dict[str, DataFrame] for dataset "
                f"'{dataset_name}', got {type(dfs).__name__}"
            )

        logger.info(
            f"[EXCEL INGESTION] Successfully loaded dataset '{dataset_name}' "
            f"(mode=multi_sheet, sheets={list(dfs.keys())})"
        )

        return dfs

    # --------------------------------------------------
    # Read Excel — all sheets
    # --------------------------------------------------
    dfs = pd.read_excel(
        file_path,
        sheet_name=None,
        header=header,
        skiprows=skip_rows,
        na_values=na_values,
    )

    if not isinstance(dfs, dict) or not all(
        isinstance(v, pd.DataFrame) for v in dfs.values()
    ):
        raise TypeError(
            f"[EXCEL INGESTION] Expected dict[str, DataFrame] for dataset "
            f"'{dataset_name}', got {type(dfs).__name__}"
        )

    logger.info(
        f"[EXCEL INGESTION] Successfully loaded dataset '{dataset_name}' "
        f"(mode=all_sheets, sheets={list(dfs.keys())})"
    )

    return dfs
