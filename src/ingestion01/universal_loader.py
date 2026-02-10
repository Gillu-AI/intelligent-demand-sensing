"""
universal_loader.py
===================

Central orchestration layer for dataset ingestion in the IDS project.

ROLE
----
This module is responsible for coordinating ingestion across multiple
data sources in a uniform, config-driven, and production-safe manner.

RESPONSIBILITIES
----------------
- Read ingestion configuration from config.yaml
- Validate critical ingestion preconditions
- Identify enabled datasets
- Dispatch ingestion to the correct source-specific loader
- Apply schema normalization and validation via schema_utils
- Enforce strict vs non-strict ingestion behavior
- Ensure demand anchor dataset is successfully loaded
- Return all ingested datasets as a dictionary

WHAT THIS MODULE DOES
---------------------
- Iterates over ingestion.datasets
- Calls csv / excel / parquet / database / api ingestion modules
- Applies schema enforcement in a fixed, deterministic order
- Aggregates outputs keyed by dataset name

WHAT THIS MODULE DOES NOT DO
----------------------------
- Does NOT read files directly
- Does NOT implement source-specific ingestion logic
- Does NOT perform feature engineering
- Does NOT modify schema rules

SCHEMA ENFORCEMENT ORDER (PER DATASET)
-------------------------------------
1. normalize_column_names
2. apply_rename_map
3. detect_duplicate_columns
4. validate_required_columns

DESIGN CONTRACT
-----------------------
Each ingestion module MUST expose:

    ingest(
        dataset_name: str,
        dataset_cfg: dict,
        global_cfg: dict,
        paths_cfg: dict
    ) -> pandas.DataFrame | dict[str, pandas.DataFrame]
"""

from typing import Dict, Any
import logging

from ingestion01 import (
    csv_ingestion,
    excel_ingestion,
    parquet_ingestion,
    database_ingestion,
    api_ingestion,
)

from utils.schema_utils import (
    normalize_column_names,
    apply_rename_map,
    detect_duplicate_columns,
    validate_required_columns,
    SchemaValidationError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Source dispatch table (FROZEN)
# ---------------------------------------------------------------------
SOURCE_DISPATCH = {
    "csv": csv_ingestion,
    "excel": excel_ingestion,
    "parquet": parquet_ingestion,
    "database": database_ingestion,
    "api": api_ingestion,
}


# ---------------------------------------------------------------------
# Universal Loader
# ---------------------------------------------------------------------
def load_all_datasets(
    config: Dict[str, Any],
    paths_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Load all enabled datasets defined in ingestion configuration.

    This function enforces:
    - Correct ingestion configuration
    - Strict vs non-strict error behavior
    - Schema normalization and validation
    - Mandatory demand anchor dataset presence

    Parameters
    ----------
    config : dict
        Full application configuration loaded from config.yaml
    paths_cfg : dict
        Resolved paths configuration

    Returns
    -------
    Dict[str, Any]
        Dictionary of ingested datasets keyed by dataset name

    Raises
    ------
    ValueError
        If ingestion configuration is invalid
    RuntimeError
        If demand anchor dataset is not loaded successfully
    """

    ingestion_cfg = config["ingestion"]
    datasets_cfg = ingestion_cfg["datasets"]
    schema_cfg = config.get("data_schema", {})

    strict_load = ingestion_cfg.get("strict_load", True)

    # -------------------------------------------------
    # Demand anchor precondition validation (INTENT)
    # -------------------------------------------------
    demand_source = ingestion_cfg.get("demand_source")
    if not isinstance(demand_source, str):
        raise ValueError(
            "ingestion.demand_source must be a string identifying a dataset name."
        )

    results: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    # -------------------------------------------------
    # Dataset ingestion loop
    # -------------------------------------------------
    for dataset_name, dataset_cfg in datasets_cfg.items():

        if not dataset_cfg.get("enabled", False):
            logger.info(f"Skipping disabled dataset: {dataset_name}")
            continue

        source_type = dataset_cfg.get("source_type")

        if source_type not in SOURCE_DISPATCH:
            raise ValueError(
                f"Unsupported source_type '{source_type}' "
                f"for dataset '{dataset_name}'"
            )

        logger.info(
            f"Loading dataset '{dataset_name}' "
            f"from source '{source_type}'"
        )

        try:
            loader_module = SOURCE_DISPATCH[source_type]

            data = loader_module.ingest(
                dataset_name=dataset_name,
                dataset_cfg=dataset_cfg,
                global_cfg=ingestion_cfg,
                paths_cfg=paths_cfg,
            )

            # -------------------------------------------------
            # Schema enforcement (GLOBAL, FROZEN)
            # -------------------------------------------------
            if schema_cfg:

                # Multi-sheet case (Excel)
                if isinstance(data, dict):
                    processed = {}
                    for key, df in data.items():
                        processed[key] = _apply_schema(
                            df=df,
                            schema_cfg=schema_cfg,
                            dataset_name=f"{dataset_name}.{key}",
                        )
                    data = processed

                # Single DataFrame
                else:
                    data = _apply_schema(
                        df=data,
                        schema_cfg=schema_cfg,
                        dataset_name=dataset_name,
                    )

            results[dataset_name] = data
            logger.info(f"Dataset '{dataset_name}' loaded successfully")

        except SchemaValidationError as exc:
            logger.error(
                f"Schema validation failed for dataset '{dataset_name}': {exc}"
            )
            if strict_load:
                raise
            errors[dataset_name] = f"SchemaError: {exc}"

        except Exception as exc:
            logger.exception(
                f"Failed to load dataset '{dataset_name}': {exc}"
            )
            if strict_load:
                raise
            errors[dataset_name] = str(exc)

    # -------------------------------------------------
    # Demand anchor postcondition validation (OUTCOME)
    # -------------------------------------------------
    if demand_source not in results:
        raise RuntimeError(
            f"Demand source dataset '{demand_source}' was not loaded successfully."
        )

    if errors:
        logger.warning(f"Ingestion completed with errors: {errors}")

    return results


# ---------------------------------------------------------------------
# Schema Application Helper (PRIVATE)
# ---------------------------------------------------------------------
def _apply_schema(
    df,
    schema_cfg: Dict[str, Any],
    dataset_name: str,
):
    """
    Apply schema normalization and validation to a single DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw ingested DataFrame
    schema_cfg : dict
        Global schema configuration
    dataset_name : str
        Dataset identifier (used for logging and error context)

    Returns
    -------
    pandas.DataFrame
        Schema-normalized and validated DataFrame

    Raises
    ------
    SchemaValidationError
        If schema validation fails
    """

    logger.info(f"Applying schema to dataset '{dataset_name}'")

    # 1. Normalize column names
    df.columns = normalize_column_names(df.columns)

    # 2. Apply rename map
    rename_map = schema_cfg.get("rename_map", {})
    df.columns = apply_rename_map(df.columns, rename_map)

    # 3. Detect duplicates
    detect_duplicate_columns(df.columns)

    # 4. Validate required & optional columns
    required_cols = schema_cfg.get("required_columns", [])
    optional_cols = schema_cfg.get("optional_columns", [])
    optional_policy = schema_cfg.get("optional_column_policy", {}).get(
        "on_missing", "ignore"
    )

    validate_required_columns(
        columns=df.columns,
        required_columns=required_cols,
        optional_columns=optional_cols,
        optional_policy=optional_policy,
    )

    return df
