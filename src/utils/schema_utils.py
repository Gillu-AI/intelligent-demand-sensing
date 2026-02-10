"""
Schema validation and normalization utilities for IDS.

Responsibilities:
- Enforce column naming standards
- Apply semantic rename mappings
- Detect schema integrity violations
- Validate required schema elements
- Fail fast with clear, actionable errors

Execution order (enforced by pipeline, not this module):
    1. normalize_column_names
    2. apply_rename_map
    3. detect_duplicate_columns
    4. validate_required_columns

This module is intentionally generic and reusable
across data-driven ML projects.
"""

from typing import Iterable, List, Dict
import logging
import re


logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """
    Raised when input data violates expected schema rules.

    This is a semantic exception used by schema utilities
    and should be handled by pipeline-level code.
    """
    pass


# ---------------------------------------------------------------------
# Column Normalization
# ---------------------------------------------------------------------
def normalize_column_names(columns: Iterable[str]) -> List[str]:
    """
    Normalize column names to snake_case.

    Rules applied:
    - Lowercase all characters
    - Replace spaces and hyphens with underscores
    - Remove non-alphanumeric characters
    - Collapse multiple underscores

    Parameters
    ----------
    columns : Iterable[str]
        Original column names

    Returns
    -------
    List[str]
        Normalized column names
    """

    normalized = []

    for col in columns:
        if not isinstance(col, str):
            raise SchemaValidationError(
                f"Column name must be a string. Found type: {type(col).__name__}"
            )

        col_clean = col.strip().lower()
        col_clean = re.sub(r"[\s\-]+", "_", col_clean)
        col_clean = re.sub(r"[^a-z0-9_]", "", col_clean)
        col_clean = re.sub(r"_+", "_", col_clean)
        col_clean = col_clean.strip("_")

        normalized.append(col_clean)

    logger.debug("Column names normalized.")

    return normalized


# ---------------------------------------------------------------------
# Rename Mapping
# ---------------------------------------------------------------------
def apply_rename_map(columns: Iterable[str], rename_map: Dict[str, List[str]]) -> List[str]:
    """
    Apply flexible rename mapping to column names.

    This function does NOT perform duplicate detection.
    Duplicate detection must be done explicitly after this step.

    Parameters
    ----------
    columns : Iterable[str]
        Normalized column names
    rename_map : Dict[str, List[str]]
        Mapping of alternative names to canonical names

    Returns
    -------
    List[str]
        Column names after applying rename rules
    """

    reverse_map = {}

    for canonical, aliases in rename_map.items():
        if not isinstance(aliases, (list, tuple)):
            raise SchemaValidationError(
                f"Rename map for '{canonical}' must be a list of aliases."
            )
        for alias in aliases:
            reverse_map[alias] = canonical

    renamed = []

    for col in columns:
        new_col = reverse_map.get(col, col)
        renamed.append(new_col)

    logger.debug("Rename mapping applied to columns.")

    return renamed


# ---------------------------------------------------------------------
# Duplicate Detection
# ---------------------------------------------------------------------
def detect_duplicate_columns(columns: Iterable[str]) -> None:
    """
    Detect duplicate column names after normalization and renaming.

    Parameters
    ----------
    columns : Iterable[str]
        Final column names after all transformations

    Raises
    ------
    SchemaValidationError
        If duplicate column names are found
    """

    seen = set()
    duplicates = set()

    for col in columns:
        if col in seen:
            duplicates.add(col)
        else:
            seen.add(col)

    if duplicates:
        raise SchemaValidationError(
            f"Duplicate columns detected after schema processing: {sorted(duplicates)}"
        )

    logger.debug("No duplicate columns detected.")


# ---------------------------------------------------------------------
# Required & Optional Column Validation
# ---------------------------------------------------------------------
def validate_required_columns(
    columns: Iterable[str], required_columns: Iterable[str],
    optional_columns: Iterable[str] | None = None,
    optional_policy: str = "ignore"
) -> None:
    """
    Validate presence of required and optional columns according
    to configured schema policy.

    This function enforces:
    - All required columns must be present (hard failure)
    - Optional columns are handled based on policy:
        - "ignore": do nothing
        - "warn": log a warning
        - "error": raise an exception

    Parameters
    ----------
    columns : Iterable[str]
        Available column names after normalization and renaming
    required_columns : Iterable[str]
        Columns that must be present
    optional_columns : Iterable[str], optional
        Columns that are optional but semantically meaningful
    optional_policy : str, default "ignore"
        Policy for handling missing optional columns:
        one of {"ignore", "warn", "error"}

    Raises
    ------
    SchemaValidationError
        If required columns are missing, or optional columns
        are missing with policy="error"
    """

    available = set(columns)
    required = set(required_columns)

    # --- Required column enforcement ---
    missing_required = required - available
    if missing_required:
        raise SchemaValidationError(
            f"Missing required columns: {sorted(missing_required)}"
        )

    logger.debug("All required columns are present.")

    # --- Optional column policy enforcement ---
    allowed_policies = {"ignore", "warn", "error"}

    if optional_policy not in allowed_policies:
        raise SchemaValidationError(
            f"Invalid optional_policy '{optional_policy}'. "
            f"Allowed values are: {sorted(allowed_policies)}"
        )
    if optional_columns:
        optional = set(optional_columns)
        missing_optional = optional - available

        if missing_optional:
            if optional_policy == "error":
                raise SchemaValidationError(
                    f"Missing optional columns (policy=error): {sorted(missing_optional)}"
                )
            elif optional_policy == "warn":
                logger.warning(
                    "Missing optional columns (policy=warn): %s",
                    sorted(missing_optional)
                )
