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

        normalized.append(col_clean)

    logger.debug("Column names normalized.")

    return normalized


# ---------------------------------------------------------------------
# Rename Mapping
# ---------------------------------------------------------------------
def apply_rename_map(
    columns: Iterable[str],
    rename_map: Dict[str, str]
) -> List[str]:
    """
    Apply flexible rename mapping to column names.

    This function does NOT perform duplicate detection.
    Duplicate detection must be done explicitly after this step.

    Parameters
    ----------
    columns : Iterable[str]
        Normalized column names
    rename_map : Dict[str, str]
        Mapping of alternative names to canonical names

    Returns
    -------
    List[str]
        Column names after applying rename rules
    """

    renamed = []

    for col in columns:
        # Safe fallback ensures pipeline does not fail
        # if a column is not present in rename_map
        new_col = rename_map.get(col, col)
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
# Required Column Validation
# ---------------------------------------------------------------------
def validate_required_columns(
    columns: Iterable[str],
    required_columns: Iterable[str]
) -> None:
    """
    Validate presence of required columns.

    Parameters
    ----------
    columns : Iterable[str]
        Available column names
    required_columns : Iterable[str]
        Columns that must be present

    Raises
    ------
    SchemaValidationError
        If any required column is missing
    """

    available = set(columns)
    required = set(required_columns)

    missing = required - available

    if missing:
        raise SchemaValidationError(
            f"Missing required columns: {sorted(missing)}"
        )

    logger.debug("All required columns are present.")
