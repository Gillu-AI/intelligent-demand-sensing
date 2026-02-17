# src/utils/schema_utils.py
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
def apply_rename_map(
    columns: Iterable[str],
    rename_map: Dict[str, List[str]]
) -> List[str]:
    """
    Apply flexible rename mapping to column names.

    This function does NOT perform duplicate detection.
    Duplicate detection must be done explicitly after this step.
    """

    reverse_map = {}
    seen_aliases = set()

    for canonical, aliases in rename_map.items():

        if not isinstance(canonical, str):
            raise SchemaValidationError(
                f"Canonical column name must be string. Found: {type(canonical).__name__}"
            )

        if not isinstance(aliases, (list, tuple)):
            raise SchemaValidationError(
                f"Rename map for '{canonical}' must be a list of aliases."
            )

        for alias in aliases:

            if not isinstance(alias, str):
                raise SchemaValidationError(
                    f"Alias for '{canonical}' must be string. Found: {type(alias).__name__}"
                )

            if alias in seen_aliases:
                raise SchemaValidationError(
                    f"Duplicate alias detected in rename_map: '{alias}'"
                )

            reverse_map[alias] = canonical
            seen_aliases.add(alias)

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
    columns: Iterable[str],
    required_columns: Iterable[str],
    optional_columns: Iterable[str] | None = None,
    optional_policy: str = "ignore"
) -> None:
    """
    Validate presence of required and optional columns according
    to configured schema policy.
    """

    available = set(columns)
    required = set(required_columns)

    missing_required = required - available
    if missing_required:
        raise SchemaValidationError(
            f"Missing required columns: {sorted(missing_required)}"
        )

    logger.debug("All required columns are present.")

    allowed_policies = {"ignore", "warn", "error"}

    if optional_policy not in allowed_policies:
        raise SchemaValidationError(
            f"Invalid optional_policy '{optional_policy}'. "
            f"Allowed values are: {sorted(allowed_policies)}"
        )

    if optional_columns is not None:

        optional = set(optional_columns)
        missing_optional = optional - available

        if missing_optional:

            if optional_policy == "error":
                raise SchemaValidationError(
                    f"Missing optional columns (policy=error): "
                    f"{sorted(missing_optional)}"
                )

            elif optional_policy == "warn":
                logger.warning(
                    "Missing optional columns (policy=warn): %s",
                    sorted(missing_optional)
                )
