"""
Centralized configuration loader for the IDS project.

Responsibilities:
- Load YAML configuration
- Validate mandatory sections
- Enforce defensive checks
- Provide a single, safe config object

This module is intentionally generic and reusable across projects.
"""

from pathlib import Path
from typing import Dict, Any
import yaml
import logging


logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when configuration validation fails."""
    pass


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate the IDS configuration file.

    Parameters
    ----------
    config_path : str
        Path to the config.yaml file

    Returns
    -------
    Dict[str, Any]
        Validated configuration dictionary

    Raises
    ------
    ConfigError
        If the configuration file is missing, malformed,
        or fails validation checks
    """

    path = Path(config_path)

    # --- Path validation ---
    if not path.exists():
        raise ConfigError(
            f"Configuration file not found at path: {path.resolve()}"
        )

    if not path.is_file():
        raise ConfigError(
            f"Configuration path is not a file: {path.resolve()}"
        )

    if path.suffix not in {".yaml", ".yml"}:
        raise ConfigError(
            f"Invalid config file format: {path.name}. Expected a YAML file."
        )

    # --- YAML loading ---
    try:
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except (OSError, PermissionError) as exc:
        raise ConfigError(
            f"Failed to read configuration file: {path.resolve()} ({exc})"
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Failed to parse YAML configuration: {exc}"
        ) from exc

    # --- Empty / malformed config ---
    if config is None:
        raise ConfigError(
            "Configuration file is empty or contains no valid YAML content."
        )

    if not isinstance(config, dict):
        raise ConfigError(
            "Top-level configuration must be a dictionary."
        )

    # --- Structural validation ---
    _validate_required_sections(config)

    logger.info("Configuration loaded and validated successfully.")

    # Defensive copy (top-level)
    return dict(config)


def _validate_required_sections(config: Dict[str, Any]) -> None:
    """
    Validate presence and structure of mandatory top-level sections.

    This enforces architectural contracts and prevents
    silent misconfiguration.
    """

    # --- Required top-level sections ---
    required_sections = {
        "project",
        "paths",
        "logging",
        "seeds",
        "data_schema",
        "ingestion",
        "features",
        "data_cleaning",
        "modeling",
        "ensemble",
        "hyperparameter_tuning",
        "inventory",
        "llm",
        "execution"
    }

    # --- Missing sections ---
    missing = required_sections - config.keys()
    if missing:
        raise ConfigError(
            f"Missing required config sections: {sorted(missing)}"
        )

    # --- Unknown sections ---
    extra_sections = set(config.keys()) - required_sections
    if extra_sections:
        raise ConfigError(
            f"Unknown top-level config sections found: {sorted(extra_sections)}"
        )

    # --- Section type validation ---
    for section in required_sections:
        if not isinstance(config.get(section), dict):
            raise ConfigError(
                f"Config section '{section}' must be a dictionary."
            )

    # =========================================================
    # LOGGING STRUCTURE VALIDATION
    # =========================================================
    logging_cfg = config.get("logging", {})

    required_logging_keys = {"level", "log_to_file", "filename"}
    missing_logging = required_logging_keys - logging_cfg.keys()

    if missing_logging:
        raise ConfigError(
            f"Missing required logging config keys: {sorted(missing_logging)}"
        )
    # =========================================================
    # EXECUTION MODE VALIDATION
    # =========================================================
    execution_cfg = config.get("execution", {})
    mode = execution_cfg.get("mode")

    allowed_modes = {"dev", "train", "prod", "backfill"}

    if not isinstance(mode, str):
        raise ConfigError(
            "Execution mode must be a string."
        )

    if mode not in allowed_modes:
        raise ConfigError(
            f"Invalid execution mode '{mode}'. "
            f"Allowed values are: {sorted(allowed_modes)}"
        )