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

    if not path.exists():
        raise ConfigError(
            f"Configuration file not found at path: {path.resolve()}"
        )

    if path.suffix not in {".yaml", ".yml"}:
        raise ConfigError(
            f"Invalid config file format: {path.name}. Expected a YAML file."
        )

    try:
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Failed to parse YAML configuration: {exc}"
        ) from exc

    if not isinstance(config, dict):
        raise ConfigError(
            "Top-level configuration must be a dictionary."
        )

    _validate_required_sections(config)

    logger.info("Configuration loaded and validated successfully.")

    return config


def _validate_required_sections(config: Dict[str, Any]) -> None:
    """
    Validate presence of mandatory top-level sections.

    This enforces architectural contracts and prevents
    silent misconfiguration.
    """

    required_sections = {
        "project",
        "paths",
        "logging",
        "seeds",
        "data_schema",
        "features",
        "models",
        "ensemble",
        "inventory",
        "llm",
        "execution"
    }

    missing = required_sections - config.keys()

    if missing:
        raise ConfigError(
            f"Missing required config sections: {sorted(missing)}"
        )
