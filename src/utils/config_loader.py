# src/utils/config_loader.py

"""
Centralized configuration loader for the IDS project.

Responsibilities:
- Load YAML configuration
- Validate mandatory sections
- Enforce defensive checks
- Validate LLM enterprise extensions
- Provide a single, safe config object

This module is intentionally generic and reusable across projects.

Design Principles:
------------------
- Fail-fast validation
- Defensive type checking
- Backward compatible
- No silent defaults
- No implicit assumptions
"""

from pathlib import Path
from typing import Dict, Any
from types import MappingProxyType
import yaml
import logging


logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when configuration validation fails."""
    pass


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)

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

    if config is None:
        raise ConfigError(
            "Configuration file is empty or contains no valid YAML content."
        )

    if not isinstance(config, dict):
        raise ConfigError(
            "Top-level configuration must be a dictionary."
        )

    _validate_required_sections(config)

    logger.info("Configuration loaded and validated successfully.")

    return dict(config)

def _freeze(obj):
    if isinstance(obj, dict):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_freeze(v) for v in obj)
    return obj


def _validate_required_sections(config: Dict[str, Any]) -> None:
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
        "forecasting",
        "llm",
        "shap",
        "visualization",
        "execution",
    }

    missing = required_sections - config.keys()
    if missing:
        raise ConfigError(
            f"Missing required config sections: {sorted(missing)}"
        )

    extra_sections = set(config.keys()) - required_sections
    if extra_sections:
        raise ConfigError(
            f"Unknown top-level config sections detected: {sorted(extra_sections)}"
        )

    for section in required_sections:
        if not isinstance(config.get(section), dict):
            raise ConfigError(
                f"Config section '{section}' must be a dictionary."
            )

    _validate_logging(config)
    _validate_seeds(config)
    _validate_execution(config)
    _validate_llm_section(config)


def _validate_logging(config: Dict[str, Any]) -> None:
    logging_cfg = config["logging"]

    required = {"level", "log_to_file", "filename"}
    missing = required - logging_cfg.keys()
    if missing:
        raise ConfigError(
            f"Missing required logging config keys: {sorted(missing)}"
        )

    if not isinstance(logging_cfg["log_to_file"], bool):
        raise ConfigError("logging.log_to_file must be boolean.")


def _validate_seeds(config: Dict[str, Any]) -> None:
    seeds_cfg = config["seeds"]

    if "global_seed" not in seeds_cfg:
        raise ConfigError("Missing 'seeds.global_seed' configuration.")

    if not isinstance(seeds_cfg["global_seed"], int):
        raise ConfigError("seeds.global_seed must be an integer.")

    if "propagate_to_models" in seeds_cfg and not isinstance(
        seeds_cfg["propagate_to_models"], bool
    ):
        raise ConfigError("seeds.propagate_to_models must be boolean.")


def _validate_execution(config: Dict[str, Any]) -> None:
    execution_cfg = config["execution"]

    mode = execution_cfg.get("mode")
    allowed_modes = {"dev", "train", "prod", "backfill"}

    if not isinstance(mode, str):
        raise ConfigError("Execution mode must be a string.")

    if mode not in allowed_modes:
        raise ConfigError(
            f"Invalid execution mode '{mode}'. "
            f"Allowed values are: {sorted(allowed_modes)}"
        )

    if "save_intermediate_outputs" in execution_cfg and not isinstance(
        execution_cfg["save_intermediate_outputs"], bool
    ):
        raise ConfigError(
            "execution.save_intermediate_outputs must be boolean."
        )

    if "overwrite_existing_outputs" in execution_cfg and not isinstance(
        execution_cfg["overwrite_existing_outputs"], bool
    ):
        raise ConfigError(
            "execution.overwrite_existing_outputs must be boolean."
        )


def _validate_llm_section(config: Dict[str, Any]) -> None:
    llm_cfg = config["llm"]

    if not isinstance(llm_cfg.get("enabled"), bool):
        raise ConfigError("llm.enabled must be boolean.")

    if not isinstance(llm_cfg.get("storage_path"), str):
        raise ConfigError("llm.storage_path must be string.")

    ttl = llm_cfg.get("scenario_ttl_hours")
    if not isinstance(ttl, int) or ttl <= 0:
        raise ConfigError("llm.scenario_ttl_hours must be positive integer.")

    if "temperature" in llm_cfg and not isinstance(
        llm_cfg["temperature"], (int, float)
    ):
        raise ConfigError("llm.temperature must be numeric.")

    if "max_tokens" in llm_cfg and not isinstance(
        llm_cfg["max_tokens"], int
    ):
        raise ConfigError("llm.max_tokens must be integer.")

    session_cfg = llm_cfg.get("session", {})
    if session_cfg:
        if "auto_create_session" in session_cfg and not isinstance(
            session_cfg["auto_create_session"], bool
        ):
            raise ConfigError(
                "llm.session.auto_create_session must be boolean."
            )

        if "include_timestamp_in_filename" in session_cfg and not isinstance(
            session_cfg["include_timestamp_in_filename"], bool
        ):
            raise ConfigError(
                "llm.session.include_timestamp_in_filename must be boolean."
            )

    guardrails = llm_cfg.get("guardrails", {})
    if guardrails:
        max_pct = guardrails.get("max_demand_change_pct")
        if not isinstance(max_pct, (int, float)):
            raise ConfigError(
                "llm.guardrails.max_demand_change_pct must be numeric."
            )

        max_lt = guardrails.get("max_lead_time_days")
        if not isinstance(max_lt, int) or max_lt <= 0:
            raise ConfigError(
                "llm.guardrails.max_lead_time_days must be positive integer."
            )

        service_range = guardrails.get("service_level_range", {})
        if service_range:
            min_val = service_range.get("min")
            max_val = service_range.get("max")

            if not (0 < min_val < 1):
                raise ConfigError(
                    "llm.guardrails.service_level_range.min must be between 0 and 1."
                )

            if not (0 < max_val < 1):
                raise ConfigError(
                    "llm.guardrails.service_level_range.max must be between 0 and 1."
                )

            if min_val >= max_val:
                raise ConfigError(
                    "service_level_range.min must be less than max."
                )
