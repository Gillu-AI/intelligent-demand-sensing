# src/utils/logger.py

"""
Centralized Logging Configuration
==================================

Provides:
- Console logging
- Optional file logging
- Config-driven log level
- Singleton logger instance

Design Principles:
------------------
- No pipeline-specific logic
- Fully reusable across projects
- Prevent duplicate handlers
- Safe for CLI / scripts / notebooks
- No side effects on root logger

Logger Name:
------------
"IDS" (project namespace)

All pipelines and modules must use this logger.
"""

import os
import logging
from typing import Dict


def get_logger(config: Dict) -> logging.Logger:
    """
    Create and configure a project-wide logger.

    This function ensures:
    - Logger is initialized only once
    - Duplicate handlers are avoided
    - Log level is validated
    - Optional file logging is respected

    Parameters
    ----------
    config : Dict
        Loaded configuration dictionary.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    logger_name = "IDS"
    logger = logging.getLogger(logger_name)

    # Prevent reconfiguration if already initialized
    if logger.handlers:
        return logger

    # --------------------------------------------------
    # Validate Log Level
    # --------------------------------------------------

    log_level_str = config["logging"]["level"].upper()

    valid_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    if log_level_str not in valid_levels:
        raise ValueError(
            f"Invalid log level '{log_level_str}'. "
            f"Valid options: {list(valid_levels.keys())}"
        )

    logger.setLevel(valid_levels[log_level_str])
    logger.propagate = False  # Prevent duplicate logging to root logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # --------------------------------------------------
    # Console Handler
    # --------------------------------------------------

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --------------------------------------------------
    # File Handler
    # --------------------------------------------------

    if config["logging"]["log_to_file"]:

        log_dir = config["paths"]["logs"]
        os.makedirs(log_dir, exist_ok=True)

        file_path = os.path.join(
            log_dir,
            config["logging"]["filename"]
        )

        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
