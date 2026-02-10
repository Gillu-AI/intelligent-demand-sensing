"""
API ingestion module for the IDS project.

ROLE
----
Source-specific ingestion layer responsible for loading data
from HTTP APIs in a strictly config-driven manner.

RESPONSIBILITIES
----------------
- Validate API configuration
- Execute HTTP requests
- Handle optional pagination
- Resolve environment variables in headers
- Return raw pandas DataFrame
- Emit production-grade logs

WHAT THIS MODULE DOES NOT DO
----------------------------
- Does NOT apply schema validation
- Does NOT perform feature engineering
- Does NOT assume defaults
- Does NOT modify response payloads

DESIGN PRINCIPLES
-----------------
- Config-driven only
- Fail fast with actionable errors
- Safe for batch and production pipelines
"""

from typing import Dict, Any
import os
import re
import logging

import requests
import pandas as pd


logger = logging.getLogger(__name__)


_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


def _resolve_env_vars(value: str) -> str:
    """Resolve ${VAR} from environment variables."""
    def replacer(match):
        var = match.group(1)
        if var not in os.environ:
            raise EnvironmentError(
                f"[API INGESTION] Environment variable '{var}' not set"
            )
        return os.environ[var]

    return _ENV_PATTERN.sub(replacer, value)


def ingest(
    dataset_name: str,
    dataset_cfg: Dict[str, Any],
    global_cfg: Dict[str, Any],
    paths_cfg: Dict[str, Any],  # required by frozen contract
) -> pd.DataFrame:
    """
    Ingest a dataset from an HTTP API.

    Parameters
    ----------
    dataset_name : str
        Logical dataset identifier
    dataset_cfg : dict
        Dataset-specific configuration
    global_cfg : dict
        Global ingestion configuration (unused, contract-required)
    paths_cfg : dict
        Paths configuration (unused, contract-required)

    Returns
    -------
    pandas.DataFrame
        Raw API response data

    Raises
    ------
    ValueError
        If configuration is invalid
    EnvironmentError
        If required environment variables are missing
    TypeError
        If API response cannot be converted to DataFrame
    """

    api_cfg = dataset_cfg.get("api")
    if not isinstance(api_cfg, dict):
        raise ValueError(
            f"[API INGESTION] Missing or invalid 'api' config for '{dataset_name}'"
        )

    # --------------------------------------------------
    # Validate required API config
    # --------------------------------------------------
    required_keys = {"method", "url", "timeout_seconds"}
    missing = required_keys - api_cfg.keys()

    if missing:
        raise ValueError(
            f"[API INGESTION] Missing API config keys for '{dataset_name}': {sorted(missing)}"
        )

    method = api_cfg["method"].upper()
    url = api_cfg["url"]
    timeout = api_cfg["timeout_seconds"]

    if not isinstance(url, str):
        raise ValueError(
            f"[API INGESTION] 'url' must be a string for '{dataset_name}'"
        )

    # --------------------------------------------------
    # Resolve headers and params
    # --------------------------------------------------
    headers = {}
    for k, v in api_cfg.get("headers", {}).items():
        headers[k] = _resolve_env_vars(v) if isinstance(v, str) else v

    params = api_cfg.get("params", {})

    logger.info(
        f"[API INGESTION] Starting API ingestion for dataset '{dataset_name}'"
    )

    records = []

    # --------------------------------------------------
    # Optional pagination
    # --------------------------------------------------
    pagination_cfg = api_cfg.get("pagination", {})
    pagination_enabled = pagination_cfg.get("enabled", False)

    page = 1
    while True:
        request_params = dict(params)

        if pagination_enabled:
            request_params[pagination_cfg["page_param"]] = page
            request_params[pagination_cfg["size_param"]] = pagination_cfg["page_size"]

        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            params=request_params,
            timeout=timeout,
        )

        response.raise_for_status()
        payload = response.json()

        if isinstance(payload, dict):
            payload = payload.get("data")

        if not isinstance(payload, list):
            raise TypeError(
                f"[API INGESTION] Expected list-like response for '{dataset_name}'"
            )

        if not payload:
            break

        records.extend(payload)

        logger.info(
            f"[API INGESTION] Retrieved page {page} with {len(payload)} records"
        )

        if not pagination_enabled:
            break

        page += 1

    if not records:
        raise ValueError(
            f"[API INGESTION] API returned no data for '{dataset_name}'"
        )

    df = pd.DataFrame(records)

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"[API INGESTION] Expected pandas DataFrame for '{dataset_name}', "
            f"got {type(df).__name__}"
        )

    logger.info(
        f"[API INGESTION] Successfully loaded dataset '{dataset_name}' "
        f"with shape {df.shape}"
    )

    return df
