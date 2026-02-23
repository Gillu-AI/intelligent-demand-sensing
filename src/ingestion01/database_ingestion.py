# src/ingestion01/database_ingestion.py

"""
Database ingestion module for the IDS project.

ROLE
----
Source-specific ingestion layer responsible for loading data
from relational databases in a strictly config-driven manner.

RESPONSIBILITIES
----------------
- Validate database configuration
- Build SQLAlchemy connection string
- Load data using table or custom query
- Handle chunked and non-chunked reads safely
- Emit structured logs for observability
- Return raw DataFrame without schema enforcement

WHAT THIS MODULE DOES NOT DO
----------------------------
- Does NOT apply schema validation
- Does NOT perform feature engineering
- Does NOT assume defaults
- Does NOT modify query semantics

DESIGN PRINCIPLES
-----------------
- Config-driven only
- Fail fast with actionable errors
- Safe for production pipelines
- Consistent with universal ingestion contract
"""

from typing import Dict, Any
import os
import logging

import pandas as pd
from sqlalchemy import create_engine


logger = logging.getLogger(__name__)


def ingest(
    dataset_name: str,
    dataset_cfg: Dict[str, Any],
    global_cfg: Dict[str, Any],
    paths_cfg: Dict[str, Any],
) -> pd.DataFrame:

    db_cfg = dataset_cfg.get("database")
    if not isinstance(db_cfg, dict):
        raise ValueError(
            f"[DB INGESTION] Missing or invalid 'database' config for '{dataset_name}'"
        )

    # --------------------------------------------------
    # Validate credentials
    # --------------------------------------------------
    user_env = db_cfg.get("username_env")
    pass_env = db_cfg.get("password_env")

    if not isinstance(user_env, str) or not isinstance(pass_env, str):
        raise ValueError(
            f"[DB INGESTION] username_env and password_env must be defined "
            f"for dataset '{dataset_name}'"
        )

    user = os.getenv(user_env)
    password = os.getenv(pass_env)

    if not user or not password:
        raise EnvironmentError(
            f"[DB INGESTION] Credentials not found in environment "
            f"({user_env}, {pass_env}) for '{dataset_name}'"
        )

    # --------------------------------------------------
    # Validate connection config
    # --------------------------------------------------
    required_keys = {"engine", "host", "port", "database", "fetch_size"}
    missing = required_keys - db_cfg.keys()

    if missing:
        raise ValueError(
            f"[DB INGESTION] Missing DB config keys for '{dataset_name}': {sorted(missing)}"
        )

    if not isinstance(db_cfg["engine"], str):
        raise ValueError(f"[DB INGESTION] 'engine' must be string for '{dataset_name}'")

    if not isinstance(db_cfg["host"], str):
        raise ValueError(f"[DB INGESTION] 'host' must be string for '{dataset_name}'")

    if not isinstance(db_cfg["port"], int):
        raise ValueError(f"[DB INGESTION] 'port' must be integer for '{dataset_name}'")

    if not isinstance(db_cfg["database"], str):
        raise ValueError(f"[DB INGESTION] 'database' must be string for '{dataset_name}'")

    if not isinstance(db_cfg["fetch_size"], int) or db_cfg["fetch_size"] <= 0:
        raise ValueError(
            f"[DB INGESTION] 'fetch_size' must be positive integer for '{dataset_name}'"
        )

    # --------------------------------------------------
    # Enforce single load mode
    # --------------------------------------------------
    has_query = "query" in db_cfg
    has_table = "table" in db_cfg

    if has_query and has_table:
        raise ValueError(
            f"[DB INGESTION] Define only ONE of 'query' or 'table' for '{dataset_name}'"
        )

    if not has_query and not has_table:
        raise ValueError(
            f"[DB INGESTION] Either 'query' or 'table' must be defined for '{dataset_name}'"
        )

    engine_str = (
        f"{db_cfg['engine']}://{user}:{password}"
        f"@{db_cfg['host']}:{db_cfg['port']}"
        f"/{db_cfg['database']}"
    )

    logger.info(
        f"[DB INGESTION] Connecting to database for dataset '{dataset_name}'"
    )

    engine = create_engine(engine_str)

    # --------------------------------------------------
    # Execute Load
    # --------------------------------------------------
    if has_query:

        sql = db_cfg["query"]
        if not isinstance(sql, str):
            raise ValueError(
                f"[DB INGESTION] 'query' must be string for '{dataset_name}'"
            )

        result = pd.read_sql(
            sql,
            con=engine,
            chunksize=db_cfg["fetch_size"],
        )

    else:
        table = db_cfg["table"]
        schema = db_cfg.get("schema")

        if not isinstance(table, str):
            raise ValueError(
                f"[DB INGESTION] 'table' must be string for '{dataset_name}'"
            )

        result = pd.read_sql_table(
            table_name=table,
            con=engine,
            schema=schema,
            chunksize=db_cfg["fetch_size"],
        )

    if isinstance(result, pd.DataFrame):
        df = result
    else:
        chunks = list(result)
        if not chunks:
            raise ValueError(
                f"[DB INGESTION] No data returned for '{dataset_name}'"
            )
        df = pd.concat(chunks, ignore_index=True)

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"[DB INGESTION] Expected pandas DataFrame for '{dataset_name}', "
            f"got {type(df).__name__}"
        )

    logger.info(
        f"[DB INGESTION] Successfully loaded dataset '{dataset_name}' "
        f"with shape {df.shape}"
    )

    return df
