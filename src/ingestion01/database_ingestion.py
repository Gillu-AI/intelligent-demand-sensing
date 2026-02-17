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
    paths_cfg: Dict[str, Any],  # required by frozen contract
) -> pd.DataFrame:
    """
    Ingest a dataset from a relational database.

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
        Raw data loaded from the database

    Raises
    ------
    ValueError
        If configuration is invalid
    EnvironmentError
        If required credentials are missing
    TypeError
        If ingestion output violates the contract
    """

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

    if not user_env or not pass_env:
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
    required_keys = {"engine", "host", "port", "database"}
    missing = required_keys - db_cfg.keys()

    if missing:
        raise ValueError(
            f"[DB INGESTION] Missing DB config keys for '{dataset_name}': {sorted(missing)}"
        )
    
    if not isinstance(db_cfg["port"], int):
        raise ValueError(
            f"[DB INGESTION] 'port' must be integer for '{dataset_name}'"
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
    # Build SQL / Execute Load
    # --------------------------------------------------
    if "query" in db_cfg:

        sql = db_cfg["query"]
        logger.info(
            f"[DB INGESTION] Executing query load for dataset '{dataset_name}'"
        )

        result = pd.read_sql(
            sql,
            con=engine,
            chunksize=db_cfg.get("fetch_size"),
        )

        if isinstance(result, pd.DataFrame):
            df = result
        else:
            chunks = list(result)
            if not chunks:
                raise ValueError(
                    f"[DB INGESTION] Query returned no data for '{dataset_name}'"
                )
            df = pd.concat(chunks, ignore_index=True)

    elif "table" in db_cfg:

        table = db_cfg["table"]
        schema = db_cfg.get("schema")

        if not isinstance(table, str):
            raise ValueError(
                f"[DB INGESTION] 'table' must be a string for '{dataset_name}'"
            )

        logger.info(
            f"[DB INGESTION] Executing table load for dataset '{dataset_name}'"
        )

        result = pd.read_sql_table(
            table_name=table,
            con=engine,
            schema=schema,
            chunksize=db_cfg.get("fetch_size"),
        )

        if isinstance(result, pd.DataFrame):
            df = result
        else:
            chunks = list(result)
            if not chunks:
                raise ValueError(
                    f"[DB INGESTION] Table returned no data for '{dataset_name}'"
                )
            df = pd.concat(chunks, ignore_index=True)

    else:
        raise ValueError(
            f"[DB INGESTION] Either 'table' or 'query' must be defined for '{dataset_name}'"
        )

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
