# src/llm05/scenario_store.py

"""
Scenario Store
==============

Handles persistent storage of scenarios.

Enterprise Responsibilities:
----------------------------
- JSON-based persistent storage
- Config-driven TTL expiration
- Enforce structured scenario record schema
- Scenario retrieval by ID
- History listing
- Auto-clean expired scenarios
- Defensive corruption handling
- Audit-safe storage logic

Design Principles:
------------------
- No silent failures
- Strict validation before saving
- Fail-fast on structural errors
- Automatic expiration cleanup
- Fully logger-driven
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from utils.helpers import ensure_directory
from utils.logger import get_logger


class ScenarioStore:
    """
    JSON-backed persistent scenario storage.

    Each stored scenario MUST contain:
        - scenario_id
        - created_at (ISO format UTC)
        - prompt
        - mode
        - output_file

    Additional metadata is allowed but required fields
    are strictly enforced.
    """

    REQUIRED_FIELDS = {
        "scenario_id",
        "created_at",
        "prompt",
        "mode",
        "output_file",
    }

    def __init__(self, config: Dict):

        if "llm" not in config:
            raise ValueError("Missing 'llm' section in configuration.")

        llm_cfg = config["llm"]

        if "storage_path" not in llm_cfg:
            raise ValueError("Missing 'llm.storage_path' in configuration.")

        if "scenario_ttl_hours" not in llm_cfg:
            raise ValueError("Missing 'llm.scenario_ttl_hours' in configuration.")

        self.config = config
        self.logger = get_logger(config)

        self.store_path = llm_cfg["storage_path"]
        self.ttl_hours = llm_cfg["scenario_ttl_hours"]

        if not isinstance(self.ttl_hours, (int, float)) or self.ttl_hours <= 0:
            raise ValueError(
                "llm.scenario_ttl_hours must be a positive number."
            )

        ensure_directory(os.path.dirname(self.store_path))

        if not os.path.exists(self.store_path):
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

        # Auto cleanup on initialization
        self._cleanup_store_on_init()

    # =========================================================
    # INTERNAL METHODS
    # =========================================================

    def _load(self) -> Dict[str, Any]:
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                if not isinstance(data, dict):
                    self.logger.warning(
                        "Scenario store structure invalid. Resetting store."
                    )
                    return {}

                return data

        except json.JSONDecodeError:
            self.logger.warning(
                "Scenario store corrupted. Resetting store."
            )
            return {}

    def _save(self, data: Dict[str, Any]) -> None:
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def _cleanup_expired(self, data: Dict[str, Any]) -> Dict[str, Any]:

        now = datetime.utcnow()
        cleaned = {}

        for sid, record in data.items():

            try:
                created = datetime.fromisoformat(record["created_at"])
            except Exception:
                self.logger.warning(
                    f"Invalid timestamp for scenario '{sid}'. Skipping."
                )
                continue

            if now - created < timedelta(hours=self.ttl_hours):
                cleaned[sid] = record
            else:
                self.logger.info(
                    f"Scenario '{sid}' expired and removed."
                )

        return cleaned

    def _cleanup_store_on_init(self) -> None:
        data = self._load()
        cleaned = self._cleanup_expired(data)
        self._save(cleaned)

    def _validate_record(self, scenario_id: str, record: Dict[str, Any]) -> None:

        if not scenario_id:
            raise ValueError("scenario_id cannot be empty.")

        if not isinstance(record, dict):
            raise ValueError("Scenario record must be a dictionary.")

        missing = self.REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(
                f"Scenario record missing required fields: {sorted(missing)}"
            )

        if record["scenario_id"] != scenario_id:
            raise ValueError(
                "scenario_id mismatch between key and record content."
            )

        # Validate ISO timestamp
        try:
            datetime.fromisoformat(record["created_at"])
        except Exception:
            raise ValueError(
                "created_at must be valid ISO format datetime string."
            )

    # =========================================================
    # PUBLIC METHODS
    # =========================================================

    def save_scenario(self, scenario_id: str, record: Dict[str, Any]) -> None:

        self._validate_record(scenario_id, record)

        data = self._load()
        data[scenario_id] = record
        data = self._cleanup_expired(data)
        self._save(data)

        self.logger.info(f"Scenario '{scenario_id}' saved successfully.")

    def get_scenario(self, scenario_id: str) -> Dict[str, Any] | None:

        data = self._load()
        data = self._cleanup_expired(data)
        self._save(data)

        return data.get(scenario_id)

    def list_scenarios(self) -> List[str]:

        data = self._load()
        data = self._cleanup_expired(data)
        self._save(data)

        return sorted(list(data.keys()))

    def delete_scenario(self, scenario_id: str) -> None:

        data = self._load()

        if scenario_id in data:
            del data[scenario_id]
            self._save(data)
            self.logger.info(f"Scenario '{scenario_id}' deleted.")
        else:
            self.logger.warning(
                f"Attempted to delete non-existing scenario '{scenario_id}'."
            )
