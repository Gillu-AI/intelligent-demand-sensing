# src/llm05/scenario_store.py

"""
Scenario Store
==============

Handles persistent storage of scenarios.

Features:
---------
- JSON-based storage
- Config-driven TTL expiration
- Scenario retrieval by ID
- History listing
- Auto-clean expired scenarios
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List
from utils.helpers import ensure_directory
from utils.logger import get_logger


class ScenarioStore:
    """
    JSON-backed persistent scenario storage.
    """

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

        if not isinstance(self.ttl_hours, (int, float)) or self.ttl_hours < 0:
            raise ValueError("llm.scenario_ttl_hours must be non-negative number.")

        ensure_directory(os.path.dirname(self.store_path))

        if not os.path.exists(self.store_path):
            with open(self.store_path, "w") as f:
                json.dump({}, f)

    def _load(self) -> Dict:
        try:
            with open(self.store_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            self.logger.warning("Scenario store corrupted. Resetting store.")
            return {}

    def _save(self, data: Dict):
        with open(self.store_path, "w") as f:
            json.dump(data, f, indent=4)

    def _cleanup_expired(self, data: Dict) -> Dict:

        now = datetime.utcnow()

        cleaned = {}
        for sid, record in data.items():

            try:
                created = datetime.fromisoformat(record["created_at"])
            except Exception:
                continue

            if now - created < timedelta(hours=self.ttl_hours):
                cleaned[sid] = record

        return cleaned

    def save_scenario(self, scenario_id: str, record: Dict):

        if not scenario_id:
            raise ValueError("scenario_id cannot be empty.")

        if "created_at" not in record:
            raise ValueError("Scenario record must contain 'created_at' field.")

        data = self._load()
        data[scenario_id] = record
        data = self._cleanup_expired(data)
        self._save(data)

        self.logger.info(f"Scenario '{scenario_id}' saved.")

    def get_scenario(self, scenario_id: str) -> Dict:

        data = self._load()
        data = self._cleanup_expired(data)
        self._save(data)

        return data.get(scenario_id)

    def list_scenarios(self) -> List[str]:

        data = self._load()
        data = self._cleanup_expired(data)
        self._save(data)

        return list(data.keys())

    def delete_scenario(self, scenario_id: str):

        data = self._load()

        if scenario_id in data:
            del data[scenario_id]
            self._save(data)
            self.logger.info(f"Scenario '{scenario_id}' deleted.")
