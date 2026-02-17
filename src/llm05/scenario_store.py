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


class ScenarioStore:
    """
    JSON-backed persistent scenario storage.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.store_path = config["llm"]["storage_path"]
        ensure_directory(os.path.dirname(self.store_path))

        if not os.path.exists(self.store_path):
            with open(self.store_path, "w") as f:
                json.dump({}, f)

    def _load(self) -> Dict:
        with open(self.store_path, "r") as f:
            return json.load(f)

    def _save(self, data: Dict):
        with open(self.store_path, "w") as f:
            json.dump(data, f, indent=4)

    def _cleanup_expired(self, data: Dict) -> Dict:
        ttl_hours = self.config["llm"]["scenario_ttl_hours"]
        now = datetime.utcnow()

        cleaned = {}
        for sid, record in data.items():
            created = datetime.fromisoformat(record["created_at"])
            if now - created < timedelta(hours=ttl_hours):
                cleaned[sid] = record

        return cleaned

    def save_scenario(self, scenario_id: str, record: Dict):
        data = self._load()
        data[scenario_id] = record
        data = self._cleanup_expired(data)
        self._save(data)

    def get_scenario(self, scenario_id: str) -> Dict:
        data = self._load()
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
