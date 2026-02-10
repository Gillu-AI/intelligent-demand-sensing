"""
test_ingestion_reference.py
===========================

Purpose
-------
End-to-end validation script for the IDS ingestion layer.

This script verifies that:
1. The Python environment is correctly set up
2. The configuration file can be loaded and validated
3. CSV and Excel datasets are ingested successfully
4. Schema enforcement is applied without errors
5. Final datasets are non-empty and usable downstream

This is NOT a unit test.
This is a functional / integration test for ingestion.
"""

# ---------------------------------------------------------------------
# 0. Environment sanity check
# ---------------------------------------------------------------------
# These imports and prints confirm:
# - Correct Conda environment is active
# - Required libraries are installed
# - Versions are known and reproducible
# ---------------------------------------------------------------------

import numpy
import pandas
import sklearn
import prophet

print("Environment versions:")
print("numpy   :", numpy.__version__)
print("pandas  :", pandas.__version__)
print("sklearn :", sklearn.__version__)
print("prophet :", prophet.__version__)


# ---------------------------------------------------------------------
# 1. Import project-level orchestration utilities
# ---------------------------------------------------------------------
# We deliberately import ONLY high-level entry points:
# - config_loader → loads & validates config.yaml
# - universal_loader → orchestrates ingestion
#
# We do NOT import:
# - csv_ingestion
# - excel_ingestion
# - schema_utils
#
# This proves the architecture works via contracts, not tight coupling.
# ---------------------------------------------------------------------

from utils.config_loader import load_config
from ingestion01.universal_loader import load_all_datasets


# ---------------------------------------------------------------------
# 2. Load and validate configuration
# ---------------------------------------------------------------------
# This step performs:
# - File existence check
# - YAML parsing
# - Required section validation
# - Unknown section detection
#
# If this passes, the entire project configuration is structurally valid.
# ---------------------------------------------------------------------

config = load_config("config/config.yaml")


# ---------------------------------------------------------------------
# 3. Resolve paths configuration
# ---------------------------------------------------------------------
# All file paths used by ingestion come from config.yaml.
# No hardcoded paths exist in ingestion logic.
#
# This allows portability across:
# - Local machines
# - CI pipelines
# - Cloud / container environments
# ---------------------------------------------------------------------

paths_cfg = config["paths"]


# ---------------------------------------------------------------------
# 4. Run ingestion pipeline
# ---------------------------------------------------------------------
# This single call:
# - Iterates over ingestion.datasets
# - Skips disabled datasets
# - Dispatches to correct loader (CSV / Excel / etc.)
# - Applies schema enforcement in a fixed order
# - Enforces strict/non-strict error behavior
# - Returns all successfully loaded datasets
# ---------------------------------------------------------------------

datasets = load_all_datasets(
    config=config,
    paths_cfg=paths_cfg,
)


# ---------------------------------------------------------------------
# 5. Inspect ingestion results
# ---------------------------------------------------------------------
# We print dataset names and shapes to allow:
# - Human verification
# - Quick debugging
#
# Handles both:
# - Single DataFrame outputs
# - Multi-sheet Excel outputs (dict[str, DataFrame])
# ---------------------------------------------------------------------

print("\nLoaded datasets:")

for dataset_name, data in datasets.items():
    print(f"\n{dataset_name}:")

    if isinstance(data, dict):
        # Multi-sheet Excel case
        for sheet_name, df in data.items():
            print(f"  {sheet_name} -> shape={df.shape}")
    else:
        # Single DataFrame case
        print(f"  shape={data.shape}")


# ---------------------------------------------------------------------
# 6. Machine-level assertions (safety checks)
# ---------------------------------------------------------------------
# These assertions ensure:
# - Expected datasets exist
# - Datasets are not empty
#
# If any assertion fails, ingestion is considered broken.
# This is the foundation for future CI tests.
# ---------------------------------------------------------------------

assert "calendar" in datasets, "Calendar dataset missing after ingestion"
assert "sales" in datasets, "Sales dataset missing after ingestion"

assert datasets["calendar"].shape[0] > 0, "Calendar dataset is empty"
assert datasets["sales"].shape[0] > 0, "Sales dataset is empty"


print("\nIngestion test completed successfully.")
