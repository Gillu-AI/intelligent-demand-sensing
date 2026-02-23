# src/pipelines/run_full_pipeline.py

"""
Full IDS Pipeline Orchestration
================================

Enterprise Orchestrator for the entire IDS system.

Executes in correct order:

1. Ingestion
2. Feature Engineering
3. Model Training
4. Inventory Planning
5. Visualization (Optional, Config-Controlled)

Design Principles:
------------------
- No business logic
- No hardcoded behavior
- Strict logger usage
- Fully config-driven execution
- Fail-fast by default
- Production-safe structure
- Visualization separated from core pipeline
"""

from utils.config_loader import load_config
from utils.logger import get_logger

from pipelines.run_ingestion import run_ingestion
from pipelines.run_features import run_features
from pipelines.run_training import run_training
from pipelines.run_inventory import run_inventory
from pipelines.run_visualization import run_visualization


# ==========================================================
# Full Pipeline Controller
# ==========================================================

def run_full_pipeline():
    """
    Execute full IDS workflow in correct sequence.

    Execution is controlled via config.execution.mode

    Modes:
    ------
    dev   → Full flow + visualization
    train → Full training cycle + visualization
    prod  → Forecast + inventory only (no retraining, no visualization)
    """

    config = load_config("config/config.yaml")
    logger = get_logger(config)

    execution_mode = config.get("execution", {}).get("mode", "train")
    visualization_enabled = config.get("visualization", {}).get("enabled", True)

    logger.info("========================================")
    logger.info("Starting IDS Full Pipeline")
    logger.info(f"Execution Mode: {execution_mode}")
    logger.info("========================================")

    try:

        # ---------------------------------------------
        # DEV / TRAIN MODE → Run Full Flow
        # ---------------------------------------------
        if execution_mode in ["dev", "train"]:

            logger.info("STEP 1: Ingestion")
            run_ingestion()

            logger.info("STEP 2: Feature Engineering")
            run_features()

            logger.info("STEP 3: Model Training")
            run_training()

            logger.info("STEP 4: Inventory Planning")
            run_inventory()

            # Optional Visualization Layer
            if visualization_enabled:
                logger.info("STEP 5: Visualization")
                run_visualization()
            else:
                logger.info("Visualization disabled via config.")

        # ---------------------------------------------
        # PROD MODE → Skip retraining & visualization
        # ---------------------------------------------
        elif execution_mode == "prod":

            logger.info("Production Mode Detected.")
            logger.info("Skipping ingestion, feature engineering, and training.")
            logger.info("Running inventory forecasting only.")

            run_inventory()

            logger.info("Visualization skipped in production mode.")

        else:
            raise ValueError(f"Unknown execution mode: {execution_mode}")

        logger.info("========================================")
        logger.info("IDS Pipeline Completed Successfully")
        logger.info("========================================")

    except Exception:
        logger.exception("Pipeline failed due to an error.")
        raise


if __name__ == "__main__":
    run_full_pipeline()
