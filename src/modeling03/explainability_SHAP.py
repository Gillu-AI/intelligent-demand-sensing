# src/modeling03/explainability_SHAP.py

"""
SHAP Explainability Module (Governed & Deterministic)
======================================================

Enterprise-grade SHAP integration aligned with IDS architecture.

Governance Rules
----------------
- Fully config-driven via config["shap"]
- Disabled automatically in production if configured
- Deterministic sampling using global seed
- Never crashes training pipeline
- Timestamp-based artifact versioning
- Model-type aware (Tree / Linear detection)

Supported Models
----------------
Tree-based:
    - RandomForest
    - XGBoost
    - LightGBM

Linear:
    - LinearRegression
    - Ridge

Unsupported:
    - Prophet
    - Custom ensembles without native support
"""

import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Any, Dict


logger = logging.getLogger(__name__)


# ==========================================================
# Internal Utilities
# ==========================================================

def _detect_model_type(model: Any) -> str:
    """
    Detect model type for SHAP explainer selection.

    Returns
    -------
    str
        "tree" | "linear" | "unsupported"
    """

    if hasattr(model, "feature_importances_"):
        return "tree"

    if hasattr(model, "coef_"):
        return "linear"

    return "unsupported"


def _get_sample_data(
    X: pd.DataFrame,
    sample_size: int,
    seed: int
) -> pd.DataFrame:
    """
    Deterministic subsampling for SHAP computation.
    """

    if not isinstance(X, pd.DataFrame):
        raise ValueError("SHAP input must be a pandas DataFrame.")

    if X.empty:
        raise ValueError("Cannot compute SHAP on empty dataset.")

    if len(X) <= sample_size:
        return X

    return X.sample(n=sample_size, random_state=seed)


# ==========================================================
# Public API
# ==========================================================

def generate_shap_explainability(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str,
    output_dir: str,
    config: Dict,
    timestamp: str
) -> None:
    """
    Generate SHAP summary plot under strict governance rules.

    Parameters
    ----------
    model : Any
        Trained model object.
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    model_name : str
        Name of selected model.
    output_dir : str
        Directory where SHAP plot will be saved.
    config : Dict
        Full project configuration.
    timestamp : str
        Timestamp for artifact versioning.
    """

    # ------------------------------------------------------
    # Governance Check
    # ------------------------------------------------------

    shap_cfg = config.get("shap", {})

    if not shap_cfg.get("enabled", False):
        logger.info("SHAP disabled via config.")
        return

    execution_mode = config.get("execution", {}).get("mode", "train")

    if execution_mode == "prod":
        logger.info("SHAP skipped: disabled in production mode.")
        return

    # ------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------

    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("Invalid SHAP output directory.")

    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError("Invalid timestamp for SHAP artifact.")

    sample_size = shap_cfg.get("sample_size", 500)

    if not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("shap.sample_size must be positive integer.")

    if "seeds" not in config or "global_seed" not in config["seeds"]:
        raise ValueError("Missing 'seeds.global_seed' for SHAP.")

    seed = config["seeds"]["global_seed"]

    model_type = _detect_model_type(model)

    if model_type == "unsupported":
        logger.info(f"SHAP skipped: '{model_name}' not supported.")
        return

    try:

        # --------------------------------------------------
        # Deterministic Sampling
        # --------------------------------------------------

        X_train_sample = _get_sample_data(X_train, sample_size, seed)
        X_test_sample = _get_sample_data(X_test, sample_size, seed)

        logger.info(
            f"SHAP sampling -> train: {len(X_train_sample)}, "
            f"test: {len(X_test_sample)}"
        )

        # --------------------------------------------------
        # Select Explainer
        # --------------------------------------------------

        if model_type == "tree":
            explainer = shap.TreeExplainer(model)

        elif model_type == "linear":
            explainer = shap.LinearExplainer(
                model,
                X_train_sample
            )

        else:
            logger.info(f"SHAP skipped: no valid explainer.")
            return

        shap_values = explainer(X_test_sample)

        # --------------------------------------------------
        # Artifact Persistence
        # --------------------------------------------------

        os.makedirs(output_dir, exist_ok=True)

        plot_path = os.path.join(
            output_dir,
            f"shap_summary_{model_name}_{timestamp}.png"
        )

        shap.summary_plot(
            shap_values,
            X_test_sample,
            show=False
        )

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"SHAP summary saved to: {plot_path}")

    except Exception as e:
        logger.exception(
            f"SHAP generation failed for '{model_name}': {str(e)}"
        )