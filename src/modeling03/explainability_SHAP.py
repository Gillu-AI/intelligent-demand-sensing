"""
SHAP Explainability Module
===========================

This module generates SHAP explainability artifacts
for supported regression models.

Design Principles:
------------------
- Fully config-driven
- Model-type aware (Tree / Linear detection)
- Safe fallback for unsupported models
- No hardcoded assumptions
- Production-safe (no crash on SHAP failure)
- Timestamp-based artifact versioning

Supported Models:
-----------------
Tree-based:
    - RandomForest
    - XGBoost
    - LightGBM

Linear:
    - LinearRegression
    - Ridge

Unsupported:
    - Prophet (skipped safely)
    - Custom ensembles without native support

Outputs:
--------
- SHAP summary plot (PNG)
Saved under:
    config["paths"]["output"]["plots"]
"""

import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict


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

    model_name = model.__class__.__name__.lower()

    # Tree models
    if hasattr(model, "feature_importances_"):
        return "tree"

    # Linear models
    if hasattr(model, "coef_"):
        return "linear"

    # Unsupported
    return "unsupported"


def _get_sample_data(
    X: pd.DataFrame,
    sample_size: int
) -> pd.DataFrame:
    """
    Subsample dataset for SHAP computation.

    Why:
    ----
    SHAP can be computationally expensive.
    Sampling keeps production safe.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    sample_size : int
        Maximum rows to sample.

    Returns
    -------
    pd.DataFrame
    """

    if len(X) <= sample_size:
        return X

    return X.sample(sample_size, random_state=42)


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
    Generate SHAP explainability plot for supported models.

    Parameters
    ----------
    model : Any
        Trained model object.
    X_train : pd.DataFrame
        Training features (used for background dataset).
    X_test : pd.DataFrame
        Test features (used for explanation).
    model_name : str
        Name of selected best model.
    output_dir : str
        Directory where SHAP plot will be saved.
    config : Dict
        Project configuration dictionary.
    timestamp : str
        Current run timestamp for versioning.

    Behavior
    --------
    - Selects appropriate SHAP explainer
    - Applies config-driven sampling
    - Saves summary plot as PNG
    - Fails gracefully if unsupported
    """

    if not config.get("explainability", {}).get("enabled", False):
        return

    sample_size = config.get("explainability", {}).get("sample_size", 500)

    model_type = _detect_model_type(model)

    if model_type == "unsupported":
        print(f"SHAP skipped: Model '{model_name}' not supported.")
        return

    try:

        # Subsample for performance
        X_train_sample = _get_sample_data(X_train, sample_size)
        X_test_sample = _get_sample_data(X_test, sample_size)

        # Select explainer
        if model_type == "tree":
            explainer = shap.TreeExplainer(model)

        elif model_type == "linear":
            explainer = shap.LinearExplainer(
                model,
                X_train_sample,
                feature_dependence="independent"
            )

        else:
            print(f"SHAP skipped: No valid explainer for '{model_name}'.")
            return

        shap_values = explainer(X_test_sample)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Generate summary plot
        shap.summary_plot(
            shap_values,
            X_test_sample,
            show=False
        )

        plot_path = os.path.join(
            output_dir,
            f"shap_summary_{model_name}_{timestamp}.png"
        )

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"SHAP summary saved to: {plot_path}")

    except Exception as e:
        # Fail-safe behavior: never crash pipeline due to SHAP
        print(f"SHAP generation failed for '{model_name}': {str(e)}")
