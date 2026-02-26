# src/modeling03/hyperparameter_tuning.py

"""
Dynamic Hyperparameter Tuning Module (Enterprise Hardened)
===========================================================

Enhancements
------------
- Deterministic TimeSeriesSplit
- No shuffle
- n_iter auto-capped
- Detailed fold logging
- CV diagnostics persistence (JSON)
- Full CV results persistence (CSV)
- Seed enforcement
- Production-safe behavior

All artifacts are saved under:
config["paths"]["output"]["metrics"]
"""

from typing import Dict
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit


# =============================================================================
# JSON SAFETY UTILITY (ONLY FIX ADDED)
# =============================================================================

def _make_json_safe(obj):
    """
    Recursively convert NumPy scalar types to native Python types
    to avoid JSON serialization errors.
    """

    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(_make_json_safe(v) for v in obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    return obj


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def tune_model(
    model,
    param_grid: Dict,
    X,
    y,
    config: Dict,
    model_name: str,
    timestamp: str,
):
    """
    Perform deterministic hyperparameter tuning with
    CV artifact persistence.

    Parameters
    ----------
    model : sklearn estimator
    param_grid : Dict
    X : pd.DataFrame
    y : pd.Series
    config : Dict
    model_name : str
    timestamp : str

    Returns
    -------
    best_model : fitted estimator
    """

    # --------------------------------------------------
    # Config Validation
    # --------------------------------------------------

    tuning_cfg = config.get("hyperparameter_tuning", {})

    if not tuning_cfg.get("enabled", False):
        return model

    if not param_grid:
        raise ValueError("param_grid cannot be empty.")

    seed = config["seeds"]["global_seed"]

    cv_splits = tuning_cfg.get("cv_splits", 5)
    n_iter = tuning_cfg.get("n_iter", 10)

    if cv_splits < 2:
        raise ValueError("cv_splits must be >= 2.")

    # Cap n_iter to avoid sklearn warning
    total_param_space = np.prod([len(v) for v in param_grid.values()])
    n_iter = min(n_iter, total_param_space)

    # --------------------------------------------------
    # Deterministic TimeSeriesSplit
    # --------------------------------------------------

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    # Log fold boundaries
    fold_boundaries = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        fold_boundaries.append({
            "fold": int(fold_idx),
            "train_start": int(train_idx[0]),
            "train_end": int(train_idx[-1]),
            "val_start": int(val_idx[0]),
            "val_end": int(val_idx[-1]),
        })

    # --------------------------------------------------
    # Randomized Search
    # --------------------------------------------------

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_mean_absolute_percentage_error",
        n_jobs=-1,
        random_state=seed,
        return_train_score=False,
    )

    search.fit(X, y)

    # --------------------------------------------------
    # Persist CV Diagnostics
    # --------------------------------------------------

    metrics_dir = config["paths"]["output"]["metrics"]
    os.makedirs(metrics_dir, exist_ok=True)

    diagnostics = {
        "model": model_name,
        "best_params": search.best_params_,
        "best_score_neg_mape": float(search.best_score_),
        "mean_test_score": float(np.mean(search.cv_results_["mean_test_score"])),
        "std_test_score": float(np.std(search.cv_results_["mean_test_score"])),
        "fold_boundaries": fold_boundaries,
        "cv_splits": int(cv_splits),
        "n_iter": int(n_iter),
        "seed": int(seed),
    }

    diagnostics = _make_json_safe(diagnostics)

    json_path = os.path.join(
        metrics_dir,
        f"cv_diagnostics_{model_name}_{timestamp}.json"
    )

    with open(json_path, "w") as f:
        json.dump(diagnostics, f, indent=4)

    # --------------------------------------------------
    # Save Full CV Results (CSV)
    # --------------------------------------------------

    cv_results_df = pd.DataFrame(search.cv_results_)

    csv_path = os.path.join(
        metrics_dir,
        f"cv_results_{model_name}_{timestamp}.csv"
    )

    cv_results_df.to_csv(csv_path, index=False)

    return search.best_estimator_