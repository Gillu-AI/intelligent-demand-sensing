# src/modeling03/ensemble.py

"""
Stacking Ensemble Module
===========================================================

Implements classic Out-Of-Fold (OOF) stacking using TimeSeriesSplit.

Design Principles
-----------------
- No data leakage
- Deterministic folds
- Time-series discipline (no shuffle)
- Config-driven CV splits
- Seed propagation
- Production-safe behavior

Stacking Strategy
-----------------
1. Generate out-of-fold predictions for each base model.
2. Train meta-model on OOF predictions.
3. Refit base models on full training data.
4. During inference:
   - Base models predict on X_test
   - Meta-model predicts using stacked test predictions

This guarantees meta-model never sees in-sample predictions.
"""

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from utils.model_utils import train_model


# ==========================================================
# Internal Utilities
# ==========================================================

def _validate_stacking_config(config: Dict) -> Dict:
    if "ensemble" not in config:
        raise ValueError("Missing 'ensemble' configuration.")

    ensemble_cfg = config["ensemble"]

    if not ensemble_cfg.get("enabled", False):
        raise ValueError("Stacking requested but ensemble.enabled is False.")

    if ensemble_cfg.get("method", "").lower() != "stacking":
        raise ValueError("Ensemble method must be 'stacking'.")

    if "seeds" not in config or "global_seed" not in config["seeds"]:
        raise ValueError("Missing 'seeds.global_seed'.")

    return ensemble_cfg


def _get_meta_model(config: Dict):
    meta_name = config["ensemble"]["meta_model"].lower()
    seed = config["seeds"]["global_seed"]

    if meta_name == "ridge":
        return Ridge(random_state=seed)

    raise ValueError(
        f"Unsupported meta_model '{meta_name}'. "
        f"Supported: ['ridge']"
    )


# ==========================================================
# OOF Stacking Implementation
# ==========================================================

def train_stacking_model(
    base_model_factories: Dict[str, callable],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Dict,
):
    """
    Train leakage-safe stacking ensemble using OOF predictions.

    Parameters
    ----------
    base_model_factories : Dict[str, callable]
        Dictionary of model factories (NOT fitted models).
    X_train : pd.DataFrame
    y_train : pd.Series
    config : Dict

    Returns
    -------
    dict
        {
            "base_models": fitted_base_models,
            "meta_model": fitted_meta_model
        }
    """

    ensemble_cfg = _validate_stacking_config(config)

    if X_train.empty or y_train.empty:
        raise ValueError("Training data cannot be empty.")

    cv_splits = config["hyperparameter_tuning"]["cv_splits"]

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    # OOF prediction storage
    oof_predictions = {
        name: np.zeros(len(X_train))
        for name in base_model_factories.keys()
    }

    # ======================================================
    # Generate OOF Predictions
    # ======================================================

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]

        for name, factory in base_model_factories.items():

            model = factory()
            train_model(model, X_tr, y_tr)

            preds = model.predict(X_val)
            oof_predictions[name][val_idx] = preds

    # Build OOF feature matrix
    stacking_X = pd.DataFrame(oof_predictions, index=X_train.index)

    # Train meta-model on OOF predictions
    meta_model = _get_meta_model(config)
    meta_model.fit(stacking_X, y_train)

    # ======================================================
    # Refit Base Models on Full Training Data
    # ======================================================

    fitted_base_models = {}

    for name, factory in base_model_factories.items():
        model = factory()
        fitted_model = train_model(model, X_train, y_train)
        fitted_base_models[name] = fitted_model

    return {
        "base_models": fitted_base_models,
        "meta_model": meta_model
    }


def predict_stacking_model(
    stacking_bundle: Dict,
    X_test: pd.DataFrame
):
    """
    Predict using trained stacking ensemble.

    Parameters
    ----------
    stacking_bundle : Dict
        Output from train_stacking_model.
    X_test : pd.DataFrame

    Returns
    -------
    np.ndarray
        Ensemble predictions.
    """

    if X_test.empty:
        raise ValueError("X_test cannot be empty.")

    base_models = stacking_bundle["base_models"]
    meta_model = stacking_bundle["meta_model"]

    stacking_X_test = pd.DataFrame(index=X_test.index)

    for name, model in base_models.items():
        stacking_X_test[name] = model.predict(X_test)

    preds = meta_model.predict(stacking_X_test)

    return preds