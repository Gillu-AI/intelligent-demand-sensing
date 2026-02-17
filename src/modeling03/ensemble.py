# src/modeling03/ensemble.py

"""
Stacking Ensemble Module
========================

Builds stacking model using top-K models
and configurable meta-model.
"""

from typing import Dict
import pandas as pd
from sklearn.linear_model import Ridge


def build_stacking_features(
    models: Dict,
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate prediction matrix from base models.

    Parameters
    ----------
    models : Dict
        Dictionary of trained base models.
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    pd.DataFrame
        DataFrame where each column is predictions
        from one base model.
    """

    if not models:
        raise ValueError("Base models dictionary cannot be empty for stacking.")

    stacking_df = pd.DataFrame(index=X.index)

    for name, model in models.items():
        stacking_df[name] = model.predict(X)

    return stacking_df


def _get_meta_model(config: Dict):
    """
    Initialize meta-model based on config.

    Currently supported:
    - ridge
    """

    if "ensemble" not in config:
        raise ValueError("Missing 'ensemble' configuration.")

    ensemble_cfg = config["ensemble"]

    if "meta_model" not in ensemble_cfg:
        raise ValueError("Missing 'ensemble.meta_model' configuration.")

    meta_name = ensemble_cfg["meta_model"].lower()

    seed = config.get("seeds", {}).get("global_seed")

    if meta_name == "ridge":
        return Ridge(random_state=seed)

    raise ValueError(
        f"Unsupported meta_model '{meta_name}'. "
        f"Currently supported: ['ridge']"
    )


def train_stacking_model(
    base_models: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Dict
):
    """
    Train stacking meta-model using base model predictions.

    Parameters
    ----------
    base_models : Dict
        Top-K trained models.
    X_train : pd.DataFrame
    y_train : pd.Series
    config : Dict

    Returns
    -------
    meta_model : trained meta-model
    """

    if "ensemble" not in config:
        raise ValueError("Missing 'ensemble' configuration.")

    ensemble_cfg = config["ensemble"]

    if not ensemble_cfg.get("enabled", False):
        raise ValueError("Stacking requested but ensemble.enabled is False.")

    if ensemble_cfg.get("method", "").lower() != "stacking":
        raise ValueError("Ensemble method must be 'stacking' for this module.")

    if X_train.empty or y_train.empty:
        raise ValueError("Training data cannot be empty for stacking.")

    stacking_X_train = build_stacking_features(base_models, X_train)

    meta_model = _get_meta_model(config)

    meta_model.fit(stacking_X_train, y_train)

    return meta_model


def predict_stacking_model(
    base_models: Dict,
    meta_model,
    X_test: pd.DataFrame
):
    """
    Predict using stacking ensemble.

    Parameters
    ----------
    base_models : Dict
    meta_model : trained meta model
    X_test : pd.DataFrame

    Returns
    -------
    np.ndarray
        Ensemble predictions.
    """

    if X_test.empty:
        raise ValueError("X_test cannot be empty for stacking prediction.")

    stacking_X_test = build_stacking_features(base_models, X_test)

    preds = meta_model.predict(stacking_X_test)

    return preds
