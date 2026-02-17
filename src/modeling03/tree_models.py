# src/modeling03/tree_models.py

from typing import Dict

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def _validate_modeling_config(config: Dict, model_key: str) -> None:
    """
    Validate required config structure for tree models.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary.")

    if "seeds" not in config or "global_seed" not in config["seeds"]:
        raise ValueError("Missing 'seeds.global_seed' in configuration.")

    if (
        "modeling" not in config
        or "models" not in config["modeling"]
        or model_key not in config["modeling"]["models"]
    ):
        raise ValueError(
            f"Missing 'modeling.models.{model_key}' configuration."
        )


def get_random_forest_model(config: Dict) -> RandomForestRegressor:
    """
    Create an unfitted Random Forest Regressor.

    Parameters
    ----------
    config : Dict
        Configuration dictionary containing modeling parameters.

    Returns
    -------
    RandomForestRegressor
        Unfitted Random Forest model.

    Notes
    -----
    - Uses base parameters from config.
    - No hyperparameter tuning applied here.
    """

    _validate_modeling_config(config, "random_forest")

    params = config["modeling"]["models"]["random_forest"].get("params", {}).copy()

    # Enforce seed & parallelism safely
    params["random_state"] = config["seeds"]["global_seed"]
    params["n_jobs"] = -1

    return RandomForestRegressor(**params)


def get_xgboost_model(config: Dict) -> XGBRegressor:
    """
    Create an unfitted XGBoost Regressor.

    Parameters
    ----------
    config : Dict
        Configuration dictionary containing modeling parameters.

    Returns
    -------
    XGBRegressor
        Unfitted XGBoost model.

    Notes
    -----
    - Uses base parameters from config.
    - No hyperparameter tuning applied here.
    """

    _validate_modeling_config(config, "xgboost")

    params = config["modeling"]["models"]["xgboost"].get("params", {}).copy()

    params["random_state"] = config["seeds"]["global_seed"]
    params["n_jobs"] = -1
    params["verbosity"] = 0

    return XGBRegressor(**params)


def get_lightgbm_model(config: Dict) -> LGBMRegressor:
    """
    Create an unfitted LightGBM Regressor.

    Parameters
    ----------
    config : Dict
        Configuration dictionary containing modeling parameters.

    Returns
    -------
    LGBMRegressor
        Unfitted LightGBM model.

    Notes
    -----
    - Uses base parameters from config.
    - No hyperparameter tuning applied here.
    """

    _validate_modeling_config(config, "lightgbm")

    params = config["modeling"]["models"]["lightgbm"].get("params", {}).copy()

    params["random_state"] = config["seeds"]["global_seed"]
    params["n_jobs"] = -1

    return LGBMRegressor(**params)
