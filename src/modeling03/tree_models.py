# src/modeling03/tree_models.py

from typing import Dict
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


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

    params = config["modeling"]["models"]["random_forest"].get("params", {})

    return RandomForestRegressor(
        **params,
        random_state=config["seeds"]["global_seed"],
        n_jobs=-1
    )


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

    params = config["modeling"]["models"]["xgboost"].get("params", {})

    return XGBRegressor(
        **params,
        random_state=config["seeds"]["global_seed"],
        n_jobs=-1,
        verbosity=0
    )


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

    params = config["modeling"]["models"]["lightgbm"].get("params", {})

    return LGBMRegressor(
        **params,
        random_state=config["seeds"]["global_seed"],
        n_jobs=-1
    )