# src/modeling03/hyperparameter_tuning.py

"""
Dynamic Hyperparameter Tuning Module
=====================================

Performs tuning only for selected top-N models
based on primary metric (MAPE).
"""

from typing import Dict
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import numpy as np


def tune_model(model, param_grid: Dict, X, y, config: Dict):
    """
    Tune given model using RandomizedSearchCV
    with TimeSeriesSplit.

    Parameters
    ----------
    model : sklearn model instance
    param_grid : Dict
        Hyperparameter search space.
    X : pd.DataFrame
    y : pd.Series
    config : Dict

    Returns
    -------
    best_model : tuned model
    """

    if "hyperparameter_tuning" not in config:
        raise ValueError("Missing 'hyperparameter_tuning' configuration.")

    tuning_cfg = config["hyperparameter_tuning"]

    if not tuning_cfg.get("enabled", False):
        return model

    if not param_grid:
        raise ValueError("param_grid cannot be empty for hyperparameter tuning.")

    if X.empty or y.empty:
        raise ValueError("X and y must not be empty for hyperparameter tuning.")

    cv_splits = tuning_cfg.get("cv_splits")
    n_iter = tuning_cfg.get("n_iter")

    if not isinstance(cv_splits, int) or cv_splits < 2:
        raise ValueError("cv_splits must be integer >= 2.")

    if not isinstance(n_iter, int) or n_iter <= 0:
        raise ValueError("n_iter must be positive integer.")

    if "seeds" not in config or "global_seed" not in config["seeds"]:
        raise ValueError("Missing 'seeds.global_seed' configuration.")

    seed = config["seeds"]["global_seed"]

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_mean_absolute_percentage_error",
        n_jobs=-1,
        random_state=seed,
    )

    search.fit(X, y)

    return search.best_estimator_
