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

    cv_splits = config["hyperparameter_tuning"]["cv_splits"]
    n_iter = config["hyperparameter_tuning"]["n_iter"]

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_mean_absolute_percentage_error",
        n_jobs=-1,
        random_state=42
    )

    search.fit(X, y)

    return search.best_estimator_
