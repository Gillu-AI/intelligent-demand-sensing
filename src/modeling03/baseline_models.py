# src/modeling03/baseline_models.py

from typing import Dict
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge


def get_linear_regression_model() -> LinearRegression:
    """
    Create an unfitted Linear Regression model.

    Returns
    -------
    LinearRegression
        Sklearn LinearRegression instance.

    Notes
    -----
    - Ordinary Least Squares model.
    - No hyperparameters to tune.
    - Used as baseline benchmark.
    """
    return LinearRegression()


def get_ridge_model(config: Dict) -> Ridge:
    """
    Create an unfitted Ridge Regression model.

    Parameters
    ----------
    config : Dict
        Configuration dictionary (used to extract global seed).

    Returns
    -------
    Ridge
        Sklearn Ridge instance.

    Notes
    -----
    - Alpha will be set externally (via tuning or config).
    - Regularized linear model.
    """
    return Ridge(random_state=config["seeds"]["global_seed"])