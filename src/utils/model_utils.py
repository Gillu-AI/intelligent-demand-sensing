# src/utils/model_utils.py

import pandas as pd


def train_model(model, X: pd.DataFrame, y: pd.Series):
    """
    Fit a machine learning model on training data.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Unfitted model instance.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.

    Returns
    -------
    model
        Fitted model instance.

    Notes
    -----
    - Generic training helper.
    - Works for all sklearn-compatible regressors.
    - Centralized to avoid duplication across model modules.
    """
    model.fit(X, y)
    return model
