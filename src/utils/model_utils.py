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

    # --------------------------------------------------
    # Defensive Validation
    # --------------------------------------------------

    if model is None:
        raise ValueError("Model instance cannot be None.")

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if not isinstance(y, (pd.Series, pd.DataFrame)):
        raise TypeError("y must be a pandas Series.")

    if X.empty:
        raise ValueError("Feature matrix X is empty.")

    if len(y) == 0:
        raise ValueError("Target vector y is empty.")

    if len(X) != len(y):
        raise ValueError(
            f"Mismatch between X and y lengths: "
            f"{len(X)} != {len(y)}"
        )

    if X.shape[1] == 0:
        raise ValueError("Feature matrix X has no columns.")

    # --------------------------------------------------
    # Model Training
    # --------------------------------------------------

    model.fit(X, y)

    return model
