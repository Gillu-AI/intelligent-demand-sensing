# src/utils/model_utils.py

import pandas as pd
from typing import Callable, Dict
from modeling03.evaluate import evaluate_regression_model


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

    if model is None:
        raise ValueError("Model instance cannot be None.")

    if not hasattr(model, "fit"):
        raise TypeError("Model must implement a fit() method.")

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if not isinstance(y, (pd.Series, pd.DataFrame)):
        raise TypeError("y must be a pandas Series or single-column DataFrame.")

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

    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("Target DataFrame must contain exactly one column.")
        y = y.iloc[:, 0]

    model.fit(X, y)

    return model


def train_and_evaluate(
    model_name: str,
    model_factory: Callable,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    results: Dict,
    trained_models: Dict,
):
    """
    Train a model using provided factory, evaluate it,
    and store results in provided dictionaries.

    Parameters
    ----------
    model_name : str
        Display name of the model.
    model_factory : Callable
        Function returning an unfitted model instance.
    X_train : pd.DataFrame
    y_train : pd.Series
    X_test : pd.DataFrame
    y_test : pd.Series
    results : Dict
        Dictionary to store evaluation metrics.
    trained_models : Dict
        Dictionary to store fitted model instances.

    Returns
    -------
    trained_model
        Fitted model instance.

    Notes
    -----
    - Pure training + evaluation helper.
    - No logging.
    - No artifact persistence.
    - Keeps pipelines thin and orchestration-only.
    """

    if not callable(model_factory):
        raise TypeError("model_factory must be callable.")

    model = model_factory()

    trained_model = train_model(model, X_train, y_train)

    if not hasattr(trained_model, "predict"):
        raise TypeError("Trained model must implement predict().")

    predictions = trained_model.predict(X_test)

    metrics = evaluate_regression_model(y_test, predictions)

    results[model_name] = metrics
    trained_models[model_name] = trained_model

    return trained_model
