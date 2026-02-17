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

    meta_name = config["ensemble"]["meta_model"].lower()

    if meta_name == "ridge":
        return Ridge()
    #elif meta_name == "linear":
    #    return LinearRegression()


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

    stacking_X_test = build_stacking_features(base_models, X_test)

    preds = meta_model.predict(stacking_X_test)

    return preds
