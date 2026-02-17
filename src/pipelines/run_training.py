# src/pipelines/run_training.py

"""
Training Pipeline Orchestration
================================

Enterprise-grade modeling workflow for IDS.

Capabilities:
-------------
- Strict time-based split (leakage safe)
- Base model training (config-driven)
- Dynamic Top-N hyperparameter tuning
- Stacking ensemble (Top-K)
- Primary metric: MAPE (%)
- Timestamped model versioning
- Feature importance export (tree models)
- SHAP explainability (delegated to explainability_SHAP.py)
- Experiment metadata snapshot
- Production-safe logging
"""

import os
import json
import joblib
import pandas as pd
from datetime import datetime
from typing import Dict

from utils.config_loader import load_config
from utils.logger import get_logger
from utils.helpers import ensure_directory

from modeling03.data_preparation import prepare_model_data
from modeling03.baseline_models import (
    get_linear_regression_model,
    get_ridge_model,
)
from modeling03.tree_models import (
    get_random_forest_model,
    get_xgboost_model,
    get_lightgbm_model,
)
from modeling03.prophet_model import ProphetRegressor
from modeling03.evaluate import (
    evaluate_regression_model,
    build_metrics_dataframe,
)
from modeling03.hyperparameter_tuning import tune_model
from modeling03.ensemble import (
    train_stacking_model,
    predict_stacking_model,
)
from modeling03.explainability_SHAP import generate_shap_explainability

from utils.model_utils import train_model


# ==========================================================
# Model - Config Mapping
# ==========================================================

MODEL_CONFIG_MAP = {
    "LinearRegression": "linear_regression",
    "Ridge": "ridge",
    "RandomForest": "random_forest",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "Prophet": "prophet",
}


# ==========================================================
# Time-Based Split
# ==========================================================

def time_based_split(df: pd.DataFrame, config: Dict):
    """
    Perform strict chronological split.

    Prevents data leakage by ensuring test data
    is always future relative to training data.
    """

    date_col = config["data_schema"]["sales"]["date_column"]
    test_days = config["modeling"]["train_test_split"]["test_days"]

    df = df.sort_values(by=date_col)

    if test_days >= len(df):
        raise ValueError("Configured test_days exceeds dataset size.")

    split_index = len(df) - test_days

    return df.iloc[:split_index], df.iloc[split_index:]


# ==========================================================
# Main Training Orchestrator
# ==========================================================

def run_training():

    config = load_config()
    logger = get_logger(config)

    logger.info("Starting training pipeline.")

    # --------------------------------------------------
    # Load Data
    # --------------------------------------------------

    data_path = os.path.join(
        config["paths"]["data"]["processed"],
        "model_ready.parquet"
    )

    if not os.path.exists(data_path):
        raise FileNotFoundError("model_ready.parquet not found.")

    df = pd.read_parquet(data_path)

    train_df, test_df = time_based_split(df, config)

    date_col = config["data_schema"]["sales"]["date_column"]

    X_train, y_train = prepare_model_data(
        train_df, config, drop_columns=(date_col,)
    )

    X_test, y_test = prepare_model_data(
        test_df, config, drop_columns=(date_col,)
    )

    results = {}
    trained_models = {}

    # ==================================================
    # Base Model Training
    # ==================================================

    def _train_and_evaluate(model_name: str, model):
        trained = train_model(model, X_train, y_train)
        preds = trained.predict(X_test)

        results[model_name] = evaluate_regression_model(y_test, preds)
        trained_models[model_name] = trained

    # Linear
    if config["modeling"]["models"]["linear_regression"]["enabled"]:
        _train_and_evaluate("LinearRegression", get_linear_regression_model())

    if config["modeling"]["models"]["ridge"]["enabled"]:
        _train_and_evaluate("Ridge", get_ridge_model(config))

    # Trees
    if config["modeling"]["models"]["random_forest"]["enabled"]:
        _train_and_evaluate("RandomForest", get_random_forest_model(config))

    if config["modeling"]["models"]["xgboost"]["enabled"]:
        _train_and_evaluate("XGBoost", get_xgboost_model(config))

    if config["modeling"]["models"]["lightgbm"]["enabled"]:
        _train_and_evaluate("LightGBM", get_lightgbm_model(config))

    # Prophet
    if config["modeling"]["models"]["prophet"]["enabled"]:
        prophet_model = ProphetRegressor(config)
        prophet_model.fit(train_df)
        preds = prophet_model.predict(test_df)

        results["Prophet"] = evaluate_regression_model(
            y_test.values, preds
        )
        trained_models["Prophet"] = prophet_model

    metrics_df = build_metrics_dataframe(results)
    metrics_df = metrics_df.sort_values(by="MAPE (%)")

    # ==================================================
    # Hyperparameter Tuning
    # ==================================================

    if config["hyperparameter_tuning"]["enabled"]:

        top_n = config["hyperparameter_tuning"]["tune_top_n"]
        top_models = metrics_df.index[:top_n]

        logger.info(f"Tuning Top {top_n} models: {list(top_models)}")

        for model_name in top_models:

            config_key = MODEL_CONFIG_MAP.get(model_name)

            if config_key not in config["hyperparameter_tuning"]:
                continue

            tuned_model = tune_model(
                trained_models[model_name],
                config["hyperparameter_tuning"][config_key],
                X_train,
                y_train,
                config
            )

            preds = tuned_model.predict(X_test)

            results[f"{model_name}_Tuned"] = \
                evaluate_regression_model(y_test, preds)

            trained_models[f"{model_name}_Tuned"] = tuned_model

        metrics_df = build_metrics_dataframe(results)
        metrics_df = metrics_df.sort_values(by="MAPE (%)")

    # ==================================================
    # Ensemble
    # ==================================================

    if config["ensemble"]["enabled"]:

        top_k = config["ensemble"]["top_k_models"]
        top_models = metrics_df.index[:top_k]

        logger.info(f"Building stacking ensemble using: {list(top_models)}")

        ensemble_base_models = {
            name: trained_models[name] for name in top_models
        }

        meta_model = train_stacking_model(
            ensemble_base_models, X_train, y_train, config
        )

        ensemble_preds = predict_stacking_model(
            ensemble_base_models, meta_model, X_test
        )

        results["StackingEnsemble"] = \
            evaluate_regression_model(y_test, ensemble_preds)

        trained_models["StackingEnsemble"] = meta_model

        metrics_df = build_metrics_dataframe(results)
        metrics_df = metrics_df.sort_values(by="MAPE (%)")

    # ==================================================
    # Final Selection
    # ==================================================

    best_model_name = metrics_df.index[0]
    best_model = trained_models[best_model_name]

    logger.info(f"Best model selected: {best_model_name}")

    # ==================================================
    # Saving Artifacts
    # ==================================================

    model_dir = config["paths"]["output"]["model"]
    metrics_dir = config["paths"]["output"]["metrics"]
    plots_dir = config["paths"]["output"]["plots"]

    ensure_directory(model_dir)
    ensure_directory(metrics_dir)
    ensure_directory(plots_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Save model (versioned + latest alias)
    joblib.dump(
        best_model,
        os.path.join(model_dir, f"best_model_{timestamp}.pkl")
    )

    joblib.dump(
        best_model,
        os.path.join(model_dir, "best_model_latest.pkl")
    )

    metrics_df.to_csv(
        os.path.join(metrics_dir, f"model_comparison_{timestamp}.csv")
    )

    # Feature Importance (Tree models only)
    if hasattr(best_model, "feature_importances_"):

        importance_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance": best_model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        importance_df.to_csv(
            os.path.join(
                model_dir,
                f"feature_importance_{timestamp}.csv"
            ),
            index=False
        )

    # SHAP (delegated to module)
    generate_shap_explainability(
        model=best_model,
        X_train=X_train,
        X_test=X_test,
        model_name=best_model_name,
        output_dir=plots_dir,
        config=config,
        timestamp=timestamp
    )

    # Metadata
    metadata = {
        "timestamp": timestamp,
        "best_model": best_model_name,
        "primary_metric": "MAPE (%)",
        "metric_value": float(metrics_df.iloc[0]["MAPE (%)"]),
        "models_evaluated": list(metrics_df.index),
        "ensemble_enabled": config["ensemble"]["enabled"]
    }

    with open(
        os.path.join(model_dir, f"experiment_metadata_{timestamp}.json"),
        "w"
    ) as f:
        json.dump(metadata, f, indent=4)

    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    run_training()
