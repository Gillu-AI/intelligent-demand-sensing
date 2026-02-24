# src/pipelines/run_training.py

"""
Training Pipeline Orchestration
================================

Enterprise-grade modeling workflow.

Responsibilities:
- Load model_ready.parquet only
- Enforce strict time-based split
- Train enabled models
- Optional hyperparameter tuning
- Optional stacking ensemble
- Deterministic multi-metric model ranking
- Persist versioned artifacts safely
- Generate SHAP (config-driven)
- Save experiment metadata
"""

import os
import json
import random
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict

from utils.config_loader import load_config
from utils.logger import get_logger
from utils.helpers import ensure_directory
from utils.model_utils import train_and_evaluate

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


# ============================================================
# VALIDATION & UTILS
# ============================================================

def enforce_global_seed(config: Dict, logger):
    seed_cfg = config.get("seeds")
    if seed_cfg is None or "global_seed" not in seed_cfg:
        raise ValueError("Missing 'seeds.global_seed' configuration.")

    seed = seed_cfg["global_seed"]

    if not isinstance(seed, int):
        raise ValueError("'seeds.global_seed' must be an integer.")

    random.seed(seed)
    np.random.seed(seed)

    logger.info(f"Global random seed set to {seed}.")


def validate_execution_policy(config: Dict):
    execution_cfg = config.get("execution")
    if execution_cfg is None:
        raise ValueError("Missing 'execution' configuration block.")

    mode = execution_cfg.get("mode")
    overwrite = execution_cfg.get("overwrite_existing_outputs")

    allowed_modes = {"dev", "train", "prod", "backfill"}

    if mode not in allowed_modes:
        raise ValueError(f"Invalid execution.mode '{mode}'. Allowed: {allowed_modes}")

    if not isinstance(overwrite, bool):
        raise ValueError("'execution.overwrite_existing_outputs' must be boolean.")

    if mode == "prod":
        raise RuntimeError("Training is not allowed in PROD mode.")

    return mode, overwrite


def deterministic_rank(metrics_df: pd.DataFrame) -> pd.DataFrame:
    primary = "MAPE (%)"
    secondary = "RMSE"
    tertiary = "MAE"

    for metric in [primary, secondary, tertiary]:
        if metric not in metrics_df.columns:
            raise ValueError(f"Required metric '{metric}' missing from metrics DataFrame.")

    metrics_df = metrics_df.sort_values(
        by=[primary, secondary, tertiary],
        ascending=[True, True, True],
    )

    metrics_df = metrics_df.sort_index(kind="mergesort")

    return metrics_df


def validate_output_paths(config: Dict):
    paths = config.get("paths", {}).get("output")
    if paths is None:
        raise ValueError("Missing 'paths.output' configuration block.")

    for key in ["model", "metrics", "plots"]:
        if key not in paths:
            raise ValueError(f"Missing 'paths.output.{key}' configuration.")


def validate_shap_config(config: Dict):
    shap_cfg = config.get("shap", {})
    if shap_cfg.get("enabled"):
        sample_size = shap_cfg.get("sample_size")
        if not isinstance(sample_size, int) or sample_size <= 0:
            raise ValueError("'shap.sample_size' must be positive integer.")


def time_based_split(df: pd.DataFrame, config: Dict):
    date_col = config["data_schema"]["sales"]["date_column"]
    test_days = config["modeling"]["train_test_split"]["test_days"]

    df = df.sort_values(by=date_col)

    if not isinstance(test_days, int) or test_days <= 0:
        raise ValueError("'modeling.train_test_split.test_days' must be positive integer.")

    if test_days >= len(df):
        raise ValueError(
            f"Configured test_days ({test_days}) exceeds dataset size ({len(df)})."
        )

    split_index = len(df) - test_days
    return df.iloc[:split_index], df.iloc[split_index:]


def _load_latest_model_ready(processed_path: str) -> pd.DataFrame:
    file_path = os.path.join(processed_path, "model_ready.parquet")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "model_ready.parquet not found. Run feature pipeline first."
        )

    return pd.read_parquet(file_path)


def get_model_factory(model_key: str, config: Dict):

    model_factory_map = {
        "linear_regression": get_linear_regression_model,
        "ridge": lambda: get_ridge_model(config),
        "random_forest": lambda: get_random_forest_model(config),
        "xgboost": lambda: get_xgboost_model(config),
        "lightgbm": lambda: get_lightgbm_model(config),
        "prophet": lambda: ProphetRegressor(config),
    }

    return model_factory_map.get(model_key)


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def run_training():

    config = load_config("config/config.yaml")
    logger = get_logger(config)

    logger.info("Starting training pipeline.")

    enforce_global_seed(config, logger)
    mode, overwrite = validate_execution_policy(config)
    validate_output_paths(config)
    validate_shap_config(config)

    hyper_cfg = config.get("hyperparameter_tuning")
    ensemble_cfg = config.get("ensemble")

    if hyper_cfg is None:
        raise ValueError("Missing 'hyperparameter_tuning' block.")

    if ensemble_cfg is None:
        raise ValueError("Missing 'ensemble' block.")

    processed_path = config["paths"]["data"]["processed"]
    df = _load_latest_model_ready(processed_path)

    logger.info(f"Total rows in model_ready: {len(df)}")

    train_df, test_df = time_based_split(df, config)

    logger.info(f"Train rows: {len(train_df)}")
    logger.info(f"Test rows: {len(test_df)}")

    date_col = config["data_schema"]["sales"]["date_column"]

    X_train, y_train = prepare_model_data(
        train_df, config, drop_columns=(date_col,)
    )
    X_test, y_test = prepare_model_data(
        test_df, config, drop_columns=(date_col,)
    )

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"Feature columns: {list(X_train.columns)}")
    logger.info(f"Unique values per feature:\n{X_train.nunique()}")

    # Guardrail: no feature collapse allowed
    if X_train.shape[1] == 0:
        raise RuntimeError(
            "Feature collapse detected: X_train has 0 columns after preparation."
        )

    if X_train.shape[0] < 10:
        logger.warning(
            "Very small training dataset detected (<10 rows). "
            "Tree-based models may fail to split."
        )

    results = {}
    trained_models = {}

    modeling_cfg = config["modeling"]["models"]

    for model_name, model_config in modeling_cfg.items():
        if model_config.get("enabled", False):

            model_factory = get_model_factory(model_name, config)

            if model_factory is None:
                raise ValueError(
                    f"No factory defined for model '{model_name}'."
                )

            train_and_evaluate(
                model_name,
                model_factory,
                X_train,
                y_train,
                X_test,
                y_test,
                results,
                trained_models,
            )

    if not results:
        raise ValueError("No models were enabled for training.")

    metrics_df = deterministic_rank(
        build_metrics_dataframe(results)
    )

    # ========================================================
    # HYPERPARAMETER TUNING
    # ========================================================

    if hyper_cfg.get("enabled"):

        top_n = hyper_cfg.get("tune_top_n")

        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("'hyperparameter_tuning.tune_top_n' must be positive integer.")

        top_models = metrics_df.index[:top_n]

        for model_name in top_models:

            tuning_cfg = hyper_cfg.get(model_name)
            if not tuning_cfg:
                continue

            fresh_model = get_model_factory(model_name, config)()

            tuned_model = tune_model(
                fresh_model,
                tuning_cfg,
                X_train,
                y_train,
                config,
            )

            preds = tuned_model.predict(X_test)

            results[f"{model_name}_tuned"] = evaluate_regression_model(
                y_test, preds
            )

            trained_models[f"{model_name}_tuned"] = tuned_model

        metrics_df = deterministic_rank(
            build_metrics_dataframe(results)
        )

    # ========================================================
    # ENSEMBLE
    # ========================================================

    if ensemble_cfg.get("enabled"):

        top_k = ensemble_cfg.get("top_k_models")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("'ensemble.top_k_models' must be positive integer.")

        top_models = metrics_df.index[:top_k]

        ensemble_base_models = {
            name: trained_models[name] for name in top_models
        }

        meta_model = train_stacking_model(
            ensemble_base_models, X_train, y_train, config
        )

        ensemble_preds = predict_stacking_model(
            ensemble_base_models, meta_model, X_test
        )

        results["stacking_ensemble"] = evaluate_regression_model(
            y_test, ensemble_preds
        )

        trained_models["stacking_ensemble"] = {
            "base_models": ensemble_base_models,
            "meta_model": meta_model,
        }

        metrics_df = deterministic_rank(
            build_metrics_dataframe(results)
        )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    best_model_name = metrics_df.index[0]
    best_model = trained_models[best_model_name]

    model_dir = config["paths"]["output"]["model"]
    metrics_dir = config["paths"]["output"]["metrics"]
    plots_dir = config["paths"]["output"]["plots"]

    ensure_directory(model_dir)
    ensure_directory(metrics_dir)
    ensure_directory(plots_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    latest_path = os.path.join(model_dir, "best_model_latest.pkl")

    if os.path.exists(latest_path) and not overwrite:
        raise FileExistsError(
            "best_model_latest.pkl exists and overwrite is disabled."
        )

    joblib.dump(best_model, os.path.join(model_dir, f"best_model_{timestamp}.pkl"))
    joblib.dump(best_model, latest_path)

    metrics_df.to_csv(
        os.path.join(metrics_dir, f"model_comparison_{timestamp}.csv")
    )

    if config.get("shap", {}).get("enabled") and hasattr(
        best_model, "predict"
    ) and not isinstance(best_model, dict):

        generate_shap_explainability(
            model=best_model,
            X_train=X_train,
            X_test=X_test,
            model_name=best_model_name,
            output_dir=plots_dir,
            config=config,
            timestamp=timestamp,
        )

    metadata = {
        "timestamp": timestamp,
        "best_model": best_model_name,
        "primary_metric": "MAPE (%)",
        "metric_value": float(metrics_df.iloc[0]["MAPE (%)"]),
        "models_evaluated": list(metrics_df.index),
        "execution_mode": mode,
    }

    with open(
        os.path.join(model_dir, f"experiment_metadata_{timestamp}.json"),
        "w",
    ) as f:
        json.dump(metadata, f, indent=4)

    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    run_training()
