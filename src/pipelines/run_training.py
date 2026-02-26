"""
================================================================================
ENTERPRISE TRAINING PIPELINE — MODE B INDUSTRIAL IMPLEMENTATION (AUDIT HARDENED)
================================================================================

Authoritative industrial-grade training orchestration layer.

Enhancements:
- Deterministic CV
- Leakage-safe OOF stacking
- Tuned-hyperparameter-safe stacking (Option A)
- Naive baselines (Mean + Last Value)
- Data shift diagnostics
- Negative R² warning layer
- Proper ranking fix
- Monitoring-ready metrics persistence
- SHAP governance
- Deterministic outputs

This file is production-audit ready.
================================================================================
"""

import os
import json
import random
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Callable
from tabulate import tabulate

from utils.config_loader import load_config
from utils.logger import get_logger
from utils.helpers import ensure_directory

from modeling03.data_preparation import prepare_model_data
from modeling03.evaluate import evaluate_regression_model
from modeling03.hyperparameter_tuning import tune_model
from modeling03.ensemble import train_stacking_model, predict_stacking_model
from modeling03.explainability_SHAP import generate_shap_explainability

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


# =============================================================================
# GLOBAL CONTROL
# =============================================================================

def enforce_global_seed(config: Dict) -> int:
    seed = config["seeds"]["global_seed"]
    random.seed(seed)
    np.random.seed(seed)
    return seed


def validate_execution_mode(config: Dict):
    mode = config["execution"]["mode"]
    overwrite = config["execution"]["overwrite_existing_outputs"]

    if mode not in {"dev", "train", "prod", "backfill"}:
        raise ValueError(f"Invalid execution.mode '{mode}'.")

    return mode, overwrite


# =============================================================================
# DATA VALIDATION
# =============================================================================

def load_model_ready(config: Dict) -> pd.DataFrame:
    path = os.path.join(
        config["paths"]["data"]["processed"],
        "model_ready.parquet"
    )

    if not os.path.exists(path):
        raise FileNotFoundError("model_ready.parquet missing.")

    df = pd.read_parquet(path)

    if df.empty:
        raise ValueError("model_ready.parquet is empty.")

    return df


def validate_dataset(df: pd.DataFrame, config: Dict):
    schema = config["data_schema"]["sales"]
    date_col = schema["date_column"]
    target_col = schema["target_column"]

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise TypeError("Date column must be datetime dtype.")

    if df[target_col].isna().any():
        raise ValueError("Target contains NaN.")

    if df[date_col].duplicated().any():
        raise ValueError("Duplicate timestamps detected.")

    return date_col, target_col


def time_split(df: pd.DataFrame, config: Dict, date_col: str):
    test_days = config["modeling"]["train_test_split"]["test_days"]

    df = df.sort_values(date_col).reset_index(drop=True)

    if test_days >= len(df):
        raise ValueError("test_days must be smaller than dataset length.")

    split_idx = len(df) - test_days

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    if train_df[date_col].max() >= test_df[date_col].min():
        raise RuntimeError("Temporal leakage detected.")

    return train_df, test_df


# =============================================================================
# MODEL FACTORY
# =============================================================================

def get_model_factory(model_key: str, config: Dict) -> Callable:
    return {
        "linear_regression": get_linear_regression_model,
        "ridge": lambda: get_ridge_model(config),
        "random_forest": lambda: get_random_forest_model(config),
        "xgboost": lambda: get_xgboost_model(config),
        "lightgbm": lambda: get_lightgbm_model(config),
        "prophet": lambda: ProphetRegressor(config),
    }.get(model_key)


def build_tuned_factory(base_key: str, best_params: Dict, config: Dict) -> Callable:
    """
    Build a fresh model factory using tuned hyperparameters.
    Prevents leakage by avoiding reuse of fitted objects.
    """

    def factory():
        base_factory = get_model_factory(base_key, config)
        model = base_factory()
        model.set_params(**best_params)
        return model

    return factory


# =============================================================================
# RANKING
# =============================================================================

def rank_models(results: Dict, config: Dict):
    ranking_cfg = config["modeling"]["ranking"]
    metric = ranking_cfg["primary_metric"]
    higher_is_better = ranking_cfg["higher_is_better"]

    return sorted(
        results.items(),
        key=lambda x: x[1][metric],
        reverse=higher_is_better
    )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_training():

    config = load_config("config/config.yaml")
    logger = get_logger(config)
    logger.info("ENTERPRISE TRAINING STARTED")

    enforce_global_seed(config)
    mode, overwrite = validate_execution_mode(config)

    df = load_model_ready(config)
    date_col, target_col = validate_dataset(df, config)
    train_df, test_df = time_split(df, config, date_col)

    X_train, y_train = prepare_model_data(train_df, config, drop_columns=(date_col,))
    X_test, y_test = prepare_model_data(test_df, config, drop_columns=(date_col,))

    results = {}
    trained_models = {}
    tuned_param_registry = {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # -------------------------------------------------------------------------
    # DATA SHIFT DIAGNOSTICS
    # -------------------------------------------------------------------------

    train_mean = float(y_train.mean())
    test_mean = float(y_test.mean())
    train_std = float(y_train.std())
    test_std = float(y_test.std())

    mean_shift_pct = ((test_mean - train_mean) / train_mean) * 100 if train_mean != 0 else 0
    std_shift_pct = ((test_std - train_std) / train_std) * 100 if train_std != 0 else 0

    print("\nTEST SET DIAGNOSTICS")
    print(f"Train Mean: {train_mean}")
    print(f"Test Mean : {test_mean}")
    print(f"Mean Shift %: {mean_shift_pct:.2f}%")
    print(f"Train Std: {train_std}")
    print(f"Test Std : {test_std}")
    print(f"Std Shift %: {std_shift_pct:.2f}%")

    # -------------------------------------------------------------------------
    # NAIVE BASELINES
    # -------------------------------------------------------------------------

    mean_pred = np.full_like(y_test, fill_value=train_mean)
    last_value = float(y_train.iloc[-1])
    last_pred = np.full_like(y_test, fill_value=last_value)

    results["baseline_mean"] = evaluate_regression_model(y_test, mean_pred)
    results["baseline_last_value"] = evaluate_regression_model(y_test, last_pred)

    # -------------------------------------------------------------------------
    # BASE MODELS
    # -------------------------------------------------------------------------

    for model_name, model_cfg in config["modeling"]["models"].items():

        if not model_cfg.get("enabled"):
            continue

        factory = get_model_factory(model_name, config)
        model = factory()

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = evaluate_regression_model(y_test, preds)
        labeled = f"{model_name}-base"

        results[labeled] = metrics
        trained_models[labeled] = model

    # -------------------------------------------------------------------------
    # HYPERPARAMETER TUNING
    # -------------------------------------------------------------------------

    if config["hyperparameter_tuning"]["enabled"]:

        ranked_base = rank_models(results, config)
        top_n = config["hyperparameter_tuning"]["tune_top_n"]

        for name, _ in ranked_base[:top_n]:

            if "-base" not in name:
                continue

            base_key = name.replace("-base", "")
            space = config["hyperparameter_tuning"].get(base_key)

            if not space:
                continue

            base_model = get_model_factory(base_key, config)()

            tuned_model = tune_model(
                base_model,
                space,
                X_train,
                y_train,
                config,
                base_key,
                timestamp
            )

            # SAFELY STORE ONLY BEST HYPERPARAMETERS
            tuned_param_registry[base_key] = tuned_model.get_params(deep=False)

            preds = tuned_model.predict(X_test)
            metrics = evaluate_regression_model(y_test, preds)

            labeled = f"{base_key}-tuned"
            results[labeled] = metrics
            trained_models[labeled] = tuned_model

    # -------------------------------------------------------------------------
    # OOF STACKING (WITH FILTER + SAFETY)
    # -------------------------------------------------------------------------

    if config["ensemble"]["enabled"]:

        ranked_all = rank_models(results, config)
        top_k = config["ensemble"]["top_k_models"]

        valid_models = [
            name for name, _ in ranked_all
            if (
                ("-base" in name or "-tuned" in name)
                and not name.startswith("baseline_")
                and name != "stacking_ensemble"
            )
        ]

        selected_models = valid_models[:top_k]

        if not selected_models:
            logger.warning("No valid models available for stacking. Skipping ensemble.")
        else:
            top_factories = {}

            for name in selected_models:
                base_key = name.split("-")[0]

                if base_key in tuned_param_registry:
                    factory = build_tuned_factory(
                        base_key,
                        tuned_param_registry[base_key],
                        config
                    )
                else:
                    factory = get_model_factory(base_key, config)

                if factory is None:
                    logger.warning(f"No factory found for model '{name}'. Skipping.")
                    continue

                top_factories[name] = factory

            if top_factories:
                stacking_bundle = train_stacking_model(
                    top_factories,
                    X_train,
                    y_train,
                    config
                )

                preds = predict_stacking_model(stacking_bundle, X_test)
                metrics = evaluate_regression_model(y_test, preds)

                results["stacking_ensemble"] = metrics
                trained_models["stacking_ensemble"] = stacking_bundle

    # -------------------------------------------------------------------------
    # FINAL RANKING
    # -------------------------------------------------------------------------

    ranked_models = rank_models(results, config)

    ranked_rows = []
    for rank, (model_name, metrics) in enumerate(ranked_models, start=1):
        row = metrics.copy()
        row["Model"] = model_name
        row["Rank"] = rank
        ranked_rows.append(row)

    ranking_df = pd.DataFrame(ranked_rows)

    print("\n================ MODEL RANKING SUMMARY ================\n")
    print(tabulate(ranking_df, headers="keys", tablefmt="grid"))

    best_model_name = ranked_models[0][0]
    best_metrics = results[best_model_name]

    # -------------------------------------------------------------------------
    # WARNINGS
    # -------------------------------------------------------------------------

    if best_metrics["R2"] < 0:
        logger.warning("Best model has negative R². Performance worse than mean baseline.")

    if best_model_name not in {"baseline_mean", "baseline_last_value"}:
        if best_metrics["MAE"] > results["baseline_mean"]["MAE"]:
            logger.warning("Best model performs worse than mean baseline.")

    # -------------------------------------------------------------------------
    # ARTIFACT GOVERNANCE
    # -------------------------------------------------------------------------

    model_dir = config["paths"]["output"]["model"]
    metrics_dir = config["paths"]["output"]["metrics"]

    ensure_directory(model_dir)
    ensure_directory(metrics_dir)

    if mode == "prod":
        joblib.dump(
            trained_models[best_model_name],
            os.path.join(model_dir, f"model_{timestamp}.pkl")
        )
    else:
        for name, model in trained_models.items():
            joblib.dump(
                model,
                os.path.join(model_dir, f"{name}_{timestamp}.pkl")
            )

    monitoring_payload = {
        "best_model": best_model_name,
        "best_metrics": best_metrics,
        "all_results": results,
        "baselines": {
            "mean": results["baseline_mean"],
            "last_value": results["baseline_last_value"]
        },
        "data_shift": {
            "train_mean": train_mean,
            "test_mean": test_mean,
            "mean_shift_pct": mean_shift_pct,
            "train_std": train_std,
            "test_std": test_std,
            "std_shift_pct": std_shift_pct
        }
    }

    with open(os.path.join(metrics_dir, f"metrics_full_{timestamp}.json"), "w") as f:
        json.dump(monitoring_payload, f, indent=4)

    ranking_df.to_csv(
        os.path.join(metrics_dir, f"metrics_full_{timestamp}.csv"),
        index=False
    )

    # -------------------------------------------------------------------------
    # SHAP GOVERNANCE
    # -------------------------------------------------------------------------

    best_model = trained_models.get(best_model_name)

    if best_model and not isinstance(best_model, dict):
        generate_shap_explainability(
            best_model,
            X_train,
            X_test,
            best_model_name,
            config["paths"]["output"]["plots"],
            config,
            timestamp
        )

    logger.info("Training completed successfully.")
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"Timestamp: {timestamp}")


if __name__ == "__main__":
    run_training()