# src/llm05/scenario_cli.py

"""
Scenario CLI Interface
======================

Interactive LLM scenario layer.
"""

import os
import pandas as pd
from datetime import datetime
from utils.config_loader import load_config
from utils.logger import get_logger
from utils.helpers import ensure_directory

from llm05.scenario_parser import parse_prompt
from llm05.scenario_engine import apply_scenario, generate_scenario_id
from llm05.scenario_store import ScenarioStore
from llm05.scenario_guardrails import validate_scenario


def run_cli(baseline_forecast_path: str):

    config = load_config()
    logger = get_logger(config)

    if not os.path.exists(baseline_forecast_path):
        raise FileNotFoundError(
            f"Baseline forecast file not found: {baseline_forecast_path}"
        )

    df = pd.read_csv(baseline_forecast_path)

    if df.empty:
        raise ValueError("Baseline forecast file is empty.")

    store = ScenarioStore(config)

    date_col = config["data_schema"]["sales"]["date_column"]
    target_col = config["data_schema"]["sales"]["target_column"]

    if date_col not in df.columns or target_col not in df.columns:
        raise ValueError(
            f"Baseline forecast must contain columns "
            f"'{date_col}' and '{target_col}'."
        )

    logger.info("Intelligent Demand Sensing — Scenario Mode Started")

    while True:

        user_input = input("Enter prompt: ").strip()

        if user_input.lower() == "exit":
            logger.info("Exiting scenario mode.")
            break

        if user_input.lower() == "history":
            logger.info("Listing stored scenarios.")
            print(store.list_scenarios())
            continue

        if user_input.lower() == "reset":
            logger.info("Scenario state reset.")
            continue

        try:
            parsed = parse_prompt(user_input)
            validate_scenario(parsed)
        except Exception as e:
            logger.warning(f"Guardrail violation: {str(e)}")
            print(f"Guardrail violation: {str(e)}")
            continue

        try:
            existing_ids = store.list_scenarios()
            scenario_id = generate_scenario_id(existing_ids)

            result_df = apply_scenario(df, parsed, date_col, target_col)

        except Exception as e:
            logger.exception(f"Scenario application failed: {str(e)}")
            print(f"Scenario failed: {str(e)}")
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        output_dir = config["paths"]["output"]["scenarios"]
        ensure_directory(output_dir)

        output_path = os.path.join(
            output_dir,
            f"{scenario_id}_{timestamp}.csv"
        )

        result_df.to_csv(output_path, index=False)

        store.save_scenario(
            scenario_id,
            {
                "created_at": datetime.utcnow().isoformat(),
                "prompt": user_input,
                "output_file": output_path
            }
        )

        logger.info(f"Scenario saved: {scenario_id}")
        print(f"Scenario saved: {scenario_id}")
        print(f"Output: {output_path}\n")
