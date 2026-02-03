import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """
    Loads and validates the YAML configuration file.

    Args:
        config_path (str): Path to config.yaml

    Returns:
        dict: Parsed configuration dictionary
    """

    config_file = Path(config_path)

    # Check if config file exists
    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found at path: {config_file}"
        )

    # Load YAML safely
    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError("Error parsing YAML file") from exc

    # Basic validation (fail fast)
    required_sections = [
        "data",
        "model",
        "forecasting",
        "features",
        "llm_agent"
    ]

    for section in required_sections:
        if section not in config:
            raise KeyError(
                f"Missing required section '{section}' in config.yaml"
            )

    return config
