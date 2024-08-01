import json
import logging

logger = logging.getLogger(__name__)


def load_config(file_path: str = "config.json") -> dict:
    """Load configuration from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file {file_path} not found.")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Error parsing configuration file {file_path}.")
        return {}