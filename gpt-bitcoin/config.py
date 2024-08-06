import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_config(file_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    default_config = {
        'min_krw_balance': 10000,  # Default minimum KRW balance
        'min_transaction_amount': 5000  # Default minimum transaction amount
    }

    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
            logger.info(f"Configuration loaded successfully from {file_path}.")

            # Update default config with values from the file
            default_config.update(config)

            # Ensure all required keys are present
            for key in default_config:
                if key not in config:
                    logger.warning(f"'{key}' not found in config file. Using default value: {default_config[key]}")

            return default_config
    except FileNotFoundError:
        logger.error(f"Configuration file {file_path} not found. Using default configuration.")
        return default_config
    except json.JSONDecodeError:
        logger.error(f"Error parsing configuration file {file_path}. Using default configuration.")
        return default_config
    except Exception as e:
        logger.error(f"Unexpected error loading configuration file {file_path}: {e}. Using default configuration.")
        return default_config


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("trading_bot.log"),
            logging.StreamHandler()
        ]
    )

    # 특정 모듈의 로그 레벨 조정
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("tweepy").setLevel(logging.WARNING)