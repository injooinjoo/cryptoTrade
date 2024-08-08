import json
import logging
from typing import Dict, Any
import sys
from logging.handlers import RotatingFileHandler

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
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 파일 핸들러 설정 (UTF-8 인코딩 사용)
    file_handler = RotatingFileHandler(
        "trading_bot.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # 콘솔 핸들러 설정 (ASCII 문자만 사용)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 특정 모듈의 로그 레벨 조정
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("tweepy").setLevel(logging.WARNING)