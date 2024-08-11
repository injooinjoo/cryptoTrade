import json
import logging
from typing import Dict, Any
import sys
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
PERFORMANCE_CALCULATION_INTERVAL = 144  # 하루에 한 번 성능 계산
SHARPE_RATIO_RISK_FREE_RATE = 0.02  # 연간 2%의 무위험 수익률 가정

def load_config(file_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    default_config = {
        'min_krw_balance': 10000,
        'min_transaction_amount': 5000,
        'database_path': 'crypto_data.db',  # 데이터베이스 경로 추가
        'trading_parameters': {
            'buy_threshold': 0.01,
            'sell_threshold': 0.01,
            # 다른 필요한 매개변수들...
        }
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

    # Custom formatter with color support
    formatter = CustomFormatter()

    file_handler = RotatingFileHandler(
        "trading_bot.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("tweepy").setLevel(logging.WARNING)


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and custom formats"""

    # Define the color codes
    COLORS = {
        'HEADER': '\033[95m',
        'OKBLUE': '\033[94m',
        'OKCYAN': '\033[96m',
        'OKGREEN': '\033[92m',
        'WARNING': '\033[93m',
        'FAIL': '\033[91m',
        'ENDC': '\033[0m',
    }

    # Define formats for different log levels
    FORMATS = {
        logging.DEBUG: "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        logging.INFO: "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        logging.WARNING: COLORS['WARNING'] + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + COLORS['ENDC'],
        logging.ERROR: COLORS['FAIL'] + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + COLORS['ENDC'],
        logging.CRITICAL: COLORS['FAIL'] + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + COLORS['ENDC'],
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)