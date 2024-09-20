import json
import logging
import os
from typing import Dict, Any
import sys
from logging.handlers import RotatingFileHandler, WatchedFileHandler

logger = logging.getLogger(__name__)
PERFORMANCE_CALCULATION_INTERVAL = 144  # 하루에 한 번 성능 계산
SHARPE_RATIO_RISK_FREE_RATE = 0.02  # 연간 2%의 무위험 수익률 가정


def load_config(file_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    default_config = {
        'min_krw_balance': 10000,
        'min_transaction_amount': 5000,
        'max_trade_amount': 1000000,
        'database_path': 'crypto_data.db',  # 데이터베이스 경로 추가
        'min_trade_amount': 5000,  # 최소 거래 금액 (KRW)
        'max_trade_ratio': 0.99,  # 최대 거래 비율 (총 자산의 99%까지 사용)
        'fee_rate': 0.0005,  # 거래 수수료율 (0.05%)
        "discord_webhook_url": "https://discord.com/api/webhooks/1215950067895238719/3XkFH-ZoZ0GkW00kgbKS5evUGW3hmaL8a6S093BEBnf_X9anMQRYlT4jc_Ywz25e_BzY",
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
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 로그 포맷 설정
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, date_format)

    # 콘솔 핸들러 (UTF-8 인코딩 사용)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.stream.reconfigure(encoding='utf-8')

    # 파일 핸들러 (UTF-8 인코딩 사용)
    file_handler = RotatingFileHandler('trading_bot.log', maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 기존 핸들러 제거 및 새 핸들러 추가
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # 다른 라이브러리의 로그 레벨 조정
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('prophet').setLevel(logging.WARNING)
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

    return root_logger


# 로깅 설정 함수 호출
logger = setup_logging()


# 로깅 에러 핸들러 추가
def handle_logging_error(exc_type, exc_value, exc_traceback):
    if isinstance(exc_value, logging.Handler.handleError):
        # 로깅 오류 발생 시 콘솔에 출력
        print(f"Logging error occurred: {exc_value}")
    else:
        # 기본 예외 처리
        sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = handle_logging_error


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