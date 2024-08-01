import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List
import logging

import schedule
import pyupbit
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from openai import OpenAI
import sqlite3

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
DB_PATH = 'trading_decisions.sqlite'
BTC_TICKER = "KRW-BTC"
CONFIG_FILE = "config.json"

# Setup OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS_KEY"), os.getenv("UPBIT_SECRET_KEY"))


def load_config(file_path: str = CONFIG_FILE) -> Dict[str, Any]:
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


config = load_config()


def initialize_db(db_path: str = DB_PATH) -> None:
    """Initialize the SQLite database and create necessary tables."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                decision TEXT,
                percentage REAL,
                btc_balance REAL,
                krw_balance REAL,
                btc_avg_buy_price REAL,
                btc_krw_price REAL
            );
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_parameters (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                value REAL
            );
        ''')
        conn.commit()
    logger.info("Database initialized successfully.")


def update_parameter(name: str, value: float, db_path: str = DB_PATH) -> None:
    """Insert or update a trading parameter in the database."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trading_parameters (name, value)
            VALUES (?, ?)
            ON CONFLICT(name) DO UPDATE SET value = excluded.value;
        ''', (name, value))
        conn.commit()
    logger.info(f"Parameter {name} updated to {value}")


def get_parameter(name: str, db_path: str = DB_PATH) -> float:
    """Retrieve a trading parameter value from the database."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT value FROM trading_parameters WHERE name = ?
        ''', (name,))
        result = cursor.fetchone()
        return result[0] if result else None

def fetch_ohlcv_with_retry(max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            df = pyupbit.get_ohlcv(BTC_TICKER, interval="minute60", count=24)
            if df is not None and not df.empty:
                logger.info(f"Successfully fetched OHLCV data on attempt {attempt + 1}")
                return df
            logger.warning(f"Attempt {attempt + 1}: Failed to fetch OHLCV data. Retrying in {delay} seconds...")
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}: Error fetching OHLCV data: {e}")
        time.sleep(delay)
    logger.error(f"Failed to fetch OHLCV data after {max_retries} attempts.")
    return None

def calculate_macd(close_prices, fast=12, slow=26, signal=9):
    exp1 = close_prices.ewm(span=fast, adjust=False).mean()
    exp2 = close_prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({'MACD': macd, 'Signal': signal_line})


def fetch_and_prepare_data() -> pd.DataFrame:
    """Fetch and prepare data for analysis."""
    try:
        df = fetch_ohlcv_with_retry()
        if df is None or df.empty:
            logger.error(f"Failed to fetch OHLCV data for {BTC_TICKER}. Returned DataFrame is None or empty.")
            return pd.DataFrame()

        logger.info(f"Successfully fetched OHLCV data. Shape: {df.shape}")

        # Add technical indicators with error checking
        try:
            df['SMA_10'] = ta.sma(df['close'], length=10)
            df['EMA_10'] = ta.ema(df['close'], length=10)
            df['RSI_14'] = ta.rsi(df['close'], length=14)
            logger.info("SMA, EMA, and RSI calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating SMA, EMA, or RSI: {e}")

        try:
            stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
            df['Stochastic_%K'] = stoch['STOCHk_14_3_3']
            df['Stochastic_%D'] = stoch['STOCHd_14_3_3']
            logger.info("Stochastic oscillator calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating Stochastic oscillator: {e}")

        try:
            macd_data = calculate_macd(df['close'])
            df['MACD'] = macd_data['MACD']
            df['MACD_Signal'] = macd_data['Signal']
            logger.info("MACD calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            df['MACD'] = 0
            df['MACD_Signal'] = 0

        try:
            bollinger = ta.bbands(df['close'], length=20, std=2)
            df['Upper_Band'] = bollinger['BBU_20_2.0']
            df['Middle_Band'] = bollinger['BBM_20_2.0']
            df['Lower_Band'] = bollinger['BBL_20_2.0']
            logger.info("Bollinger Bands calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")

        try:
            df['Market_Sentiment'] = (df['close'] - df['close'].rolling(window=40).mean()) / df['close'].rolling(window=40).std()
            df['Price_Divergence'] = (df['close'] - df['SMA_10']) / df['SMA_10']
            logger.info("Market Sentiment and Price Divergence calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating Market Sentiment or Price Divergence: {e}")

        logger.info(f"Final DataFrame shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error in fetch_and_prepare_data: {e}")
        logger.exception("Traceback:")
        return pd.DataFrame()


def get_instructions(file_path: str) -> str:
    """Read instructions from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"Instructions file {file_path} not found.")
    except Exception as e:
        logger.error(f"An error occurred while reading the instructions file: {e}")
    return ""


def analyze_data_with_gpt4(data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze data using GPT-4 and return a decision."""
    instructions = get_instructions(config.get("instructions_file", "instructions_v5.md"))
    data_json = data.to_json(orient='split')

    latest_data = data.iloc[-1]
    logger.info(
        f"Latest data point: Close: {latest_data['close']}, RSI: {latest_data['RSI_14']}, Stochastic %K: {latest_data['Stochastic_%K']}, Stochastic %D: {latest_data['Stochastic_%D']}, MACD: {latest_data['MACD']}, MACD Signal: {latest_data['MACD_Signal']}")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": data_json}
            ],
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        advice = response.choices[0].message.content
        logger.info(f"GPT-4 advice: {advice}")
        return parse_decision(advice)
    except Exception as e:
        logger.error(f"Error in gpt-4o-mini analysis: {e}")
        return {"decision": "hold", "percentage": 0, "reason": "Error in analysis", "target_price": None}


def parse_decision(advice: str) -> Dict[str, Any]:
    """Parse the decision from the GPT-4 advice, including target prices for buy/sell actions."""
    try:
        decision = json.loads(advice)
        return {
            "decision": decision.get("decision", "hold"),
            "percentage": float(decision.get("percentage", 0)),
            "reason": decision.get("reason", ""),
            "target_price": float(decision.get("target_price", 0))
        }
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing GPT-4 advice: {e}. Using default 'hold' decision.")
        return {"decision": "hold", "percentage": 0, "reason": "Error parsing advice", "target_price": None}


def cancel_existing_orders() -> None:
    """Cancel all existing orders."""
    try:
        orders = upbit.get_order(BTC_TICKER)  # Changed from get_open_orders to get_order
        for order in orders:
            upbit.cancel_order(order['uuid'])
        logger.info("All existing orders cancelled successfully.")
    except Exception as e:
        logger.error(f"Failed to cancel existing orders: {e}")


def execute_buy(percentage: float, target_price: float) -> None:
    """Execute a buy limit order at a specified target price."""
    try:
        krw_balance = upbit.get_balance("KRW")

        if krw_balance < 5000:
            logger.info(f"Buy order not placed: KRW balance ({krw_balance}) is less than 5000 KRW")
            return

        amount_to_invest = krw_balance * (percentage / 100)
        target_price = float(target_price)
        amount_to_buy = amount_to_invest / target_price

        logger.info(
            f"Buy order details: KRW balance: {krw_balance}, Amount to invest: {amount_to_invest}, Target price: {target_price}, Amount to buy: {amount_to_buy}")

        min_order_size = 0.00001  # 10,000 satoshis, adjust this based on Upbit's requirements
        if amount_to_buy < min_order_size:
            logger.info(
                f"Buy order not placed: Amount to buy ({amount_to_buy} BTC) is below the minimum order size ({min_order_size} BTC)")
            return

        if amount_to_invest > config.get("min_transaction_amount", 5000):
            result = upbit.buy_limit_order(BTC_TICKER, target_price, amount_to_buy)
            logger.info(f"Buy limit order placed successfully: {result}")
        else:
            logger.info(
                f"Buy order not placed: Amount to invest ({amount_to_invest} KRW) is below the minimum transaction amount ({config.get('min_transaction_amount', 5000)} KRW)")
    except Exception as e:
        logger.error(f"Failed to execute buy limit order: {e}")
        logger.exception("Traceback:")


def execute_sell(percentage: float, target_price: float) -> None:
    """Execute a sell limit order at a specified target price."""
    try:
        btc_balance = upbit.get_balance("BTC")
        amount_to_sell = btc_balance * (percentage / 100)
        target_price = float(target_price)  # Ensure target_price is a float
        if amount_to_sell * target_price > config.get("min_transaction_amount", 5000):
            result = upbit.sell_limit_order(BTC_TICKER, target_price, amount_to_sell)
            logger.info(f"Sell limit order placed successfully: {result}")
        else:
            logger.info(f"Sell order not placed: Amount too small. BTC balance: {btc_balance}, Amount to sell: {amount_to_sell}, Target price: {target_price}")
    except Exception as e:
        logger.error(f"Failed to execute sell limit order: {e}")
        logger.exception("Traceback:")


def make_decision_and_execute() -> None:
    """Make a trading decision and execute it based on dynamic parameters."""
    logger.info("Making decision and executing...")
    cancel_existing_orders()

    data = fetch_and_prepare_data()
    if data.empty:
        logger.error("No data available for analysis. Skipping this iteration.")
        return

    try:
        decision = analyze_data_with_gpt4(data)
        logger.info(f"Decision: {decision}")

        krw_balance = upbit.get_balance("KRW")

        if decision['decision'] == "buy":
            if krw_balance < 5000:
                logger.info(f"Skipping buy decision: KRW balance ({krw_balance}) is less than 5000 KRW")
            else:
                execute_buy(decision['percentage'], decision['target_price'])
        elif decision['decision'] == "sell":
            execute_sell(decision['percentage'], decision['target_price'])
        else:
            logger.info("Holding based on advice")

        # Log the decision
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO decisions (timestamp, decision, percentage, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                decision['decision'],
                decision['percentage'],
                upbit.get_balance("BTC"),
                krw_balance,
                upbit.get_avg_buy_price("BTC"),
                pyupbit.get_current_price(BTC_TICKER)
            ))
            conn.commit()
    except Exception as e:
        logger.error(f"Error in make_decision_and_execute: {e}")
        logger.exception("Traceback:")


def check_upbit_connection():
    try:
        balance = upbit.get_balance("KRW")
        logger.info(f"Successfully connected to Upbit. KRW balance: {balance}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Upbit API: {e}")
        return False


def main() -> None:
    """Main function to run the trading bot."""
    if not os.getenv("UPBIT_ACCESS_KEY") or not os.getenv("UPBIT_SECRET_KEY"):
        logger.error("Upbit API keys are missing. Please check your .env file.")
        return

    if not check_upbit_connection():
        logger.error("Failed to connect to Upbit. Please check your API keys and internet connection.")
        return

    initialize_db()

    # Initialize parameters from config
    for param, value in config.get("trading_parameters", {}).items():
        update_parameter(param, value)

    make_decision_and_execute()

    # Schedule the task to run every 2 hours at 1 minute past the hour
    for hour in range(0, 24, 2):
        schedule_time = f"{hour:02d}:01"
        schedule.every().day.at(schedule_time).do(make_decision_and_execute)

    logger.info("Trading bot started. Running on schedule.")
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()