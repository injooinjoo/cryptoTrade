# import os
#
# from dotenv import load_dotenv
#
# load_dotenv()
# import pyupbit
# import pandas as pd
# import pandas_ta as ta
# import json
# from openai import OpenAI
# import schedule
# import time
# from datetime import datetime
# import sqlite3
#
# # Setup
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS_KEY"), os.getenv("UPBIT_SECRET_KEY"))
#
#
# def initialize_db(db_path='trading_decisions.sqlite'):
#     with sqlite3.connect(db_path) as conn:
#         cursor = conn.cursor()
#         cursor.execute('''
#             CREATE TABLE IF NOT EXISTS decisions (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 timestamp DATETIME,
#                 decision TEXT,
#                 percentage REAL,
#                 btc_balance REAL,
#                 krw_balance REAL,
#                 btc_avg_buy_price REAL,
#                 btc_krw_price REAL
#             );
#         ''')
#         conn.commit()
#
#
# def save_decision_to_db(decision, current_status):
#     db_path = 'trading_decisions.sqlite'
#     with sqlite3.connect(db_path) as conn:
#         cursor = conn.cursor()
#
#         # Parsing current_status from JSON to Python dict
#         status_dict = json.loads(current_status)
#         current_price = pyupbit.get_orderbook(ticker="KRW-BTC")['orderbook_units'][0]["ask_price"]
#
#         # Preparing data for insertion
#         data_to_insert = (
#             decision.get('decision'),
#             decision.get('percentage', 100),  # Defaulting to 100 if not provided
#             status_dict.get('btc_balance'),
#             status_dict.get('krw_balance'),
#             status_dict.get('btc_avg_buy_price'),
#             current_price
#         )
#
#         # Inserting data into the database
#         cursor.execute('''
#             INSERT INTO decisions (timestamp, decision, percentage, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price)
#             VALUES (datetime('now', 'localtime'), ?, ?, ?, ?, ?, ?)
#         ''', data_to_insert)
#
#         conn.commit()
#
#
# def fetch_last_decisions(db_path='trading_decisions.sqlite', num_decisions=10):
#     with sqlite3.connect(db_path) as conn:
#         cursor = conn.cursor()
#         cursor.execute('''
#             SELECT timestamp, decision, percentage, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price
#             FROM decisions
#             ORDER BY timestamp DESC
#             LIMIT ?
#         ''', (num_decisions,))
#         decisions = cursor.fetchall()
#
#         if decisions:
#             formatted_decisions = []
#             for decision in decisions:
#                 # Converting timestamp to milliseconds since the Unix epoch
#                 ts = datetime.strptime(decision[0], "%Y-%m-%d %H:%M:%S")
#                 ts_millis = int(ts.timestamp() * 1000)
#
#                 formatted_decision = {
#                     "timestamp": ts_millis,
#                     "decision": decision[1],
#                     "percentage": decision[2],
#                     "btc_balance": decision[3],
#                     "krw_balance": decision[4],
#                     "btc_avg_buy_price": decision[5],
#                     "btc_krw_price": decision[6]
#                 }
#                 formatted_decisions.append(str(formatted_decision))
#             return "\n".join(formatted_decisions)
#         else:
#             return "No decisions found."
#
#
# def get_current_status():
#     orderbook = pyupbit.get_orderbook(ticker="KRW-BTC")
#     current_time = orderbook['timestamp']
#     btc_balance = 0
#     krw_balance = 0
#     btc_avg_buy_price = 0
#     balances = upbit.get_balances()
#     print(balances)
#     for b in balances:
#         if b['currency'] == "BTC":
#             btc_balance = b['balance']
#             btc_avg_buy_price = b['avg_buy_price']
#         if b['currency'] == "KRW":
#             krw_balance = b['balance']
#
#     current_status = {'current_time': current_time, 'orderbook': orderbook, 'btc_balance': btc_balance, 'krw_balance': krw_balance, 'btc_avg_buy_price': btc_avg_buy_price}
#     return json.dumps(current_status)
#
#
# def fetch_and_prepare_data():
#     # Fetch data
#     df_daily = pyupbit.get_ohlcv("KRW-BTC", "day", count=30)
#     df_hourly = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=24)
#
#     # Define a helper function to add indicators
#     def add_indicators(df):
#         # Moving Averages
#         df['SMA_10'] = ta.sma(df['close'], length=10)
#         df['EMA_10'] = ta.ema(df['close'], length=10)
#
#         # RSI
#         df['RSI_14'] = ta.rsi(df['close'], length=14)
#
#         # Stochastic Oscillator
#         stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
#         df = df.join(stoch)
#
#         # MACD
#         ema_fast = df['close'].ewm(span=12, adjust=False).mean()
#         ema_slow = df['close'].ewm(span=26, adjust=False).mean()
#         df['MACD'] = ema_fast - ema_slow
#         df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
#         df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
#
#         # Bollinger Bands
#         df['Middle_Band'] = df['close'].rolling(window=20).mean()
#         # Calculate the standard deviation of closing prices over the last 20 days
#         std_dev = df['close'].rolling(window=20).std()
#         # Calculate the upper band (Middle Band + 2 * Standard Deviation)
#         df['Upper_Band'] = df['Middle_Band'] + (std_dev * 2)
#         # Calculate the lower band (Middle Band - 2 * Standard Deviation)
#         df['Lower_Band'] = df['Middle_Band'] - (std_dev * 2)
#
#         return df
#
#     # Add indicators to both dataframes
#     df_daily = add_indicators(df_daily)
#     df_hourly = add_indicators(df_hourly)
#
#     combined_df = pd.concat([df_daily, df_hourly], keys=['daily', 'hourly'])
#     combined_data = combined_df.to_json(orient='split')
#
#     return json.dumps(combined_data)
#
#
# def get_instructions(file_path):
#     try:
#         with open(file_path, "r", encoding="utf-8") as file:
#             instructions = file.read()
#         return instructions
#     except FileNotFoundError:
#         print("File not found.")
#     except Exception as e:
#         print("An error occurred while reading the file:", e)
#
#
# def analyze_data_with_gpt4(data_json, last_decisions, current_status):
#     instructions_path = "instructions_v4.md"
#     try:
#         instructions = get_instructions(instructions_path)
#         if not instructions:
#             print("No instructions found.")
#             return None
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": instructions},
#                 # {"role": "user", "content": news_data},
#                 {"role": "user", "content": data_json},
#                 {"role": "user", "content": last_decisions},
#                 # {"role": "user", "content": fear_and_greed},
#                 {"role": "user", "content": current_status},
#                 # {"role": "user", "content": [{"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{current_base64_image}"}}]}
#             ],
#             response_format={"type": "json_object"}
#         )
#         advice = response.choices[0].message.content
#         return advice
#     except Exception as e:
#         print(f"Error in analyzing data with GPT-4: {e}")
#         return None
#
#
# def execute_buy(percentage):
#     print("Attempting to buy BTC with a percentage of KRW balance...")
#     try:
#         krw_balance = upbit.get_balance("KRW")
#         amount_to_invest = krw_balance * (percentage / 100)
#         if amount_to_invest > 5000:  # Ensure the order is above the minimum threshold
#             result = upbit.buy_market_order("KRW-BTC", amount_to_invest * 0.9995)  # Adjust for fees
#             print("Buy order successful:", result)
#     except Exception as e:
#         print(f"Failed to execute buy order: {e}")
#
#
# def execute_sell(percentage):
#     print("Attempting to sell a percentage of BTC...")
#     try:
#         btc_balance = upbit.get_balance("BTC")
#         amount_to_sell = btc_balance * (percentage / 100)
#         current_price = pyupbit.get_orderbook(ticker="KRW-BTC")['orderbook_units'][0]["ask_price"]
#         if current_price * amount_to_sell > 5000:  # Ensure the order is above the minimum threshold
#             result = upbit.sell_market_order("KRW-BTC", amount_to_sell)
#             print("Sell order successful:", result)
#     except Exception as e:
#         print(f"Failed to execute sell order: {e}")
#
#
# def make_decision_and_execute():
#     print("Making decision and executing...")
#     try:
#         # news_data = get_news_data()
#         data_json = fetch_and_prepare_data()
#         last_decisions = fetch_last_decisions()
#         # fear_and_greed = fetch_fear_and_greed_index(limit=30)
#         current_status = get_current_status()
#         # current_base64_image = get_current_base64_image()
#     except Exception as e:
#         print(f"Error: {e}")
#     else:
#         advice = analyze_data_with_gpt4(data_json, last_decisions, current_status)
#         decision = {}
#         if '"decision": "hold"' in advice:
#             decision['decision'] = "hold"
#
#         elif '"decision": "buy"' in advice:
#             decision['decision'] = "buy"
#
#         elif '"decision": "sell"' in advice:
#             decision['decision'] = "sell"
#         else:
#             raise Exception("Invalid decision")
#
#         decision['percentage'] = int(advice[advice.find("percentage") + 12:advice.find(",", advice.find("percentage"))])
#         decision['reason'] = advice[advice.find("reason") + 9:advice.find("}", advice.find("reason"))]
#
#         print(decision)
#
#         if not len(decision):
#             print("Failed to make a decision after maximum retries.")
#             return
#         else:
#             try:
#                 percentage = decision.get('percentage', 100)
#
#                 if decision.get('decision') == "buy":
#                     execute_buy(percentage)
#                 elif decision.get('decision') == "sell":
#                     execute_sell(percentage)
#
#                 save_decision_to_db(decision, current_status)
#             except Exception as e:
#                 print(f"Failed to execute the decision or save to DB: {e}")
#
#
# if __name__ == "__main__":
#     initialize_db()
#     make_decision_and_execute()
#     schedule.every().day.at("00:01").do(make_decision_and_execute)
#
#     # Schedule the task to run at 08:01
#     schedule.every().day.at("08:01").do(make_decision_and_execute)
#
#     # Schedule the task to run at 16:01
#     schedule.every().day.at("16:01").do(make_decision_and_execute)
#
#     while True:
#         schedule.run_pending()
#         time.sleep(1)


import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List

import schedule
import pyupbit
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from openai import OpenAI
import sqlite3

# Load environment variables
load_dotenv()

# Constants
DB_PATH = 'trading_decisions.sqlite'
INSTRUCTIONS_PATH = "instructions_v4.md"
DECISION_TIMES = ["00:01", "08:01", "16:01"]
BTC_TICKER = "KRW-BTC"

# Setup clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS_KEY"), os.getenv("UPBIT_SECRET_KEY"))


def initialize_db(db_path: str = DB_PATH) -> None:
    """Initialize the SQLite database."""
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
        conn.commit()


def save_decision_to_db(decision: Dict[str, Any], current_status: str, db_path: str = DB_PATH) -> None:
    """Save the trading decision to the database."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        status_dict = json.loads(current_status)
        current_price = pyupbit.get_orderbook(ticker=BTC_TICKER)['orderbook_units'][0]["ask_price"]

        data_to_insert = (
            decision.get('decision'),
            decision.get('percentage', 100),
            status_dict.get('btc_balance'),
            status_dict.get('krw_balance'),
            status_dict.get('btc_avg_buy_price'),
            current_price
        )

        cursor.execute('''
            INSERT INTO decisions (timestamp, decision, percentage, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price)
            VALUES (datetime('now', 'localtime'), ?, ?, ?, ?, ?, ?)
        ''', data_to_insert)
        conn.commit()


def fetch_last_decisions(db_path: str = DB_PATH, num_decisions: int = 10) -> str:
    """Fetch the last n decisions from the database."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, decision, percentage, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price
            FROM decisions
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (num_decisions,))
        decisions = cursor.fetchall()

        if decisions:
            formatted_decisions = []
            for decision in decisions:
                ts = datetime.strptime(decision[0], "%Y-%m-%d %H:%M:%S")
                ts_millis = int(ts.timestamp() * 1000)
                formatted_decision = {
                    "timestamp": ts_millis,
                    "decision": decision[1],
                    "percentage": decision[2],
                    "btc_balance": decision[3],
                    "krw_balance": decision[4],
                    "btc_avg_buy_price": decision[5],
                    "btc_krw_price": decision[6]
                }
                formatted_decisions.append(str(formatted_decision))
            return "\n".join(formatted_decisions)
        else:
            return "No decisions found."


def get_current_status() -> str:
    """Get the current trading status."""
    orderbook = pyupbit.get_orderbook(ticker=BTC_TICKER)
    current_time = orderbook['timestamp']
    btc_balance = krw_balance = btc_avg_buy_price = 0
    balances = upbit.get_balances()

    for b in balances:
        if b['currency'] == "BTC":
            btc_balance = b['balance']
            btc_avg_buy_price = b['avg_buy_price']
        elif b['currency'] == "KRW":
            krw_balance = b['balance']

    current_status = {
        'current_time': current_time,
        'orderbook': orderbook,
        'btc_balance': btc_balance,
        'krw_balance': krw_balance,
        'btc_avg_buy_price': btc_avg_buy_price
    }
    return json.dumps(current_status)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataframe."""
    df['SMA_10'] = ta.sma(df['close'], length=10)
    df['EMA_10'] = ta.ema(df['close'], length=10)
    df['RSI_14'] = ta.rsi(df['close'], length=14)
    df = df.join(ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3))

    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']

    df['Middle_Band'] = df['close'].rolling(window=20).mean()
    std_dev = df['close'].rolling(window=20).std()
    df['Upper_Band'] = df['Middle_Band'] + (std_dev * 2)
    df['Lower_Band'] = df['Middle_Band'] - (std_dev * 2)

    return df


def fetch_and_prepare_data() -> str:
    """Fetch and prepare data for analysis."""
    df_daily = pyupbit.get_ohlcv(BTC_TICKER, "day", count=30)
    df_hourly = pyupbit.get_ohlcv(BTC_TICKER, interval="minute60", count=24)

    df_daily = add_indicators(df_daily)
    df_hourly = add_indicators(df_hourly)

    combined_df = pd.concat([df_daily, df_hourly], keys=['daily', 'hourly'])
    combined_data = combined_df.to_json(orient='split')

    return json.dumps(combined_data)


def get_instructions(file_path: str) -> str:
    """Read instructions from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("An error occurred while reading the file:", e)
    return ""


def analyze_data_with_gpt4(data_json: str, last_decisions: str, current_status: str) -> str:
    """Analyze data using GPT-4."""
    instructions = get_instructions(INSTRUCTIONS_PATH)
    if not instructions:
        print("No instructions found.")
        return None

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": data_json},
                {"role": "user", "content": last_decisions},
                {"role": "user", "content": current_status},
            ],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in analyzing data with gpt-4o-mini: {e}")
        return None


def execute_buy(percentage: float) -> None:
    """Execute a buy order."""
    try:
        krw_balance = upbit.get_balance("KRW")
        amount_to_invest = krw_balance * (percentage / 100)
        if amount_to_invest > 5000:
            result = upbit.buy_market_order(BTC_TICKER, amount_to_invest * 0.9995)
            print("Buy order successful:", result)
    except Exception as e:
        print(f"Failed to execute buy order: {e}")


def execute_sell(percentage: float) -> None:
    """Execute a sell order."""
    try:
        btc_balance = upbit.get_balance("BTC")
        amount_to_sell = btc_balance * (percentage / 100)
        current_price = pyupbit.get_orderbook(ticker=BTC_TICKER)['orderbook_units'][0]["ask_price"]
        if current_price * amount_to_sell > 5000:
            result = upbit.sell_market_order(BTC_TICKER, amount_to_sell)
            print("Sell order successful:", result)
    except Exception as e:
        print(f"Failed to execute sell order: {e}")


def parse_decision(advice: str) -> Dict[str, Any]:
    """Parse the decision from the GPT-4 advice."""
    decision = {}
    if '"decision": "hold"' in advice:
        decision['decision'] = "hold"
    elif '"decision": "buy"' in advice:
        decision['decision'] = "buy"
    elif '"decision": "sell"' in advice:
        decision['decision'] = "sell"
    else:
        raise ValueError("Invalid decision")

    decision['percentage'] = int(advice[advice.find("percentage") + 12:advice.find(",", advice.find("percentage"))])
    decision['reason'] = advice[advice.find("reason") + 9:advice.find("}", advice.find("reason"))]

    return decision


def make_decision_and_execute() -> None:
    """Make a trading decision and execute it."""
    print("Making decision and executing...")
    try:
        data_json = fetch_and_prepare_data()
        last_decisions = fetch_last_decisions()
        current_status = get_current_status()

        advice = analyze_data_with_gpt4(data_json, last_decisions, current_status)
        decision = parse_decision(advice)

        print(decision)

        if decision:
            percentage = decision.get('percentage', 100)
            if decision['decision'] == "buy":
                execute_buy(percentage)
            elif decision['decision'] == "sell":
                execute_sell(percentage)

            save_decision_to_db(decision, current_status)
        else:
            print("Failed to make a decision.")
    except Exception as e:
        print(f"Error in make_decision_and_execute: {e}")


def main() -> None:
    """Main function to run the trading bot."""
    initialize_db()
    make_decision_and_execute()

    # Schedule the task to run every 2 hours at 1 minute past the hour
    for hour in range(0, 24, 2):
        schedule_time = f"{hour:02d}:01"
        schedule.every().day.at(schedule_time).do(make_decision_and_execute)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()