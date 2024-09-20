import os
import time
import pyupbit
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import logging

# Logger 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 데이터 저장 디렉토리 설정
DATA_DIR = './data/'

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def fetch_all_historical_data(ticker, interval="minute1"):
    end_date = datetime.now()
    start_date = datetime(2017, 9, 25)  # Upbit 거래소 시작일
    all_data = []
    last_fetched_date = None

    while start_date < end_date:
        df = pyupbit.get_ohlcv(ticker, interval=interval, to=end_date, count=200)
        if df is None or df.empty:
            break

        current_fetched_date = df.index[0]
        if last_fetched_date and current_fetched_date >= last_fetched_date:
            logger.info(f"{ticker}: Reached earliest available data at {current_fetched_date}")
            break

        all_data.append(df)
        last_fetched_date = current_fetched_date
        end_date = current_fetched_date - timedelta(minutes=1)
        logger.info(f"{ticker}: Fetched data until {end_date}")
        time.sleep(0.1)

    return pd.concat(all_data[::-1]) if all_data else pd.DataFrame()

def recreate_ohlcv_data_table(conn):
    with conn:
        conn.execute("DROP TABLE IF EXISTS ohlcv_data")
        conn.execute('''
            CREATE TABLE ohlcv_data (
                date TEXT PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                value REAL
            )
        ''')

def save_to_database(df, db_path):
    conn = sqlite3.connect(db_path)
    recreate_ohlcv_data_table(conn)

    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)
    df['value'] = df['close'] * df['volume']
    df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))

    insert_query = '''
        INSERT OR REPLACE INTO ohlcv_data (date, open, high, low, close, volume, value)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    '''
    data_tuples = [tuple(row) for row in df.to_numpy()]

    conn.executemany(insert_query, data_tuples)
    conn.commit()
    conn.close()

def fetch_and_save_btc():
    ensure_dir(DATA_DIR)
    ticker = "KRW-BTC"  # BTC 페어만 처리
    logger.info(f"Fetching data for {ticker}...")
    historical_data = fetch_all_historical_data(ticker)

    if not historical_data.empty:
        db_path = os.path.join(DATA_DIR, f"{ticker.replace('-', '_')}.db")
        logger.info(f"Saving data for {ticker} to {db_path}...")
        save_to_database(historical_data, db_path)
        logger.info(f"Completed saving data for {ticker}.")
    else:
        logger.error(f"No data was fetched for {ticker}. Skipping...")

def main():
    logger.info("Starting to fetch and save data for BTC on Upbit...")
    fetch_and_save_btc()
    logger.info("BTC data fetching and saving completed.")

if __name__ == "__main__":
    main()
