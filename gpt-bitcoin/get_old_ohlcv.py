import time
import pyupbit
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import logging

# Logger 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_all_historical_data(ticker="KRW-BTC", interval="minute10"):
    end_date = datetime.now()
    start_date = datetime(2017, 9, 25)  # Upbit 거래소 시작일
    # 17년 9월 25일
    all_data = []
    last_fetched_date = None

    while start_date < end_date:
        df = pyupbit.get_ohlcv(ticker, interval=interval, to=end_date, count=200)
        if df is None or df.empty:
            break

        current_fetched_date = df.index[0]
        if last_fetched_date and current_fetched_date >= last_fetched_date:
            logger.info(f"Reached earliest available data at {current_fetched_date}")
            break

        all_data.append(df)
        last_fetched_date = current_fetched_date
        end_date = current_fetched_date - timedelta(minutes=1)
        logger.info(f"Fetched data until {end_date}")
        time.sleep(0.1)

    return pd.concat(all_data[::-1]) if all_data else pd.DataFrame()


def recreate_ohlcv_data_table(conn):
    with conn:
        conn.execute("DROP TABLE IF EXISTS ohlcv_data")
        logger.info("Dropped existing ohlcv_data table if it existed.")

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
        logger.info("Recreated ohlcv_data table with the correct schema.")


def save_to_database(df, db_path='crypto_data.db'):
    conn = sqlite3.connect(db_path)
    recreate_ohlcv_data_table(conn)

    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)
    df['value'] = df['close'] * df['volume']

    # 날짜를 문자열로 변환 (ISO 형식)
    df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))

    # 데이터 삽입을 위한 UPSERT SQL 구문 작성
    insert_query = '''
        INSERT OR REPLACE INTO ohlcv_data (date, open, high, low, close, volume, value)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    '''
    data_tuples = [tuple(row) for row in df.to_numpy()]

    conn.executemany(insert_query, data_tuples)
    conn.commit()
    conn.close()
    logger.info("Data has been saved to the database.")


def main():
    logger.info("Fetching all historical data...")
    historical_data = fetch_all_historical_data()

    if not historical_data.empty:
        logger.info("Saving data to database...")
        save_to_database(historical_data)
        logger.info("Data fetching and saving completed.")
    else:
        logger.error("No data was fetched. Check your connection or API limits.")


if __name__ == "__main__":
    main()