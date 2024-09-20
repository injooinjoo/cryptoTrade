# fetch_data.py

import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_ohlcv_data(exchange_id='binance', symbol='BTC/USDT', timeframe='5m', start_date=None, end_date=None):
    exchange = getattr(ccxt, exchange_id)()
    exchange.load_markets()

    if start_date is None:
        start_date = exchange.parse8601('2020-01-01T00:00:00Z')  # 더 과거로 설정하여 데이터 범위를 넓힘
    else:
        start_date = exchange.parse8601(start_date)

    if end_date is None:
        end_date = exchange.milliseconds()
    else:
        end_date = exchange.parse8601(end_date)

    all_ohlcv = []
    limit = 1000  # Most exchanges have a limit of 1000 candles per request

    while start_date < end_date:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, start_date, limit)
            if len(ohlcv) == 0:
                print("No data fetched in this period.")
                break

            all_ohlcv += ohlcv
            start_date = ohlcv[-1][0] + 1
            print(f"Fetched data up to {datetime.fromtimestamp(start_date/1000)}")
            time.sleep(exchange.rateLimit / 1000)  # Respect the rate limit
        except ccxt.NetworkError as e:
            print(f'Fetch OHLCV failed due to a network error: {e}')
            time.sleep(60)  # Wait for 1 minute before retrying
            continue
        except ccxt.ExchangeError as e:
            print(f'Fetch OHLCV failed due to an exchange error: {e}')
            break

    if len(all_ohlcv) == 0:
        print("No data fetched at all.")
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    print(f"Fetched {len(df)} data points from {exchange_id} for {symbol} in {timeframe} timeframe")
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    return df


def update_data(filename='crypto_data.csv', exchange_id='binance', symbol='BTC/USDT', timeframe='5m'):
    try:
        existing_data = pd.read_csv(filename, index_col='timestamp', parse_dates=True)
        last_date = existing_data.index[-1]
        start_date = (last_date + timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%SZ')
    except FileNotFoundError:
        existing_data = pd.DataFrame()
        # 시작 날짜를 명확하게 설정 (예시: 2020년 1월 1일)
        start_date = '2020-01-01T00:00:00Z'

    new_data = fetch_ohlcv_data(exchange_id, symbol, timeframe, start_date)

    if not existing_data.empty:
        updated_data = pd.concat([existing_data, new_data])
        updated_data = updated_data[~updated_data.index.duplicated(keep='last')]
    else:
        updated_data = new_data

    updated_data.to_csv(filename)
    print(f"Data updated and saved to {filename}")


if __name__ == "__main__":
    update_data()