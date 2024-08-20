import pandas as pd
import numpy as np
import sqlite3
from ta.trend import MACD
from ta.momentum import RSIIndicator


def load_data(db_path, symbol, start_date, end_date):
    """SQLite 데이터베이스에서 OHLCV 데이터 로드"""
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT timestamp, open, high, low, close, volume
    FROM crypto_data
    WHERE symbol = '{symbol}'
    AND timestamp BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY timestamp
    """
    df = pd.read_sql_query(query, conn)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    conn.close()
    return df


def add_indicators(df):
    """기술적 지표 추가"""
    df['SMA10'] = df['close'].rolling(window=10).mean()
    df['SMA20'] = df['close'].rolling(window=20).mean()

    macd = MACD(df['close'], window_fast=30, window_slow=60, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    rsi = RSIIndicator(df['close'], window=14)
    df['RSI'] = rsi.rsi()

    return df


def generate_signals(df):
    """매수 신호 생성"""
    df['Signal'] = 0

    golden_cross = (df['SMA10'] > df['SMA20']) & (df['SMA10'].shift(1) <= df['SMA20'].shift(1))
    macd_golden_cross = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
    macd_below_zero = df['MACD'] < 0
    rsi_above_50 = df['RSI'] > 50

    df.loc[golden_cross & macd_golden_cross & macd_below_zero & rsi_above_50, 'Signal'] = 1

    return df


def backtest(df):
    """백테스트 실행"""
    df['Position'] = df['Signal'].diff()
    df['Returns'] = df['close'].pct_change()
    df['Strategy_Returns'] = df['Position'].shift(1) * df['Returns']

    total_return = df['Strategy_Returns'].sum()
    sharpe_ratio = df['Strategy_Returns'].mean() / df['Strategy_Returns'].std() * np.sqrt(365 * 24)  # 시간단위 데이터 가정

    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # 누적 수익률 계산
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    print(f"Final Cumulative Return: {df['Cumulative_Returns'].iloc[-1]:.2f}")


def main():
    db_path = r"C:\Dev\GPT_BTC\pythonProject\gpt-bitcoin\crypto_data.db"
    symbol = 'BTCUSDT'  # 분석하려는 암호화폐 심볼
    start_date = '2023-01-01'  # 시작 날짜
    end_date = '2023-12-31'  # 종료 날짜

    df = load_data(db_path, symbol, start_date, end_date)
    df = add_indicators(df)
    df = generate_signals(df)
    backtest(df)

    # 결과 저장
    df.to_csv('backtest_results.csv')
    print("Results saved to 'backtest_results.csv'")


if __name__ == "__main__":
    main()