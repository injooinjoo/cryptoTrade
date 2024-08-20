import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_data_from_db(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.sort_index()
    logging.info(f"{db_path} 데이터 로드 완료. 행 수: {len(df)}")
    return df


def calculate_indicators(df):
    df['SMA10'] = df['close'].rolling(window=10).mean()
    df['SMA20'] = df['close'].rolling(window=20).mean()

    exp1 = df['close'].ewm(span=30, adjust=False).mean()
    exp2 = df['close'].ewm(span=60, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df


def backtest_strategy(df):
    df['Buy_Signal'] = (
            (df['SMA10'] > df['SMA20']) &
            (df['SMA10'].shift(1) <= df['SMA20'].shift(1)) &
            (df['MACD'] > df['Signal']) &
            (df['MACD'].shift(1) <= df['Signal'].shift(1)) &
            (df['MACD'] < 0) &
            (df['RSI'] > 50)
    )

    df['Position'] = 0
    df['Entry_Price'] = 0.0
    df['Exit_Price'] = 0.0
    df['Trade_Return'] = 0.0

    for i in range(len(df)):
        if df['Buy_Signal'].iloc[i]:
            if i + 3 < len(df):
                df['Position'].iloc[i:i + 3] = 1
                df['Entry_Price'].iloc[i] = df['close'].iloc[i]
                df['Exit_Price'].iloc[i + 2] = df['close'].iloc[i + 2]
                df['Trade_Return'].iloc[i + 2] = (df['Exit_Price'].iloc[i + 2] / df['Entry_Price'].iloc[i]) - 1

    df['Strategy_Returns'] = df['Position'].shift(1) * df['close'].pct_change()

    return df


def run_backtest(db_path):
    df = get_data_from_db(db_path, 'ohlcv_data')
    df = calculate_indicators(df)
    df = backtest_strategy(df)

    total_return = (1 + df['Strategy_Returns']).prod() - 1
    buy_hold_return = (1 + df['close'].pct_change()).prod() - 1
    sharpe_ratio = np.sqrt(252 * 24 * 20) * df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()

    return {
        'db_file': os.path.basename(db_path),
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'sharpe_ratio': sharpe_ratio,
        'trade_count': df['Buy_Signal'].sum()
    }


# 메인 실행 코드
data_dir = './data/'
db_files = [f for f in os.listdir(data_dir) if f.endswith('.db')]

results = []
total_strategy_return = 0
total_buy_hold_return = 0

for db_file in db_files:
    db_path = os.path.join(data_dir, db_file)
    result = run_backtest(db_path)
    results.append(result)
    total_strategy_return += result['total_return']
    total_buy_hold_return += result['buy_hold_return']

    logging.info(f"파일: {result['db_file']}")
    logging.info(f"  전략 수익률: {result['total_return']:.2f}")
    logging.info(f"  Buy and Hold 수익률: {result['buy_hold_return']:.2f}")
    logging.info(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    logging.info(f"  거래 횟수: {result['trade_count']}")
    logging.info("--------------------")

logging.info(f"\n총 전략 수익률: {total_strategy_return:.2f}")
logging.info(f"총 Buy and Hold 수익률: {total_buy_hold_return:.2f}")

# 결과를 데이터프레임으로 변환하여 CSV로 저장
results_df = pd.DataFrame(results)
results_df.to_csv('backtest_results.csv', index=False)
logging.info("백테스트 결과가 'backtest_results.csv' 파일로 저장되었습니다.")

# 전략 수익률과 Buy and Hold 수익률 비교 그래프
plt.figure(figsize=(12, 6))
plt.bar(range(len(results)), [r['total_return'] for r in results], alpha=0.5, label='Strategy')
plt.bar(range(len(results)), [r['buy_hold_return'] for r in results], alpha=0.5, label='Buy and Hold')
plt.xlabel('DB Files')
plt.ylabel('Returns')
plt.title('Strategy vs Buy and Hold Returns')
plt.legend()
plt.xticks(range(len(results)), [r['db_file'] for r in results], rotation=90)
plt.tight_layout()
plt.savefig('returns_comparison.png')
logging.info("수익률 비교 그래프가 'returns_comparison.png' 파일로 저장되었습니다.")