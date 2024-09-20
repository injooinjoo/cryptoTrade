import numpy as np
import pandas as pd
import pandas_ta as ta
from tqdm import tqdm

# Constants
BUY = 1
SELL = -1
CLEAR = 0

# Minimax normalization
def minimax(x, p, min_val, max_val):
    hi = x.rolling(window=p).max()
    lo = x.rolling(window=p).min()
    return (max_val - min_val) * (x - lo) / (hi - lo) + min_val


def calculate_indicators(df, short_window, long_window):
    print("Calculating technical indicators...")
    df['RSI_Long'] = ta.rsi(df['Close'], length=long_window)
    df['RSI_Short'] = ta.rsi(df['Close'], length=short_window)
    df['CCI_Long'] = ta.cci(df['High'], df['Low'], df['Close'], length=long_window)
    df['CCI_Short'] = ta.cci(df['High'], df['Low'], df['Close'], length=short_window)
    df['ROC_Long'] = ta.roc(df['Close'], length=long_window)
    df['ROC_Short'] = ta.roc(df['Close'], length=short_window)
    df['Volume_Long'] = minimax(df['Volume'], long_window, 0, 99)
    df['Volume_Short'] = minimax(df['Volume'], short_window, 0, 99)
    print("Technical indicators calculated.")
    return df


def get_features(df, indicator):
    rs, rf = df['RSI_Long'], df['RSI_Short']
    cs, cf = df['CCI_Long'], df['CCI_Short']
    os, of = df['ROC_Long'], df['ROC_Short']
    vs, vf = df['Volume_Long'], df['Volume_Short']

    if indicator == 'RSI':
        return rs, rf
    elif indicator == 'CCI':
        return cs, cf
    elif indicator == 'ROC':
        return os, of
    elif indicator == 'Volume':
        return vs, vf
    else:  # 'All'
        return df[['RSI_Long', 'CCI_Long', 'ROC_Long', 'Volume_Long']].mean(axis=1), \
            df[['RSI_Short', 'CCI_Short', 'ROC_Short', 'Volume_Short']].mean(axis=1)

# Main kNN strategy
def run_knn_strategy(df, start_date, stop_date, indicator='All', short_window=14, long_window=28, base_k=252,
                     filter_volatility=False, bars=300):
    print("Starting KNN strategy...")
    feature1, feature2, directions = [], [], []
    predictions = []
    bars_count = 0
    signal = CLEAR
    k = int(np.floor(np.sqrt(base_k)))

    df = calculate_indicators(df, short_window, long_window)
    df_range = df.loc[start_date:stop_date]

    buy_signals = pd.Series(np.nan, index=df.index)
    sell_signals = pd.Series(np.nan, index=df.index)
    clear_signals = pd.Series(np.nan, index=df.index)

    prev_signal = CLEAR

    print("Processing data points...")
    # tqdm is used to show a progress bar
    for i in tqdm(range(len(df_range))):
    # for i in range(len(df_range)):
        f1, f2 = get_features(df_range, indicator)
        class_label = int(np.sign(df_range['Close'].iloc[i] - df_range['Close'].iloc[i-1])) if i > 0 else 0

        if i >= long_window:
            feature1.append(f1.iloc[i])
            feature2.append(f2.iloc[i])
            directions.append(class_label)

        size = len(directions)
        maxdist = -999.0

        for j in range(size):
            d = np.sqrt((f1.iloc[i] - feature1[j])**2 + (f2.iloc[i] - feature2[j])**2)
            if d > maxdist:
                maxdist = d
                if len(predictions) >= k:
                    predictions.pop(0)
                predictions.append(directions[j])

        prediction = sum(predictions)

        filter_condition = True
        if filter_volatility:
            atr_short = ta.atr(df_range['High'], df_range['Low'], df_range['Close'], length=10)
            atr_long = ta.atr(df_range['High'], df_range['Low'], df_range['Close'], length=40)
            filter_condition = atr_short.iloc[i] > atr_long.iloc[i]

        long = prediction > 0 and filter_condition
        short = prediction < 0 and filter_condition
        clear = not (long or short)

        if bars_count == bars:
            signal = CLEAR
            bars_count = 0
        else:
            bars_count += 1

        signal = BUY if long else SELL if short else CLEAR if clear else prev_signal

        changed = signal != prev_signal
        start_long_trade = changed and signal == BUY
        start_short_trade = changed and signal == SELL
        clear_condition = changed and signal == CLEAR

        maxpos = df_range['High'].rolling(window=10).max().iloc[i]
        minpos = df_range['Low'].rolling(window=10).min().iloc[i]

        if start_long_trade:
            buy_signals.iloc[i] = minpos
        elif start_short_trade:
            sell_signals.iloc[i] = maxpos
        elif clear_condition:
            clear_signals.iloc[i] = df_range['Close'].iloc[i]

        prev_signal = signal

    df['Buy_Signal'] = buy_signals
    df['Sell_Signal'] = sell_signals
    df['Clear_Signal'] = clear_signals

    print("KNN strategy completed.")
    return df

# 사용 예시:
# df = pd.read_csv('crypto_data.csv', index_col='timestamp', parse_dates=True)
# start_date = pd.Timestamp('2000-01-01 00:00:00')
# stop_date = pd.Timestamp('2025-12-31 23:45:00')
# result_df = run_knn_strategy(df, start_date, stop_date)