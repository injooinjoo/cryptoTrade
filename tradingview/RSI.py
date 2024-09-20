# RSI.py

import pandas as pd
import numpy as np


class RSI:
    def __init__(self, period=14, overbought=70, oversold=30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def calculate(self, df):
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        return df

    def generate_signals(self, df):
        signals = []
        for i in range(len(df)):
            if i < self.period:
                signals.append('HOLD')
                continue

            if df['RSI'].iloc[i] > self.overbought:
                signals.append('SELL')
            elif df['RSI'].iloc[i] < self.oversold:
                signals.append('BUY')
            else:
                signals.append('HOLD')

        return signals


def run_rsi_strategy(df, period=14, overbought=70, oversold=30):
    rsi = RSI(period, overbought, oversold)
    df = rsi.calculate(df)
    signals = rsi.generate_signals(df)
    return signals

# 사용 예:
# df = pd.read_csv('your_data.csv')
# signals = run_rsi_strategy(df)
# df['RSI_Signal'] = signals