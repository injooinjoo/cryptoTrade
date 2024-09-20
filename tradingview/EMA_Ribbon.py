# EMA_Ribbon.py

import pandas as pd
import numpy as np


class EMAribbon:
    def __init__(self, periods=[5, 10, 20, 30, 40, 50]):
        self.periods = periods

    def calculate(self, df):
        for period in self.periods:
            df[f'EMA_{period}'] = self.calculate_ema(df['Close'], period)
        return df

    def calculate_ema(self, prices, period):
        ema = prices.ewm(span=period, adjust=False).mean()
        return ema

    def generate_signals(self, df):
        signals = []
        for i in range(len(df)):
            if i < max(self.periods):
                signals.append('HOLD')
                continue

            ema_values = [df[f'EMA_{period}'].iloc[i] for period in self.periods]

            if all(ema_values[i] > ema_values[i + 1] for i in range(len(ema_values) - 1)):
                signals.append('BUY')
            elif all(ema_values[i] < ema_values[i + 1] for i in range(len(ema_values) - 1)):
                signals.append('SELL')
            else:
                signals.append('HOLD')

        return signals


def run_ema_ribbon_strategy(df):
    ribbon = EMAribbon()
    df = ribbon.calculate(df)
    signals = ribbon.generate_signals(df)
    return signals

# 사용 예:
# df = pd.read_csv('your_data.csv')
# signals = run_ema_ribbon_strategy(df)
# df['EMA_Signal'] = signals