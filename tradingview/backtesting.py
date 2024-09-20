# backtesting.py

import pandas as pd
import numpy as np
from kNN import run_knn_strategy
from EMA_Ribbon import run_ema_ribbon_strategy
from RSI import run_rsi_strategy


class Backtester:
    def __init__(self, data, initial_capital=10000):
        self.data = data
        self.initial_capital = initial_capital
        self.position = 0
        self.cash = initial_capital
        self.portfolio_value = []

    def run(self, strategy):
        for i in range(len(self.data)):
            signal = strategy[i]
            price = self.data['Close'].iloc[i]

            if signal == 'BUY' and self.position == 0:
                shares_to_buy = self.cash // price
                self.position += shares_to_buy
                self.cash -= shares_to_buy * price
            elif signal == 'SELL' and self.position > 0:
                self.cash += self.position * price
                self.position = 0

            current_value = self.cash + self.position * price
            self.portfolio_value.append(current_value)

        return pd.Series(self.portfolio_value, index=self.data.index)


def calculate_metrics(portfolio_value):
    returns = portfolio_value.pct_change()
    cumulative_returns = (1 + returns).cumprod() - 1
    total_return = cumulative_returns.iloc[-1]
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
    max_drawdown = (portfolio_value / portfolio_value.cummax() - 1).min()

    return {
        'Total Return': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }


def run_backtest(df):
    # KNN 전략
    knn_signals = run_knn_strategy(df)
    knn_backtester = Backtester(df)
    knn_portfolio = knn_backtester.run(knn_signals)
    knn_metrics = calculate_metrics(knn_portfolio)

    # EMA Ribbon 전략
    ema_signals = run_ema_ribbon_strategy(df)
    ema_backtester = Backtester(df)
    ema_portfolio = ema_backtester.run(ema_signals)
    ema_metrics = calculate_metrics(ema_portfolio)

    # RSI 전략
    rsi_signals = run_rsi_strategy(df)
    rsi_backtester = Backtester(df)
    rsi_portfolio = rsi_backtester.run(rsi_signals)
    rsi_metrics = calculate_metrics(rsi_portfolio)

    # 결과 출력
    print("KNN Strategy Metrics:")
    print(knn_metrics)
    print("\nEMA Ribbon Strategy Metrics:")
    print(ema_metrics)
    print("\nRSI Strategy Metrics:")
    print(rsi_metrics)


if __name__ == "__main__":
    df = pd.read_csv('crypto_data.csv', index_col='timestamp', parse_dates=True)
    run_backtest(df)