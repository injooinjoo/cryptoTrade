import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Callable
import logging

logger = logging.getLogger(__name__)


class AdvancedBacktester:
    def __init__(self, data: pd.DataFrame, initial_balance: float, fee_rate: float):
        self.data = data
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate

    def run_walkforward_analysis(self, strategy: Callable, window_size: int, step_size: int) -> Dict:
        if len(self.data) < window_size:
            logger.warning(f"Not enough data for walk-forward analysis. Using all data for single run.")
            return self.single_run_backtest(strategy)

        results = []
        for start in range(0, len(self.data) - window_size, step_size):
            end = start + window_size
            train_data = self.data.iloc[start:end]
            test_data = self.data.iloc[end:min(end + step_size, len(self.data))]

            if len(test_data) == 0:
                break

            optimized_params = self.optimize_strategy(strategy, train_data)
            test_result = self.backtest(strategy, test_data, optimized_params)
            results.append(test_result)

        return self.aggregate_results(results)

    def single_run_backtest(self, strategy: Callable) -> Dict:
        params = {'sma_short': 5, 'sma_long': 10}  # Default params for small dataset
        result = self.backtest(strategy, self.data, params)
        return {
            'mean_return': result['total_return'],
            'std_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': self.calculate_max_drawdown([result['total_return']])
        }

    def optimize_strategy(self, strategy: Callable, data: pd.DataFrame) -> Dict:
        def objective(params):
            int_params = [max(1, int(round(p))) for p in params]  # Ensure params are at least 1
            result = self.backtest(strategy, data, {'sma_short': int_params[0], 'sma_long': int_params[1]})
            return -result['total_return']

        initial_params = [5, 10]  # Smaller initial values
        bounds = [(2, min(20, len(data) // 2)), (5, min(50, len(data) - 1))]

        result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)

        optimized_params = [max(1, int(round(p))) for p in result.x]
        return dict(zip(['sma_short', 'sma_long'], optimized_params))

    def backtest(self, strategy: Callable, data: pd.DataFrame, params: Dict) -> Dict:
        balance = self.initial_balance
        position = 0
        trades = []

        for i in range(1, len(data)):
            signal = strategy(data.iloc[:i], params)
            current_price = data['close'].iloc[i]

            if signal == 1 and balance > 0:  # Buy signal
                buy_amount = balance * 0.99  # Consider fees
                position += buy_amount / current_price * (1 - self.fee_rate)
                balance = 0
                trades.append(('buy', current_price, position))
            elif signal == -1 and position > 0:  # Sell signal
                balance += position * current_price * (1 - self.fee_rate)
                position = 0
                trades.append(('sell', current_price, balance))

        final_balance = balance + position * data['close'].iloc[-1]
        total_return = (final_balance - self.initial_balance) / self.initial_balance * 100

        return {
            'final_balance': final_balance,
            'total_return': total_return,
            'num_trades': len(trades)
        }

    def aggregate_results(self, results: List[Dict]) -> Dict:
        if not results:
            logger.warning("No results to aggregate.")
            return {'mean_return': 0, 'std_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}

        returns = [r['total_return'] for r in results]
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0,
            'max_drawdown': self.calculate_max_drawdown(returns)
        }

    def calculate_max_drawdown(self, returns: List[float]) -> float:
        if not returns:
            return 0
        cumulative_returns = np.cumprod(1 + np.array(returns) / 100)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        return np.max(drawdown) * 100


def run_backtest(historical_data: pd.DataFrame):
    backtester = AdvancedBacktester(historical_data, initial_balance=10_000_000, fee_rate=0.0005)

    # Dynamically adjust window_size and step_size based on data length
    data_length = len(historical_data)
    window_size = min(180, max(5, data_length // 2))
    step_size = max(1, data_length // 10)

    logger.info(f"Running backtest with window_size={window_size}, step_size={step_size}")
    results = backtester.run_walkforward_analysis(example_strategy, window_size=window_size, step_size=step_size)
    logger.info(f"Advanced backtest results: {results}")
    return results


def example_strategy(data: pd.DataFrame, params: Dict) -> int:
    if len(data) < params['sma_long']:
        return 0

    sma_short = data['close'].rolling(window=params['sma_short']).mean().iloc[-1]
    sma_long = data['close'].rolling(window=params['sma_long']).mean().iloc[-1]

    if sma_short > sma_long:
        return 1  # Buy signal
    elif sma_short < sma_long:
        return -1  # Sell signal
    else:
        return 0  # Hold