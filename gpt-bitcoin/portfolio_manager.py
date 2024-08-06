import numpy as np
import pandas as pd
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class PortfolioManager:
    def __init__(self, upbit_client, tickers: List[str]):
        """Initialize the PortfolioManager with a list of tickers and an Upbit client."""
        self.upbit_client = upbit_client
        self.tickers = tickers
        self.weights = self._initialize_weights()

    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize equal weights for all tickers in the portfolio."""
        return {ticker: 1.0 / len(self.tickers) for ticker in self.tickers}

    def get_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Fetch historical OHLCV data for each ticker over a specified number of days."""
        data = {}
        for ticker in self.tickers:
            try:
                ohlcv = self.upbit_client.get_ohlcv(ticker, interval="day", count=days)
                data[ticker] = ohlcv['close']
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame(data)

    def calculate_correlation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate the correlation matrix of returns between the tickers."""
        return data.pct_change().corr()

    def optimize_weights(self, data: pd.DataFrame) -> Dict[str, float]:
        """Optimize the portfolio weights using a simple Monte Carlo simulation."""
        returns = data.pct_change().mean()
        cov_matrix = data.pct_change().cov()

        num_assets = len(self.tickers)
        num_portfolios = 10000

        results = np.zeros((3, num_portfolios))
        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)

            portfolio_return = np.sum(returns * weights)
            portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            results[0, i] = portfolio_return
            results[1, i] = portfolio_stddev
            results[2, i] = results[0, i] / results[1, i]  # Sharpe ratio

        best_idx = np.argmax(results[2])
        best_weights = results[:, best_idx]

        return {ticker: weight for ticker, weight in zip(self.tickers, best_weights)}

    def rebalance_portfolio(self):
        """Rebalance the portfolio based on historical data and optimized weights."""
        historical_data = self.get_historical_data()
        correlation = self.calculate_correlation(historical_data)
        logger.info(f"Asset correlation:\n{correlation}")

        optimized_weights = self.optimize_weights(historical_data)
        logger.info(f"Optimized weights: {optimized_weights}")

        self.weights = optimized_weights

    def get_investment_recommendation(self, available_balance: float) -> Dict[str, float]:
        """Generate an investment recommendation based on the available balance and current weights."""
        return {ticker: available_balance * weight for ticker, weight in self.weights.items()}
