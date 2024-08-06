# auto_adjustment.py

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AutoAdjustment:
    def __init__(self, initial_params: Dict[str, float], adjustment_threshold: float = 0.1):
        self.params = initial_params
        self.adjustment_threshold = adjustment_threshold
        self.performance_history = []
        self.param_history = []

    def update_performance(self, performance: float):
        self.performance_history.append(performance)
        self.param_history.append(self.params.copy())

    def adjust_params(self) -> Dict[str, float]:
        if len(self.performance_history) < 10:  # 충분한 데이터가 쌓일 때까지 기다림
            return self.params

        recent_performance = np.mean(self.performance_history[-5:])
        overall_performance = np.mean(self.performance_history)

        if recent_performance < overall_performance * (1 - self.adjustment_threshold):
            logger.info("Performance degradation detected. Adjusting parameters.")
            best_params = self.param_history[np.argmax(self.performance_history)]

            # 현재 파라미터와 최고 성능 파라미터의 중간값으로 조정
            for key in self.params:
                self.params[key] = (self.params[key] + best_params[key]) / 2

        return self.params


class AnomalyDetector:
    def __init__(self, window_size: int = 100, contamination: float = 0.01):
        self.window_size = window_size
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)

    def detect_anomalies(self, data: pd.DataFrame) -> Tuple[List[bool], List[float]]:
        if len(data) < self.window_size:
            return [False] * len(data), [0] * len(data)

        features = self._extract_features(data)
        scaled_features = self.scaler.fit_transform(features)

        anomalies = self.isolation_forest.fit_predict(scaled_features)
        anomaly_scores = self.isolation_forest.score_samples(scaled_features)

        return (anomalies == -1).tolist(), anomaly_scores.tolist()

    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        features = []
        for i in range(len(data) - self.window_size + 1):
            window = data.iloc[i:i + self.window_size]
            feature = [
                window['close'].mean(),
                window['close'].std(),
                window['volume'].mean(),
                window['close'].pct_change().mean(),
                window['close'].pct_change().std(),
                (window['close'].iloc[-1] - window['close'].iloc[0]) / window['close'].iloc[0],
            ]
            features.append(feature)
        return np.array(features)


class MarketRegimeDetector:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size

    def detect_regime(self, data: pd.DataFrame) -> str:
        if len(data) < self.window_size:
            return "Unknown"

        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(window=self.window_size).std().iloc[-1]
        trend = self._calculate_trend(data)

        if volatility > returns.std() * 1.5:
            regime = "High Volatility"
        elif trend > 0.05:
            regime = "Bull Market"
        elif trend < -0.05:
            regime = "Bear Market"
        else:
            regime = "Sideways Market"

        return regime

    def _calculate_trend(self, data: pd.DataFrame) -> float:
        y = data['close'].values
        x = np.arange(len(y))
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope / data['close'].mean()