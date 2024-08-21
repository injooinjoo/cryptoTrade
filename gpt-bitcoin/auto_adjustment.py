import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AutoAdjustment:
    def __init__(self, initial_params: Dict[str, float]):
        self.params = initial_params
        self.decision_threshold = 0.1
        if 'risk_factor' not in self.params:
            self.params['risk_factor'] = 1.0
        if 'stop_loss_factor' not in self.params:
            self.params['stop_loss_factor'] = 0.02
        self.performance_history = []
        self.param_history = []
        self.adjustment_threshold = 0.05

    def update_performance(self, performance: float):
        self.performance_history.append(performance)
        self.param_history.append(self.params.copy())

    def adjust_params(self) -> Dict[str, float]:
        if len(self.performance_history) < 10:
            return self.params

        recent_performance = np.mean(self.performance_history[-5:])
        overall_performance = np.mean(self.performance_history)

        if recent_performance < overall_performance * (1 - self.adjustment_threshold):
            best_params = self.param_history[np.argmax(self.performance_history)]

            for key in self.params:
                self.params[key] = (self.params[key] + best_params[key]) / 2

        return self.params

    def adjust_weights(self, model_performances: Dict[str, float]) -> Dict[str, float]:
        total_performance = sum(model_performances.values())
        if total_performance == 0:
            return {model: 1.0 / len(model_performances) for model in model_performances}

        adjusted_weights = {model: performance / total_performance for model, performance in model_performances.items()}
        return adjusted_weights

    def adjust_trade_ratio(self, price_volatility: float, base_ratio: float) -> float:
        volatility_factor = 1 + (price_volatility - 0.02) / 0.02  # 2% 변동성을 기준으로 조정
        adjusted_ratio = base_ratio * volatility_factor
        return max(0.1, min(adjusted_ratio, 1.0))  # 10% ~ 100% 범위로 제한

    def adjust_stop_loss(self, price_volatility: float, base_stop_loss: float) -> float:
        volatility_factor = 1 + (price_volatility - 0.02) / 0.02  # 2% 변동성을 기준으로 조정
        adjusted_stop_loss = base_stop_loss * volatility_factor
        return max(0.01, min(adjusted_stop_loss, 0.1))  # 1% ~ 10% 범위로 제한


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
    def __init__(self, window_size: int = 20, volatility_threshold: float = 0.02):
        self.window_size = window_size
        self.volatility_threshold = volatility_threshold

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

    def suggest_volatility_strategy(self, volatility: float) -> str:
        if volatility > self.volatility_threshold * 1.5:
            return "매우 높은 변동성: 포지션 크기 축소, 넓은 스톱로스/이익실현 범위 설정"
        elif volatility > self.volatility_threshold:
            return "높은 변동성: 중간 크기 포지션, 약간 넓은 스톱로스/이익실현 범위"
        elif volatility < self.volatility_threshold / 2:
            return "낮은 변동성: 큰 포지션 크기, 좁은 스톱로스/이익실현 범위"
        else:
            return "보통 변동성: 평균적인 포지션 크기와 스톱로스/이익실현 범위"


class DynamicTradingFrequencyAdjuster:
    def __init__(self, initial_threshold: float = 0.05, adjustment_factor: float = 0.01):
        self.decision_threshold = initial_threshold
        self.adjustment_factor = adjustment_factor
        self.recent_decisions = []
        self.max_threshold = 0.2
        self.min_threshold = 0.03
        self.volatility_threshold = 0.02

    def adjust_threshold(self, market_volatility: float):
        if market_volatility > self.volatility_threshold:
            self.decision_threshold = min(self.decision_threshold + self.adjustment_factor, self.max_threshold)
        elif market_volatility < self.volatility_threshold / 2:
            self.decision_threshold = max(self.decision_threshold - self.adjustment_factor, self.min_threshold)

    def should_trade(self, weighted_decision: float, market_regime: str) -> bool:
        if market_regime == 'high_volatility':
            return abs(weighted_decision) > self.decision_threshold * 1.2
        elif market_regime == 'low_volatility':
            return abs(weighted_decision) > self.decision_threshold * 0.8
        else:
            return abs(weighted_decision) > self.decision_threshold

    def update_recent_decisions(self, decision: str):
        self.recent_decisions.append(decision)
        if len(self.recent_decisions) > 20:
            self.recent_decisions.pop(0)

    def get_trading_frequency(self) -> float:
        if not self.recent_decisions:
            return 0
        return sum(1 for d in self.recent_decisions if d != 'hold') / len(self.recent_decisions)

    def get_volatility_adjusted_position_size(self, base_position_size: float, market_volatility: float) -> float:
        if market_volatility > self.volatility_threshold:
            return max(base_position_size * 0.8, 1.0)
        elif market_volatility < self.volatility_threshold / 2:
            return min(base_position_size * 1.2, 100.0)
        else:
            return base_position_size

    def get_volatility_adjusted_stop_loss(self, base_stop_loss: float, market_volatility: float) -> float:
        if market_volatility > self.volatility_threshold:
            return base_stop_loss * 1.2
        elif market_volatility < self.volatility_threshold / 2:
            return base_stop_loss * 0.8
        else:
            return base_stop_loss