# auto_adjustment.py
import logging
from typing import Dict, List, Tuple, Any

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

    def generate_initial_strategy(self, backtest_results, ml_accuracy) -> Dict[str, float]:
        initial_strategy = {
            'risk_factor': 1.0,
            'stop_loss_factor': 0.02,
            'take_profit_factor': 0.03,
            'trade_frequency': 1.0
        }

        if 'sharpe_ratio' in backtest_results:
            sharpe_ratio = backtest_results['sharpe_ratio']
            if sharpe_ratio > 1.5:
                initial_strategy['risk_factor'] *= 1.2
                initial_strategy['trade_frequency'] *= 1.1
            elif sharpe_ratio < 0.5:
                initial_strategy['risk_factor'] *= 0.8
                initial_strategy['trade_frequency'] *= 0.9

        if 'max_drawdown' in backtest_results:
            max_drawdown = backtest_results['max_drawdown']
            if max_drawdown > 0.2:
                initial_strategy['stop_loss_factor'] *= 1.2
            elif max_drawdown < 0.1:
                initial_strategy['stop_loss_factor'] *= 0.8

        if ml_accuracy > 0.7:
            initial_strategy['take_profit_factor'] *= 1.2
        elif ml_accuracy < 0.5:
            initial_strategy['take_profit_factor'] *= 0.8

        return initial_strategy

    def update_performance(self, performance: float):
        self.performance_history.append(performance)
        self.param_history.append(self.params.copy())

    def adjust_params(self) -> Dict[str, float]:
        if len(self.performance_history) < 10:  # 충분한 데이터가 쌓일 때까지 기다림
            return self.params

        recent_performance = np.mean(self.performance_history[-5:])
        overall_performance = np.mean(self.performance_history)

        if recent_performance < overall_performance * (1 - self.adjustment_threshold):
            best_params = self.param_history[np.argmax(self.performance_history)]

            # 현재 파라미터와 최고 성능 파라미터의 중간값으로 조정
            for key in self.params:
                self.params[key] = (self.params[key] + best_params[key]) / 2

        return self.params

    def generate_final_decision(self, gpt4_advice, ml_prediction, rl_action, current_regime, backtest_results,
                                dynamic_target, upbit_client, ml_accuracy):
        # gpt4_advice['decision']을 숫자로 변환
        gpt4_decision_value = 1 if gpt4_advice['decision'] == 'buy' else -1 if gpt4_advice['decision'] == 'sell' else 0

        weighted_decision = (
                gpt4_decision_value * 0.4 +
                ml_prediction * 0.3 +
                (rl_action - 1) * 0.3  # rl_action을 -1, 0, 1로 변환 (0: sell, 1: hold, 2: buy)
        )

        should_trade = self.should_trade(weighted_decision, current_regime)

        if not should_trade:
            return {
                "decision": "hold",
                "percentage": 0,
                "target_price": None,
                "stop_loss": None,
                "take_profit": None,
                "reasoning": "Default hold decision due to analysis error",
                "risk_assessment": "high",
            }

        try:
            current_price = upbit_client.get_current_price("KRW-BTC")
            if current_price is None:
                logger.error("Failed to get current price from Upbit")
                return {
                    "decision": "hold",
                    "percentage": 0,
                    "target_price": None,
                    "stop_loss": None,
                    "take_profit": None,
                    "reasoning": "Failed to get current price",
                    "risk_assessment": "high",
                }
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return {
                "decision": "hold",
                "percentage": 0,
                "target_price": None,
                "stop_loss": None,
                "take_profit": None,
                "reasoning": "Error getting current price",
                "risk_assessment": "high",
            }

        base_position_size = float(gpt4_advice.get('percentage', 10))  # 기본 포지션 크기
        adjusted_position_size = self.get_volatility_adjusted_position_size(base_position_size,
                                                                            backtest_results.get('volatility', 0.01))

        # 최소 거래 비율 설정 (예: 1%)
        min_trade_percentage = 1.0
        adjusted_position_size = max(adjusted_position_size, min_trade_percentage)

        base_stop_loss = current_price * 0.98  # 기본 손절가 (2% 하락)
        adjusted_stop_loss = self.get_volatility_adjusted_stop_loss(base_stop_loss,
                                                                    backtest_results.get('volatility', 0.01))

        target_price = gpt4_advice.get('target_price')
        if target_price is None or not isinstance(target_price, (int, float)):
            target_price = current_price

        take_profit = gpt4_advice.get('take_profit')
        if take_profit is None or not isinstance(take_profit, (int, float)):
            take_profit = current_price * 1.03  # 기본 3% 이익 실현

        decision = 'buy' if weighted_decision > 0 else 'sell' if weighted_decision < 0 else 'hold'

        if decision == 'hold':
            return {
                "decision": "hold",
                "percentage": 0,
                "target_price": None,
                "stop_loss": None,
                "take_profit": None,
                "reasoning": "Weighted decision resulted in hold",
                "risk_assessment": "low",
            }

        return {
            'decision': decision,
            'percentage': adjusted_position_size,
            'target_price': float(target_price),
            'stop_loss': adjusted_stop_loss,
            'take_profit': float(take_profit),
            'reasoning': f"Decision based on weighted analysis. Market regime: {current_regime}, ML Accuracy: {ml_accuracy:.2f}",
            'risk_assessment': 'medium',
        }

    def should_trade(self, weighted_decision: float, market_regime: str) -> bool:
        """Determine if a trade should be executed based on the weighted decision and market regime."""
        if market_regime == 'high_volatility':
            return abs(weighted_decision) > self.decision_threshold * 1.5
        elif market_regime == 'low_volatility':
            return abs(weighted_decision) > self.decision_threshold * 0.5
        else:
            return abs(weighted_decision) > self.decision_threshold

    def get_volatility_adjusted_position_size(self, base_position_size: float, market_volatility: float) -> float:
        """Adjust position size based on market volatility."""
        volatility_threshold = 0.02  # 예시 임계값, 필요에 따라 조정
        if market_volatility > volatility_threshold:
            return max(base_position_size * 0.8, 1.0)  # 최소 1% 유지
        elif market_volatility < volatility_threshold / 2:
            return min(base_position_size * 1.2, 100.0)  # 최대 100% 제한
        else:
            return base_position_size

    def get_volatility_adjusted_stop_loss(self, base_stop_loss: float, market_volatility: float) -> float:
        volatility_threshold = 0.02
        if market_volatility > volatility_threshold * 1.5:
            return base_stop_loss * 1.3  # 매우 높은 변동성
        elif market_volatility > volatility_threshold:
            return base_stop_loss * 1.2  # 높은 변동성
        elif market_volatility < volatility_threshold / 2:
            return base_stop_loss * 0.9  # 낮은 변동성
        else:
            return base_stop_loss * 1.1  # 보통 변동성

    def adjust_strategy_based_on_performance(self, performance_metrics: Dict[str, Any]):
        if 'win_rate' in performance_metrics:
            if performance_metrics['win_rate'] < 40:
                self.decision_threshold *= 1.1  # 승률이 낮으면 거래 빈도를 줄임
            elif performance_metrics['win_rate'] > 60:
                self.decision_threshold *= 0.9  # 승률이 높으면 거래 빈도를 늘림

        if 'sharpe_ratio' in performance_metrics:
            if performance_metrics['sharpe_ratio'] < 0.5:
                self.params['risk_factor'] *= 0.9  # 샤프 비율이 낮으면 리스크를 줄임
            elif performance_metrics['sharpe_ratio'] > 1.5:
                self.params['risk_factor'] *= 1.1  # 샤프 비율이 높으면 리스크를 높임

        if 'max_drawdown' in performance_metrics:
            if performance_metrics['max_drawdown'] > 0.2:
                self.params['stop_loss_factor'] *= 1.1  # 최대 낙폭이 크면 손절 기준을 높임

    def adjust_strategy(self, backtest_results: Dict[str, Any], ml_accuracy: float, ml_performance: float) -> Dict[str, float]:
        sharpe_ratio = backtest_results.get('sharpe_ratio', 0)
        if sharpe_ratio > 1.5:
            self.params['risk_factor'] *= 1.1
        elif sharpe_ratio < 0.5:
            self.params['risk_factor'] *= 0.9

        max_drawdown = backtest_results.get('max_drawdown', 0)
        if max_drawdown > 0.2:
            self.params['stop_loss_factor'] *= 1.1
        elif max_drawdown < 0.1:
            self.params['stop_loss_factor'] *= 0.9

        if ml_accuracy > 0.7:
            self.params['risk_factor'] *= 1.1
        elif ml_accuracy < 0.5:
            self.params['risk_factor'] *= 0.9

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
        if len(self.recent_decisions) > 20:  # Keep only last 20 decisions
            self.recent_decisions.pop(0)

    def get_trading_frequency(self) -> float:
        if not self.recent_decisions:
            return 0
        return sum(1 for d in self.recent_decisions if d != 'hold') / len(self.recent_decisions)

    def get_volatility_adjusted_position_size(self, base_position_size: float, market_volatility: float) -> float:
        if market_volatility > self.volatility_threshold:
            return max(base_position_size * 0.8, 1.0)  # 최소 1% 유지
        elif market_volatility < self.volatility_threshold / 2:
            return min(base_position_size * 1.2, 100.0)  # 최대 100% 제한
        else:
            return base_position_size

    def get_volatility_adjusted_stop_loss(self, base_stop_loss: float, market_volatility: float) -> float:
        if market_volatility > self.volatility_threshold:
            return base_stop_loss * 1.2  # Widen stop loss range in high volatility
        elif market_volatility < self.volatility_threshold / 2:
            return base_stop_loss * 0.8  # Tighten stop loss range in low volatility
        else:
            return base_stop_loss
