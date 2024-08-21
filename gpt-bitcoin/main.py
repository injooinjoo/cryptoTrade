import logging
import time
from typing import Dict, Any

import numpy as np
import pandas as pd
from collections import deque

from api_client import UpbitClient, OpenAIClient
from data_manager import DataManager
from backtesting_and_ml import MLPredictor, XGBoostPredictor, LSTMPredictor, RLAgent, ARIMAPredictor, ProphetPredictor, \
    TransformerPredictor
from auto_adjustment import AutoAdjustment, AnomalyDetector, MarketRegimeDetector, DynamicTradingFrequencyAdjuster
from performance_monitor import PerformanceMonitor
from trading_logic import analyze_data_with_gpt4, TradingDecisionMaker
logger = logging.getLogger(__name__)
from discord_notifier import send_discord_message
from config import load_config


class TradingLoop:
    def __init__(self, upbit_client, openai_client, data_manager, ml_predictor, xgboost_predictor,
                 lstm_predictor, rl_agent, auto_adjuster, anomaly_detector, regime_detector,
                 dynamic_adjuster, performance_monitor, config):
        self.upbit_client = upbit_client
        self.openai_client = openai_client
        self.data_manager = data_manager
        self.ml_predictor = ml_predictor
        self.xgboost_predictor = xgboost_predictor
        self.lstm_predictor = lstm_predictor
        self.rl_agent = rl_agent
        self.auto_adjuster = auto_adjuster
        self.anomaly_detector = anomaly_detector
        self.regime_detector = regime_detector
        self.dynamic_adjuster = dynamic_adjuster
        self.performance_monitor = performance_monitor
        self.config = config

        self.arima_predictor = ARIMAPredictor()
        self.prophet_predictor = ProphetPredictor()
        self.transformer_predictor = TransformerPredictor(input_dim=5, d_model=32, nhead=2, num_layers=1,
                                                          dim_feedforward=64, seq_length=10)

        self.decision_maker = TradingDecisionMaker(config)

        self.counter = 0
        self.last_trade_time = None
        self.trading_interval = 600  # 10 minutes
        self.evaluation_delay = 600
        self.pending_evaluations = deque()

        self.initialize_model_performance()

        # 초기 데이터로 모델 학습
        initial_data = self.data_manager.ensure_sufficient_data()
        self.train_models(initial_data)

    def initialize_model_performance(self):
        self.model_performance = {
            model: {
                'predictions': [],
                'actual_values': [],
                'accuracy': 0.0
            } for model in ['gpt', 'ml', 'xgboost', 'rl', 'lstm', 'arima', 'prophet', 'transformer']
        }

    def train_models(self, data):
        # 데이터 샘플링 (예: 최근 1000개 데이터포인트만 사용)
        sampled_data = data.tail(1000).copy()

        logger.info(f"Sampled data shape: {sampled_data.shape}")
        logger.info(f"Sampled data columns: {sampled_data.columns}")

        X, y = self.data_manager.prepare_data_for_ml(sampled_data)

        logger.info(f"Prepared X shape: {X.shape}, y shape: {y.shape}")

        self.ml_predictor.train(X, y)
        self.xgboost_predictor.train(X, y)
        self.lstm_predictor.train(sampled_data)
        self.rl_agent.train(sampled_data)
        try:
            self.arima_predictor.train(sampled_data)
        except Exception as e:
            logger.error(f"ARIMA 모델 훈련 중 오류 발생: {e}")
        self.prophet_predictor.train(sampled_data)
        self.transformer_predictor.train(sampled_data)

    def run(self):
        while True:
            try:
                current_time = time.time()

                self.evaluate_pending_trades()
                data = self.data_manager.ensure_sufficient_data()

                # 기술적 지표 추가
                data_with_indicators = self.data_manager.add_technical_indicators(data)

                if data_with_indicators.empty or len(data_with_indicators) < 33:
                    logger.warning("충분한 데이터가 없습니다. 다음 주기를 기다립니다.")
                    time.sleep(60)
                    continue

                if self.last_trade_time is None or (current_time - self.last_trade_time) >= self.trading_interval:
                    self.last_trade_time = current_time

                    self.cancel_existing_orders()

                    market_analysis = self.analyze_market(data_with_indicators)
                    anomalies, anomaly_scores = self.anomaly_detector.detect_anomalies(data_with_indicators)
                    current_regime = self.regime_detector.detect_regime(data_with_indicators)

                    current_price = self.upbit_client.get_current_price("KRW-BTC")

                    predictions = self.get_predictions(data_with_indicators, current_price)

                    gpt4_advice = analyze_data_with_gpt4(
                        data=data_with_indicators,
                        openai_client=self.openai_client,
                        params=self.config['trading_parameters'],
                        upbit_client=self.upbit_client,
                        average_accuracy=self.data_manager.get_average_accuracy(),
                        anomalies=anomalies,
                        anomaly_scores=anomaly_scores,
                        market_regime=current_regime,
                        ml_prediction=predictions['ml'],
                        xgboost_prediction=predictions['xgboost'],
                        rl_action=predictions['rl'],
                        lstm_prediction=predictions['lstm'],
                        arima_prediction=predictions['arima'],
                        prophet_prediction=predictions['prophet'],
                        transformer_prediction=predictions['transformer'],
                        backtest_results={},  # TODO: Implement backtest results
                        market_analysis=market_analysis,
                        current_balance=self.upbit_client.get_balance("KRW"),
                        current_btc_balance=self.upbit_client.get_balance("BTC"),
                        hodl_performance=0,  # TODO: Implement HODL performance calculation
                        current_performance=0,  # TODO: Implement current performance calculation
                        trading_history=self.data_manager.get_recent_decisions()
                    )
                    predictions['gpt'] = gpt4_advice['target_price']

                    decision = self.decision_maker.make_decision(predictions, current_price)

                    if isinstance(data_with_indicators.index, pd.DatetimeIndex):
                        decision_time = data_with_indicators.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        decision_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

                    self.data_manager.save_decision(decision_time, decision['decision'])

                    result = self.decision_maker.execute_trade(self.upbit_client, decision)
                    if result['success']:
                        self.add_pending_evaluation(decision, result)
                        self.performance_monitor.record_trade(result)

                    self.last_trade_time = current_time
                    self.counter += 1

                if self.counter % 6 == 0:  # Every hour (6 * 10 minutes)
                    self.periodic_update()

                time.sleep(10)

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                logger.exception("Traceback:")
                time.sleep(60)

    def get_predictions(self, data: pd.DataFrame, current_price: float) -> Dict[str, float]:
        return {
            'ml': self.ml_predictor.predict(data),
            'xgboost': self.xgboost_predictor.predict(data),
            'lstm': self.lstm_predictor.predict(data),
            'rl': self.rl_agent.predict_price(self.prepare_state(data)),
            'arima': self.arima_predictor.predict(),
            'prophet': self.prophet_predictor.predict(future_periods=1),
            'transformer': self.transformer_predictor.predict(data)
        }

    def analyze_market(self, data: pd.DataFrame) -> Dict[str, Any]:
        current_price = data['close'].iloc[-1]
        sma_20 = data['sma'].iloc[-1]
        rsi_14 = data['rsi'].iloc[-1]

        return {
            'current_price': current_price,
            'sma_20': sma_20,
            'rsi_14': rsi_14,
            'trend': 'Bullish' if current_price > sma_20 else 'Bearish',
            'overbought_oversold': 'Overbought' if rsi_14 > 70 else 'Oversold' if rsi_14 < 30 else 'Neutral'
        }

    def prepare_state(self, data: pd.DataFrame) -> np.ndarray:
        return np.array([
            data['close'].iloc[-1],
            data['volume'].iloc[-1],
            data['close'].pct_change().iloc[-1],
            data['sma'].iloc[-1],
            data['rsi'].iloc[-1]
        ]).reshape(1, -1)

    def add_pending_evaluation(self, decision: Dict[str, Any], result: Dict[str, Any]):
        evaluation_time = time.time() + self.evaluation_delay
        self.pending_evaluations.append({
            'decision': decision,
            'result': result,
            'evaluation_time': evaluation_time
        })

    def evaluate_pending_trades(self):
        current_time = time.time()
        while self.pending_evaluations and self.pending_evaluations[0]['evaluation_time'] <= current_time:
            trade = self.pending_evaluations.popleft()
            actual_price = self.upbit_client.get_current_price("KRW-BTC")
            self.decision_maker.update_weights(actual_price)
            self.performance_monitor.update_trade_result(trade['decision'], actual_price)

    def cancel_existing_orders(self):
        try:
            orders = self.upbit_client.get_order("KRW-BTC")
            for order in orders:
                self.upbit_client.cancel_order(order['uuid'])
            logger.info("모든 미체결 주문이 취소되었습니다.")
        except Exception as e:
            logger.error(f"미체결 주문 취소 중 오류 발생: {e}")

    def periodic_update(self):
        self.auto_adjuster.adjust_params()
        self.dynamic_adjuster.adjust_threshold(self.calculate_market_volatility(self.data_manager.get_recent_data()))

        performance_summary = self.performance_monitor.get_performance_summary()
        send_discord_message(performance_summary)

    def calculate_market_volatility(self, data: pd.DataFrame) -> float:
        returns = data['close'].pct_change().dropna()
        return returns.std() * np.sqrt(252)


# Main function
def main():
    config = load_config()
    upbit_client = UpbitClient(config['upbit_access_key'], config['upbit_secret_key'])
    openai_client = OpenAIClient(config['openai_api_key'])
    data_manager = DataManager(upbit_client)

    ml_predictor = MLPredictor()
    xgboost_predictor = XGBoostPredictor()
    lstm_predictor = LSTMPredictor()
    rl_agent = RLAgent(state_size=5, action_size=3)
    auto_adjuster = AutoAdjustment(config.get('initial_params', {}))
    anomaly_detector = AnomalyDetector()
    regime_detector = MarketRegimeDetector()
    dynamic_adjuster = DynamicTradingFrequencyAdjuster()
    performance_monitor = PerformanceMonitor(upbit_client)

    trading_loop = TradingLoop(
        upbit_client, openai_client, data_manager, ml_predictor, xgboost_predictor,
        lstm_predictor, rl_agent, auto_adjuster, anomaly_detector, regime_detector,
        dynamic_adjuster, performance_monitor, config
    )

    trading_loop.run()


if __name__ == "__main__":
    main()