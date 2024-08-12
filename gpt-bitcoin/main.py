import logging
import threading
import time
from collections import deque
from typing import Dict, Any

import numpy as np
import pandas as pd
import schedule
from dotenv import load_dotenv

from api_client import UpbitClient, OpenAIClient, PositionManager
from auto_adjustment import AutoAdjustment, AnomalyDetector, MarketRegimeDetector, DynamicTradingFrequencyAdjuster
from backtesting_and_ml import MLPredictor, RLAgent, run_backtest, LSTMPredictor, XGBoostPredictor
from config import load_config, setup_logging
from data_manager import DataManager
from discord_notifier import send_discord_message
from performance_monitor import PerformanceMonitor
from trading_logic import analyze_data_with_gpt4, execute_trade, data_manager

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)
config = load_config()
pd.set_option('future.no_silent_downcasting', True)


def update_data():
    data_manager.update_data()


class PositionMonitor(threading.Thread):
    def __init__(self, upbit_client, position_manager, stop_loss_percentage: float, take_profit_percentage: float):
        threading.Thread.__init__(self)
        self.upbit_client = upbit_client
        self.position_manager = position_manager
        self.stop_loss_percentage = stop_loss_percentage
        self.take_profit_percentage = take_profit_percentage
        self.running = True
        self.active_orders = {}
        self.highest_price = 0
        self.trailing_stop_percentage = 0.02

    def run(self):
        while self.running:
            try:
                position = self.position_manager.get_position()
                btc_balance = position['btc_balance']
                current_price = self.upbit_client.get_current_price("KRW-BTC")

                if btc_balance > 0:
                    avg_buy_price = self.upbit_client.get_avg_buy_price("BTC")

                    for order_id, order in list(self.active_orders.items()):
                        if current_price <= order['stop_loss']:
                            self._execute_sell(order_id, current_price, btc_balance, "Stop Loss")
                        elif current_price >= order['take_profit']:
                            self._execute_sell(order_id, current_price, btc_balance, "Take Profit")

                    profit_percentage = (current_price - avg_buy_price) / avg_buy_price * 100

                    if profit_percentage <= -self.stop_loss_percentage:
                        self._execute_sell(None, current_price, btc_balance, "Dynamic Stop Loss")
                    elif profit_percentage >= self.take_profit_percentage:
                        self._execute_sell(None, current_price, btc_balance, "Dynamic Take Profit")

                    self.highest_price = max(self.highest_price, current_price)
                    if current_price <= self.highest_price * (1 - self.trailing_stop_percentage):
                        self._execute_sell(None, current_price, btc_balance, "Trailing Stop")

                time.sleep(10)
            except Exception as e:
                logger.error(f"Error in position monitoring: {e}")
                time.sleep(60)

    def _execute_sell(self, order_id, current_price, amount, reason):
        try:
            logger.info(
                f"Executing sell for {'order ' + order_id if order_id else 'dynamic trigger'} at price {current_price} due to {reason}")
            result = self.upbit_client.sell_market_order("KRW-BTC", amount)
            if result:
                logger.info(f"Sell order executed successfully: {result}")
                if order_id:
                    self.remove_order(order_id)
                self.position_manager.update_position()
                send_discord_message(f"🚨 {reason} 실행: {amount:.8f} BTC sold at ₩{current_price:,}")
            else:
                logger.warning(
                    f"Sell order execution failed for {'order ' + order_id if order_id else 'dynamic trigger'}")
        except Exception as e:
            logger.error(f"Error executing sell for {'order ' + order_id if order_id else 'dynamic trigger'}: {e}")

    def remove_order(self, order_id: str):
        if order_id in self.active_orders:
            del self.active_orders[order_id]
            logger.info(f"Removed order: {order_id}")

    def stop(self):
        self.running = False


class TradingLoop:
    def __init__(self, upbit_client, openai_client, data_manager, ml_predictor, rl_agent,
                 auto_adjuster, anomaly_detector, regime_detector, dynamic_adjuster,
                 performance_monitor, position_monitor):
        self.upbit_client = upbit_client
        self.openai_client = openai_client
        self.data_manager = data_manager
        self.ml_predictor = ml_predictor
        self.rl_agent = rl_agent
        self.auto_adjuster = auto_adjuster
        self.anomaly_detector = anomaly_detector
        self.regime_detector = regime_detector
        self.dynamic_adjuster = dynamic_adjuster
        self.performance_monitor = performance_monitor
        self.xgboost_predictor = XGBoostPredictor()
        self.lstm_predictor = LSTMPredictor()  # LSTM 예측기 추가
        self.position_monitor = position_monitor
        self.counter = 0
        self.last_trade_time = None
        self.trading_interval = 600  # 10분
        self.last_decision = None
        self.buy_and_hold_performance = None
        self.config = load_config()
        self.initial_balance = None
        self.trading_history = []
        self.initial_balance = None
        self.trade_history = []
        self.pending_evaluations = deque()
        self.evaluation_delay = 600  # 10분 (초 단위)
        self.upbit_client = upbit_client
        self.openai_client = openai_client
        self.data_manager = data_manager
        self.ml_predictor = ml_predictor
        self.rl_agent = rl_agent
        self.auto_adjuster = auto_adjuster
        self.anomaly_detector = anomaly_detector
        self.regime_detector = regime_detector
        self.dynamic_adjuster = dynamic_adjuster
        self.performance_monitor = performance_monitor
        self.xgboost_predictor = XGBoostPredictor()
        self.lstm_predictor = LSTMPredictor()
        self.position_monitor = position_monitor
        self.counter = 0
        self.last_trade_time = None
        self.trading_interval = 600
        self.last_decision = None
        self.buy_and_hold_performance = None
        self.config = load_config()
        self.initial_balance = None
        self.trading_history = []
        self.trade_history = []

        # 모델 가중치 초기화
        self.ml_weight = 0.2
        self.gpt_weight = 0.2
        self.xgboost_weight = 0.2
        self.rl_weight = 0.2
        self.lstm_weight = 0.2

        self.strategy_performance = {
            'gpt': {'success': 0, 'failure': 0},
            'ml': {'success': 0, 'failure': 0},
            'xgboost': {'success': 0, 'failure': 0},
            'rl': {'success': 0, 'failure': 0},
            'lstm': {'success': 0, 'failure': 0}
        }
        # 가중치 조정 비율 설정
        self.weight_adjustment_rate = 0.01
        self.last_weight_update_time = time.time()

        # 새로운 속성 추가
        self.prediction_history = {
            'gpt': [],
            'ml': [],
            'xgboost': [],
            'rl': [],
            'lstm': []
        }
        self.prediction_accuracy = {
            'gpt': 0.5,
            'ml': 0.5,
            'xgboost': 0.5,
            'rl': 0.5,
            'lstm': 0.5
        }
        self.model_predictions = {
            'gpt': [], 'ml': [], 'xgboost': [], 'rl': [], 'lstm': []
        }
        self.weights = {
            'gpt': 0.2, 'ml': 0.2, 'xgboost': 0.2, 'rl': 0.2, 'lstm': 0.2
        }
        self.weight_adjustment = 0.01
        self.prediction_history = []

    def run(self, initial_strategy, initial_backtest_results, historical_data):
        self.data_manager.check_table_structure()
        self.config['trading_parameters'] = initial_strategy
        last_backtest_results = initial_backtest_results
        self.data_manager.fetch_extended_historical_data(days=365)

        # HODL 성능 계산
        initial_price = historical_data['close'].iloc[0]
        final_price = historical_data['close'].iloc[-1]
        self.hodl_performance = (final_price - initial_price) / initial_price

        # LSTM 모델 초기 학습
        self.lstm_predictor.train(historical_data)

        # XGBoost 모델 초기 학습
        X, y = self.data_manager.prepare_data_for_ml(historical_data)
        self.xgboost_predictor.train(X, y)

        while True:
            try:
                current_time = time.time()

                # 보류 중인 거래 평가 실행
                self.evaluate_pending_trades()

                if self.last_trade_time is None or (current_time - self.last_trade_time) >= self.trading_interval:
                    # 기존 미체결 주문 취소
                    self.cancel_existing_orders()
                    data = self.data_manager.ensure_sufficient_data()
                    data['date'] = pd.to_datetime(data['date'])
                    data.set_index('date', inplace=True)

                    ml_prediction = self.ml_predictor.predict(data)
                    xgboost_prediction = self.xgboost_predictor.predict(data)
                    rl_action = self.rl_agent.act(self.prepare_state(data))
                    lstm_prediction = self.lstm_predictor.predict(data)

                    latest_data = self.data_manager.ensure_sufficient_data()
                    latest_data['date'] = pd.to_datetime(latest_data['date'])
                    latest_data.set_index('date', inplace=True)
                    historical_data = pd.concat([historical_data, latest_data])
                    historical_data = historical_data[~historical_data.index.duplicated(keep='last')]
                    historical_data.sort_index(inplace=True)

                    market_analysis = self.analyze_market(data)
                    anomalies, anomaly_scores = self.anomaly_detector.detect_anomalies(data)
                    current_regime = self.regime_detector.detect_regime(data)
                    average_accuracy = self.data_manager.get_average_accuracy()
                    market_volatility = self.calculate_market_volatility(data)
                    self.dynamic_adjuster.adjust_threshold(market_volatility)
                    self.trading_interval = max(300, int(600 * self.dynamic_adjuster.decision_threshold))

                    gpt4_advice = analyze_data_with_gpt4(
                        data=data,
                        openai_client=self.openai_client,
                        params=self.config['trading_parameters'],
                        upbit_client=self.upbit_client,
                        average_accuracy=self.data_manager.get_average_accuracy(),
                        anomalies=anomalies,
                        anomaly_scores=anomaly_scores,
                        market_regime=current_regime,
                        ml_prediction=ml_prediction,
                        xgboost_prediction=xgboost_prediction,
                        rl_action=rl_action,
                        lstm_prediction=lstm_prediction,
                        backtest_results=last_backtest_results,
                        market_analysis=market_analysis,
                        current_balance=self.upbit_client.get_balance("KRW"),
                        current_btc_balance=self.upbit_client.get_balance("BTC"),
                        hodl_performance=self.hodl_performance,
                        current_performance=self.calculate_current_performance(),
                        trading_history=self.get_recent_trading_history()
                    )

                    # 예측 결과 기록
                    self.record_predictions(gpt4_advice['decision'], ml_prediction, xgboost_prediction, rl_action,
                                            lstm_prediction)

                    # 이전 예측 평가 및 가중치 업데이트
                    self.evaluate_previous_predictions()

                    # 예측 결과 및 가중치 로깅
                    self.log_predictions_and_weights()

                    # 새로운 의사결정 메서드 호출
                    decision = self.make_weighted_decision(gpt4_advice, ml_prediction, xgboost_prediction, rl_action,
                                                           lstm_prediction)

                    # 성능 모니터 업데이트
                    strategy_performance = self.calculate_strategy_performance()
                    hodl_performance = self.calculate_hodl_performance(historical_data)
                    current_balance = self.upbit_client.get_balance("KRW") + self.upbit_client.get_balance(
                        "BTC") * self.upbit_client.get_current_price("KRW-BTC")

                    self.performance_monitor.update(
                        strategy_performance,
                        hodl_performance,
                        self.prediction_accuracy,
                        self.calculate_model_weights(),
                        current_balance,
                        decision['decision'],
                        decision['target_price'],
                        decision['percentage']
                    )

                    # 'hold' 결정에 대한 처리 추가
                    if decision['decision'] == 'hold':
                        # 'hold' 결정을 성공으로 간주하고, 수익은 0으로 처리
                        self.performance_monitor.update_trade_result(True, 0)

                        # 성공률 업데이트
                        buy_hold_success_rate, sell_success_rate = self.calculate_success_rates()
                        self.performance_monitor.update_success_rates(buy_hold_success_rate, sell_success_rate)

                    # 거래 실행
                    if decision['decision'] in ['buy', 'sell']:
                        result = execute_trade(self.upbit_client, decision, self.config)
                        if result['success']:
                            # 거래 결과를 보류 중인 평가 목록에 추가
                            self.add_pending_evaluation(decision, result)

                            # 거래 정보 기록
                            trade_info = {
                                'timestamp': time.time(),
                                'decision': decision['decision'],
                                'price': result['price'],
                                'amount': result['amount']
                            }
                            self.performance_monitor.record_trade(trade_info)

                    self.last_trade_time = current_time

                    # 성능 보고서 출력 (10분마다)
                    if self.counter % 60 == 0:
                        performance_summary = self.performance_monitor.get_performance_summary()
                        logger.info(f"Performance Summary:\n{performance_summary}")
                        send_discord_message(performance_summary)

                    self.counter += 1

                # 주기적 업데이트 (예: 1시간마다)
                if self.counter % 360 == 0:  # 6시간마다 (10분 * 36)
                    self.periodic_update(historical_data)

                time.sleep(10)  # 10초마다 체크

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                logger.exception("Traceback:")
                time.sleep(60)

    def add_pending_evaluation(self, decision, result):
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
            success = self.evaluate_trade_success(trade['decision'], trade['result'])
            self.update_strategy_performance(trade['decision']['strategy'], success)

    def calculate_market_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(window=window).std().iloc[-1]
        return volatility * np.sqrt(252)  # 연간화된 변동성

    def update_strategy_performance(self, strategy, success):
        if success:
            self.strategy_performance[strategy]['success'] += 1
        else:
            self.strategy_performance[strategy]['failure'] += 1

    def evaluate_previous_predictions(self):
        if len(self.prediction_history) < 2:
            return

        previous_predictions, previous_price = self.prediction_history[-2]
        current_price = self.upbit_client.get_current_price("KRW-BTC")
        price_change = current_price - previous_price

        for model, prediction in previous_predictions.items():
            if (prediction > 0 and price_change > 0) or (prediction < 0 and price_change < 0):
                self.weights[model] += self.weight_adjustment
            else:
                self.weights[model] = max(0, self.weights[model] - self.weight_adjustment)

        logger.info(f"Updated weights: {self.weights}")

    def log_predictions_and_weights(self):
        logger.info("모델별 가중치:")
        for model, weight in self.weights.items():
            logger.info(f"{model.upper()}: 가중치 {weight:.4f}")

        logger.info("최근 예측 결과:")
        for model in self.model_predictions:
            recent_predictions = [pred[model] for pred, _ in self.prediction_history[-5:]]
            logger.info(f"{model.upper()}: {recent_predictions}")

    def analyze_market(self, data: pd.DataFrame) -> Dict[str, Any]:
        current_price = data['close'].iloc[-1]
        sma_20 = data['close'].rolling(window=20).mean().iloc[-1]
        rsi_14 = self.calculate_rsi(data['close'], period=14)

        return {
            'current_price': current_price,
            'sma_20': sma_20,
            'rsi_14': rsi_14,
            'trend': 'Bullish' if current_price > sma_20 else 'Bearish',
            'overbought_oversold': 'Overbought' if rsi_14 > 70 else 'Oversold' if rsi_14 < 30 else 'Neutral'
        }

    def prepare_state(self, data: pd.DataFrame) -> np.ndarray:
        state = np.array([
            data['close'].iloc[-1],
            data['volume'].iloc[-1],
            data['close'].pct_change().iloc[-1],
            data['close'].rolling(window=20).mean().iloc[-1],
            self.calculate_rsi(data['close'], period=14)
        ])
        return state.reshape(1, -1)

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))

    def get_recent_trading_history(self, n=5):
        return self.trading_history[-n:]

    def periodic_update(self, historical_data):
        last_backtest_results = run_backtest(self.data_manager, historical_data)
        ml_accuracy, _, ml_performance = self.ml_predictor.train(self.data_manager)

        X, y = self.data_manager.prepare_data_for_ml(historical_data)
        xgboost_accuracy, _, _ = self.xgboost_predictor.train(X, y)
        lstm_accuracy = self.lstm_predictor.train(historical_data)  # LSTM 모델 재학습

        self.performance_monitor.update_ml_metrics(ml_accuracy, ml_performance['loss'])
        self.performance_monitor.update_xgboost_metrics(xgboost_accuracy)
        self.performance_monitor.update_lstm_metrics(lstm_accuracy)  # LSTM 메트릭 업데이트

        adjusted_strategy = self.auto_adjuster.adjust_strategy(last_backtest_results, ml_accuracy, ml_performance)
        self.config['trading_parameters'] = adjusted_strategy

        logger.info(f"Strategy readjusted: {adjusted_strategy}")
        logger.info(f"Recent backtest results: {last_backtest_results}")
        logger.info(f"ML accuracy: {ml_accuracy}")
        logger.info(f"XGBoost accuracy: {xgboost_accuracy}")
        logger.info(f"LSTM accuracy: {lstm_accuracy}")

        current_performance = self.calculate_current_performance()
        if current_performance <= self.hodl_performance:
            logger.warning("Current strategy is underperforming HODL strategy. Adjusting parameters.")
            self.adjust_strategy_for_better_performance()

    def calculate_current_performance(self):
        try:
            recent_data = self.data_manager.get_recent_trades(days=30)  # 최근 30일 데이터
            if recent_data.empty:
                logger.warning("최근 거래 데이터가 없습니다.")
                return 0

            initial_price = recent_data['close'].iloc[0]
            final_price = recent_data['close'].iloc[-1]
            performance = (final_price - initial_price) / initial_price

            daily_returns = recent_data['close'].pct_change().dropna()
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()

            cumulative_returns = (1 + daily_returns).cumprod()
            max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()

            # 실제 거래 성과 반영
            current_balance = self.upbit_client.get_balance("KRW") + \
                              self.upbit_client.get_balance("BTC") * self.upbit_client.get_current_price("KRW-BTC")

            if self.initial_balance is None:
                self.initial_balance = current_balance  # 초기 잔액이 설정되지 않았다면 현재 잔액으로 설정

            actual_performance = (current_balance - self.initial_balance) / self.initial_balance if self.initial_balance else 0

            logger.info(f"현재 전략 성능: 시장 수익률 {performance:.2%}, 실제 수익률 {actual_performance:.2%}, "
                        f"샤프 비율 {sharpe_ratio:.2f}, 최대 손실폭 {max_drawdown:.2%}")

            # 종합 성능 점수 계산 (시장 수익률, 실제 수익률, 샤프 비율, 최대 손실폭 고려)
            performance_score = 0.3 * performance + 0.3 * actual_performance + 0.2 * sharpe_ratio - 0.2 * max_drawdown

            return performance_score

        except Exception as e:
            logger.error(f"성능 계산 중 오류 발생: {e}")
            return 0

    def adjust_strategy_for_better_performance(self):
        try:
            current_performance = self.calculate_current_performance()

            if current_performance <= self.hodl_performance:
                logger.warning(
                    f"현재 전략 성능({current_performance:.4f})이 HODL 전략 성능({self.hodl_performance:.4f})보다 낮습니다. 전략을 조정합니다.")

                # 거래 빈도 조정
                self.trading_interval = min(self.trading_interval * 1.2, 3600)  # 최대 1시간까지 증가

                # 리스크 및 거래 파라미터 조정
                self.config['trading_parameters']['risk_factor'] = self.config['trading_parameters'].get('risk_factor',
                                                                                                         1.0) * 0.9  # 리스크 10% 감소
                self.config['trading_parameters']['stop_loss_factor'] = self.config['trading_parameters'].get(
                    'stop_loss_factor', 0.02) * 1.1  # 손절 폭 10% 증가

                # take_profit_factor가 없는 경우 기본값 설정
                if 'take_profit_factor' not in self.config['trading_parameters']:
                    self.config['trading_parameters']['take_profit_factor'] = 0.03  # 기본값 3%
                self.config['trading_parameters']['take_profit_factor'] *= 0.9  # 익절 폭 10% 감소

                # 모델 가중치 조정
                self.adjust_model_weights()

                # 기술적 지표 임계값 조정
                self.adjust_technical_indicators()

                logger.info(f"전략 조정 완료: 거래 간격 {self.trading_interval}초, "
                            f"리스크 팩터 {self.config['trading_parameters']['risk_factor']:.2f}, "
                            f"손절 팩터 {self.config['trading_parameters']['stop_loss_factor']:.2f}, "
                            f"익절 팩터 {self.config['trading_parameters']['take_profit_factor']:.2f}")
                logger.info(f"모델 가중치: ML {self.ml_weight:.2f}, GPT {self.gpt_weight:.2f}, "
                            f"XGBoost {self.xgboost_weight:.2f}, RL {self.rl_weight:.2f}, LSTM {self.lstm_weight:.2f}")
                logger.info(f"기술적 지표 임계값: RSI 매수 {self.config['trading_parameters'].get('rsi_buy_threshold', 30)}, "
                            f"RSI 매도 {self.config['trading_parameters'].get('rsi_sell_threshold', 70)}")

            else:
                logger.info(
                    f"현재 전략 성능({current_performance:.4f})이 HODL 전략 성능({self.hodl_performance:.4f})보다 높습니다. 전략을 유지합니다.")

        except Exception as e:
            logger.error(f"전략 조정 중 오류 발생: {e}")
            logger.exception("상세 오류:")

    def adjust_model_weights(self):
        self.ml_weight = max(self.ml_weight * 0.9, 0.1)
        self.gpt_weight = min(self.gpt_weight * 1.1, 0.5)
        self.xgboost_weight = max(self.xgboost_weight * 0.95, 0.1)
        self.rl_weight = max(self.rl_weight * 0.95, 0.1)
        self.lstm_weight = min(self.lstm_weight * 1.05, 0.3)

    def adjust_technical_indicators(self):
        self.config['trading_parameters']['rsi_buy_threshold'] = max(
            self.config['trading_parameters'].get('rsi_buy_threshold', 30) - 2, 20)
        self.config['trading_parameters']['rsi_sell_threshold'] = min(
            self.config['trading_parameters'].get('rsi_sell_threshold', 70) + 2, 80)

    def cancel_existing_orders(self):
        try:
            orders = self.upbit_client.get_order("KRW-BTC")
            for order in orders:
                self.upbit_client.cancel_order(order['uuid'])
            logger.info("모든 미체결 주문이 취소되었습니다.")
        except Exception as e:
            logger.error(f"미체결 주문 취소 중 오류 발생: {e}")

    def record_predictions(self, gpt4_decision, ml_prediction, xgboost_prediction, rl_action, lstm_prediction):
        current_price = self.upbit_client.get_current_price("KRW-BTC")
        predictions = {
            'gpt': 1 if gpt4_decision == 'buy' else -1 if gpt4_decision == 'sell' else 0,
            'ml': ml_prediction,
            'xgboost': xgboost_prediction,
            'rl': 1 if rl_action == 2 else -1 if rl_action == 0 else 0,
            'lstm': 1 if lstm_prediction > current_price else -1 if lstm_prediction < current_price else 0
        }
        self.prediction_history.append((predictions, current_price))
        if len(self.prediction_history) > 100:
            self.prediction_history.pop(0)

    def make_weighted_decision(self, gpt4_advice, ml_prediction, xgboost_prediction, rl_action, lstm_prediction):
        current_price = self.upbit_client.get_current_price("KRW-BTC")

        decisions = {
            'gpt': 1 if gpt4_advice['decision'] == 'buy' else -1 if gpt4_advice['decision'] == 'sell' else 0,
            'ml': ml_prediction,
            'xgboost': xgboost_prediction,
            'rl': 1 if rl_action == 2 else -1 if rl_action == 0 else 0,
            'lstm': 1 if lstm_prediction > current_price else -1 if lstm_prediction < current_price else 0
        }

        weighted_decision = sum(decisions[model] * self.weights[model] for model in decisions)

        logger.info(f"Weighted contributions:")
        for model in decisions:
            logger.info(f"{model.upper()}: {decisions[model] * self.weights[model]:.4f}")
        logger.info(f"Total weighted decision: {weighted_decision:.4f}")

        # 결정 로직
        if weighted_decision > 0.1:
            final_decision = 'buy'
            target_price = gpt4_advice['potential_buy']['target_price'] if gpt4_advice['decision'] == 'hold' else \
            gpt4_advice['target_price']
            percentage = gpt4_advice['potential_buy']['percentage'] if gpt4_advice['decision'] == 'hold' else \
            gpt4_advice['percentage']
        elif weighted_decision < -0.1:
            final_decision = 'sell'
            target_price = gpt4_advice['potential_sell']['target_price'] if gpt4_advice['decision'] == 'hold' else \
            gpt4_advice['target_price']
            percentage = gpt4_advice['potential_sell']['percentage'] if gpt4_advice['decision'] == 'hold' else \
            gpt4_advice['percentage']
        else:
            final_decision = 'hold'
            target_price = None
            percentage = 0

        logger.info(f"Final decision: {final_decision}")
        logger.info(f"Target price: {target_price}")
        logger.info(f"Percentage: {percentage}%")

        # 예측 결과 기록
        return {
            'decision': final_decision,
            'percentage': percentage,
            'target_price': target_price,
            'stop_loss': gpt4_advice.get('stop_loss'),
            'take_profit': gpt4_advice.get('take_profit'),
            'reasoning': f"Weighted decision ({weighted_decision:.4f}) based on individual model predictions. {gpt4_advice['reasoning']}"
        }

    def calculate_strategy_performance(self):
        if self.initial_balance is None:
            self.initial_balance = self.upbit_client.get_balance("KRW") + \
                                   self.upbit_client.get_balance("BTC") * self.upbit_client.get_current_price("KRW-BTC")
        current_balance = self.upbit_client.get_balance("KRW") + \
                          self.upbit_client.get_balance("BTC") * self.upbit_client.get_current_price("KRW-BTC")
        return (current_balance - self.initial_balance) / self.initial_balance * 100

    def calculate_hodl_performance(self, historical_data):
        if len(historical_data) < 2:
            return 0
        initial_price = historical_data['close'].iloc[0]
        current_price = historical_data['close'].iloc[-1]
        return (current_price - initial_price) / initial_price * 100

    def calculate_model_weights(self):
        total_accuracy = sum(self.prediction_accuracy.values())
        if total_accuracy == 0:
            return {model: 0.2 for model in self.prediction_accuracy.keys()}
        return {model: acc / total_accuracy for model, acc in self.prediction_accuracy.items()}

    def evaluate_trade_success(self, decision, result):
        current_price = self.upbit_client.get_current_price("KRW-BTC")
        if decision['decision'] == 'buy':
            return current_price > result['price']
        elif decision['decision'] == 'sell':
            return current_price < result['price']
        return False

    def calculate_success_rates(self):
        if not self.trade_history:
            return 0, 0
        buy_hold_trades = [trade for trade in self.trade_history if trade['decision'] in ['buy', 'hold']]
        sell_trades = [trade for trade in self.trade_history if trade['decision'] == 'sell']

        buy_hold_success = sum(1 for trade in buy_hold_trades if trade['success']) / len(
            buy_hold_trades) if buy_hold_trades else 0
        sell_success = sum(1 for trade in sell_trades if trade['success']) / len(sell_trades) if sell_trades else 0

        return buy_hold_success * 100, sell_success * 100


def main():
    upbit_client = UpbitClient(config['upbit_access_key'], config['upbit_secret_key'])
    openai_client = OpenAIClient(config['openai_api_key'])
    data_manager = DataManager(upbit_client)

    # 필요한 컬럼을 테이블에 추가합니다.
    data_manager.add_decision_column()
    # 디버그 정보 출력
    data_manager.debug_database()

    ml_predictor = MLPredictor()
    rl_agent = RLAgent(state_size=12, action_size=3)
    auto_adjuster = AutoAdjustment(config.get('initial_params', {}))
    anomaly_detector = AnomalyDetector()
    regime_detector = MarketRegimeDetector()
    dynamic_adjuster = DynamicTradingFrequencyAdjuster()
    performance_monitor = PerformanceMonitor(upbit_client)
    position_manager = PositionManager(upbit_client)

    position_monitor = PositionMonitor(upbit_client, position_manager, stop_loss_percentage=5,
                                       take_profit_percentage=10)
    position_monitor.start()

    # 주기적 데이터 업데이트 스케줄링
    schedule.every(10).minutes.do(update_data)

    try:
        historical_data = data_manager.ensure_sufficient_data(1440)  # 24시간 데이터

        historical_data['date'] = pd.to_datetime(historical_data['date'])
        historical_data.set_index('date', inplace=True)

        if historical_data.empty:
            logger.error("Failed to retrieve historical data.")
            return

        initial_backtest_results = run_backtest(data_manager, historical_data)
        ml_accuracy, _, _ = ml_predictor.train(data_manager)

        initial_strategy = auto_adjuster.generate_initial_strategy(initial_backtest_results, ml_accuracy)

        trading_loop = TradingLoop(
            upbit_client, openai_client, data_manager, ml_predictor, rl_agent,
            auto_adjuster, anomaly_detector, regime_detector, dynamic_adjuster,
            performance_monitor, position_monitor
        )
        trading_loop.run(initial_strategy, initial_backtest_results, historical_data)
    finally:
        position_monitor.stop()


if __name__ == "__main__":
    main()