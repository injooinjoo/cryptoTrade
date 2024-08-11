import logging
import threading
import time

import numpy as np
import pandas as pd
import schedule
from typing import Dict, Any
from trading_logic import execute_sell, execute_buy
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

    def add_order(self, order_id: str, order_type: str, price: float, stop_loss: float, take_profit: float):
        self.active_orders[order_id] = {
            'type': order_type,
            'price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        logger.info(f"Added new order: {order_id} with stop loss at {stop_loss} and take profit at {take_profit}")

    def remove_order(self, order_id: str):
        if order_id in self.active_orders:
            del self.active_orders[order_id]
            logger.info(f"Removed order: {order_id}")

    def update_stop_loss_take_profit(self, order_id: str, stop_loss: float, take_profit: float):
        if order_id in self.active_orders:
            self.active_orders[order_id]['stop_loss'] = stop_loss
            self.active_orders[order_id]['take_profit'] = take_profit
            logger.info(f"Updated order {order_id}: new stop loss at {stop_loss} and take profit at {take_profit}")

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

        # 모델 가중치 초기화
        self.ml_weight = 0.2
        self.gpt_weight = 0.3
        self.xgboost_weight = 0.2
        self.rl_weight = 0.15
        self.lstm_weight = 0.15

        if 'trading_parameters' not in self.config:
            self.config['trading_parameters'] = {}
        if 'buy_threshold' not in self.config['trading_parameters']:
            self.config['trading_parameters']['buy_threshold'] = 0.01
        if 'sell_threshold' not in self.config['trading_parameters']:
            self.config['trading_parameters']['sell_threshold'] = 0.01

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

                if self.last_trade_time is None or (current_time - self.last_trade_time) >= self.trading_interval:
                    # 기존 미체결 주문 취소
                    self.cancel_existing_orders()
                    data = self.data_manager.ensure_sufficient_data()
                    data['date'] = pd.to_datetime(data['date'])
                    data.set_index('date', inplace=True)

                    ml_prediction = self.ml_predictor.predict(data)
                    xgboost_prediction = self.xgboost_predictor.predict(data)
                    rl_action = self.rl_agent.act(self.prepare_state(data))
                    lstm_prediction = self.lstm_predictor.predict(data)  # LSTM 예측 추가

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

                    decision = self.make_trading_decision(gpt4_advice, ml_prediction, xgboost_prediction, rl_action,
                                                          lstm_prediction)

                    if self.last_decision:
                        self.review_last_decision(data)

                    if decision['decision'] in ['buy', 'sell']:
                        result = execute_trade(self.upbit_client, decision, self.config)
                        if result['success']:
                            logger.info(f"Trade executed: {result['message']}")
                            actual_result = "성공" if result['success'] else "실패"
                            self.performance_monitor.add_decision(decision['decision'], decision['reasoning'],
                                                                  actual_result)

                            # 다음 거래 전에 개선 제안 요청
                            improvement_suggestion = self.performance_monitor.get_improvement_suggestion(
                                self.openai_client)
                            logger.info(f"Improvement suggestion: {improvement_suggestion}")

                            if result['uuid']:
                                self.position_monitor.add_order(
                                    result['uuid'], decision['decision'], decision['target_price'],
                                    decision['stop_loss'], decision['take_profit']
                                )
                            self.last_decision = decision
                        else:
                            logger.warning(f"Trade failed: {result['message']}")

                    self.last_trade_time = current_time

                    if self.counter % 144 == 0:  # 약 1일마다
                        self.periodic_update(historical_data)

                    self.send_performance_report()
                    self.record_performance(decision)
                    self.counter += 1

                self.monitor_positions()

                time.sleep(10)  # 10초마다 체크

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                logger.exception("Traceback:")
                time.sleep(60)

    def calculate_market_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(window=window).std().iloc[-1]
        return volatility * np.sqrt(252)  # 연간화된 변동성

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

    # def record_trade(self, trade):
    #     self.trading_history.append(trade)
    #     if len(self.trading_history) > 100:  # 최근 100개의 거래만 유지
    #         self.trading_history.pop(0)
    #
    # def get_initial_balance(self):
    #     krw_balance = self.upbit_client.get_balance("KRW")
    #     btc_balance = self.upbit_client.get_balance("BTC")
    #     btc_price = self.upbit_client.get_current_price("KRW-BTC")
    #     return krw_balance + (btc_balance * btc_price)

    def record_performance(self, decision: Dict[str, Any]):
        try:
            current_price = self.upbit_client.get_current_price("KRW-BTC")
            current_balance = self.upbit_client.get_balance("KRW")
            current_btc_amount = self.upbit_client.get_balance("BTC")

            # Assuming you have functions or data to populate these
            parameter_dict = {}  # Example: populate with actual parameters if available
            market_regime = "bull"  # Example: determine the market regime
            anomalies_detected = []  # Example: populate based on detected anomalies
            ml_model_accuracy = self.ml_predictor.get_accuracy()  # Example: retrieve model accuracy
            ml_model_loss = self.ml_predictor.get_loss()  # Example: retrieve model loss

            self.performance_monitor.record(
                decision=decision,
                current_price=current_price,
                balance=current_balance,
                btc_amount=current_btc_amount,
                params=parameter_dict,
                regime=market_regime,
                anomalies=anomalies_detected,
                ml_accuracy=ml_model_accuracy,
                ml_loss=ml_model_loss
            )
            logger.info(f"Performance recorded: {decision['decision']} at {current_price}")
        except Exception as e:
            logger.error(f"Error recording performance: {e}")

    def send_performance_report(self):
        try:
            report = self.performance_monitor.get_performance_summary()
            send_discord_message(report)
            logger.info("Performance report sent")
        except Exception as e:
            logger.error(f"Error sending performance report: {e}")

    def make_trading_decision(self, gpt4_advice, ml_prediction, xgboost_prediction, rl_action, lstm_prediction):
        weights = {'gpt4': 0.25, 'ml': 0.2, 'xgboost': 0.2, 'rl': 0.15, 'lstm': 0.2}

        gpt4_decision = 1 if gpt4_advice['decision'] == 'buy' else -1 if gpt4_advice['decision'] == 'sell' else 0
        ml_decision = 1 if ml_prediction == 1 else -1 if ml_prediction == 0 else 0
        xgboost_decision = 1 if xgboost_prediction == 1 else -1 if xgboost_prediction == 0 else 0
        rl_decision = 1 if rl_action == 2 else -1 if rl_action == 0 else 0
        lstm_decision = 1 if lstm_prediction > 0 else -1 if lstm_prediction < 0 else 0

        weighted_decision = (
                gpt4_decision * weights['gpt4'] +
                ml_decision * weights['ml'] +
                xgboost_decision * weights['xgboost'] +
                rl_decision * weights['rl'] +
                lstm_decision * weights['lstm']
        )

        logger.info(
            f"Decision weights - GPT4: {weights['gpt4']}, ML: {weights['ml']}, XGBoost: {weights['xgboost']}, RL: {weights['rl']}, LSTM: {weights['lstm']}")
        logger.info(
            f"Individual decisions - GPT4: {gpt4_decision}, ML: {ml_decision}, XGBoost: {xgboost_decision}, RL: {rl_decision}, LSTM: {lstm_decision}")
        logger.info(f"Weighted decision: {weighted_decision}")

        final_decision = 'buy' if weighted_decision > 0.1 else 'sell' if weighted_decision < -0.1 else 'hold'

        logger.info(f"Final trading decision: {final_decision}")

        if final_decision == 'buy' and gpt4_advice['decision'] == 'hold':
            percentage = gpt4_advice['potential_buy']['percentage']
            target_price = gpt4_advice['potential_buy']['target_price']
        elif final_decision == 'sell' and gpt4_advice['decision'] == 'hold':
            percentage = gpt4_advice['potential_sell']['percentage']
            target_price = gpt4_advice['potential_sell']['target_price']
        else:
            percentage = gpt4_advice['percentage']
            target_price = gpt4_advice['target_price']

        return {
            'decision': final_decision,
            'percentage': percentage,
            'target_price': target_price,
            'stop_loss': gpt4_advice['stop_loss'],
            'take_profit': gpt4_advice['take_profit'],
            'reasoning': f"Weighted decision based on GPT-4 ({weights['gpt4']:.2f}), ML ({weights['ml']:.2f}), XGBoost ({weights['xgboost']:.2f}), RL ({weights['rl']:.2f}), and LSTM ({weights['lstm']:.2f}). {gpt4_advice.get('reasoning', '')}"
        }

    def review_last_decision(self, current_data):
        if self.last_decision:
            last_price = self.last_decision.get('target_price')
            current_price = current_data['close'].iloc[-1]
            if self.last_decision['decision'] == 'buy':
                success = current_price > last_price
            elif self.last_decision['decision'] == 'sell':
                success = current_price < last_price
            else:
                success = abs(current_price - last_price) / last_price < 0.01

            logger.info(f"Last decision review: {'Successful' if success else 'Unsuccessful'}")

            if not success:
                self.adjust_strategy_based_on_failure(self.last_decision, current_price)

    def adjust_strategy_based_on_failure(self, last_decision, current_price):
        if last_decision['decision'] == 'buy' and current_price < last_decision['target_price']:
            self.config['trading_parameters']['risk_factor'] *= 0.95
        elif last_decision['decision'] == 'sell' and current_price > last_decision['target_price']:
            self.config['trading_parameters']['risk_factor'] *= 1.05

        logger.info(f"Adjusted trading parameters: {self.config['trading_parameters']}")

    def monitor_positions(self):
        current_price = self.upbit_client.get_current_price("KRW-BTC")
        for order_id, order in self.position_monitor.active_orders.items():
            if order['type'] == 'buy':
                if current_price <= order['stop_loss'] or current_price >= order['take_profit']:
                    execute_sell(self.upbit_client, order_id, current_price, "Stop Loss/Take Profit")
            elif order['type'] == 'sell':
                if current_price >= order['stop_loss'] or current_price <= order['take_profit']:
                    execute_buy(self.upbit_client, order_id, current_price, "Stop Loss/Take Profit")

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
    performance_monitor = PerformanceMonitor()
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