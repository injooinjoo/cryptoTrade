import logging
import os
import time
from typing import Dict, Any
import threading

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from api_client import UpbitClient, OpenAIClient, PositionManager
from auto_adjustment import AutoAdjustment, AnomalyDetector, MarketRegimeDetector, DynamicTradingFrequencyAdjuster
from backtesting_and_ml import MLPredictor, RLAgent, run_backtest
from config import load_config, setup_logging
from data_manager import DataManager
from discord_notifier import send_discord_message, send_performance_summary
from performance_monitor import PerformanceMonitor
from trading_logic import analyze_data_with_gpt4, execute_trade, default_hold_decision

# 환경 변수 로드
load_dotenv()

# 로깅 설정
setup_logging()
logger = logging.getLogger(__name__)

# 설정 로드
config = load_config()
upbit_client = UpbitClient(config['upbit_access_key'], config['upbit_secret_key'])
openai_client = OpenAIClient(config['openai_api_key'])

class PositionMonitor(threading.Thread):
    def __init__(self, upbit_client, position_manager, stop_loss_percentage: float, take_profit_percentage: float):
        threading.Thread.__init__(self)
        self.upbit_client = upbit_client
        self.position_manager = position_manager
        self.stop_loss_percentage = stop_loss_percentage
        self.take_profit_percentage = take_profit_percentage
        self.running = True
        self.active_orders: Dict[str, Dict[str, Any]] = {}

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

                time.sleep(10)
            except Exception as e:
                logger.error(f"Error in position monitoring: {e}")
                time.sleep(60)

    def _execute_sell(self, order_id: str, current_price: float, amount: float, reason: str):
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
        self.position_monitor = position_monitor
        self.counter = 0
        self.last_trade_time = None

    def run(self, initial_strategy, initial_backtest_results):
        global config

        config['trading_parameters'] = initial_strategy
        last_backtest_results = initial_backtest_results

        while True:
            try:
                current_time = pd.Timestamp.now()

                if self.last_trade_time is None or (current_time - self.last_trade_time) >= pd.Timedelta(minutes=10):
                    logger.info("10분이 경과하여 새로운 거래 분석을 시작합니다.")

                    multi_timeframe_data = self.data_manager.safe_fetch_multi_timeframe_data()
                    data = multi_timeframe_data['short']

                    if len(data) < 1381:
                        logger.warning(f"데이터 부족: {len(data)} 행. 부분적인 분석만 수행합니다.")
                        ml_prediction = None
                        rl_action = None
                    else:
                        ml_prediction = self.ml_predictor.predict(data)
                        rl_action = self.rl_agent.act(self.prepare_state(data))

                    market_analysis = self.analyze_market(data)
                    anomalies, anomaly_scores = self.anomaly_detector.detect_anomalies(data)
                    current_regime = self.regime_detector.detect_regime(data)
                    average_accuracy = self.data_manager.get_average_accuracy()

                    gpt4_advice = analyze_data_with_gpt4(
                        data,
                        self.openai_client,
                        config['trading_parameters'],
                        self.upbit_client,
                        average_accuracy,
                        anomalies,
                        anomaly_scores,
                        current_regime,
                        ml_prediction,
                        rl_action,
                        last_backtest_results
                    )

                    decision = self.make_trading_decision(gpt4_advice, ml_prediction, rl_action)

                    if decision['decision'] in ['buy', 'sell']:
                        result = execute_trade(self.upbit_client, decision, config)
                        if result:
                            print(result)
                            order_id = result['uuid']
                            self.position_monitor.add_order(
                                order_id,
                                decision['decision'],
                                decision['target_price'],
                                decision['stop_loss'],
                                decision['take_profit']
                            )

                    self.record_performance(decision)

                    self.last_trade_time = current_time

                    if self.counter % 144 == 0:  # 약 1일마다
                        if len(data) >= 1381:
                            last_backtest_results = run_backtest(data)
                            ml_accuracy, _, ml_performance = self.ml_predictor.train(data)

                            adjusted_strategy = self.auto_adjuster.adjust_strategy(last_backtest_results, ml_accuracy,
                                                                                   ml_performance)
                            config['trading_parameters'] = adjusted_strategy

                            logger.info(f"전략 재조정: {adjusted_strategy}")
                            logger.info(f"최근 백테스트 결과: {last_backtest_results}")

                    self.counter += 1

                else:
                    time.sleep(10)

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                logger.exception("Traceback:")
                time.sleep(60)

    def analyze_market(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the market using technical indicators and return the analysis."""

        current_price = data['close'].iloc[-1]

        # Simple Moving Average (SMA)
        sma_60 = data['close'].rolling(window=60).mean().iloc[-1]
        sma_200 = data['close'].rolling(window=200).mean().iloc[-1]

        # Exponential Moving Average (EMA)
        ema_60 = data['close'].ewm(span=60, adjust=False).mean().iloc[-1]

        # Relative Strength Index (RSI)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_14 = 100 - (100 / (1 + rs)).iloc[-1]

        # Bollinger Bands
        bb_middle = sma_60
        bb_upper = bb_middle + (2 * data['close'].rolling(window=60).std().iloc[-1])
        bb_lower = bb_middle - (2 * data['close'].rolling(window=60).std().iloc[-1])

        # Market Trend
        trend = "Bullish" if current_price > sma_60 else "Bearish"

        # Volume Analysis
        avg_volume = data['volume'].mean()
        current_volume = data['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume != 0 else 1.0

        return {
            "current_price": current_price,
            "sma_60": sma_60,
            "sma_200": sma_200,
            "ema_60": ema_60,
            "rsi_14": rsi_14,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "trend": trend,
            "volume_ratio": volume_ratio
        }

    def prepare_state(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare the state vector for the RL agent."""

        # 상태에 포함할 주요 지표 선택
        close_price = data['close'].values[-1]
        sma_60 = data['close'].rolling(window=60).mean().values[-1]
        sma_200 = data['close'].rolling(window=200).mean().values[-1]
        rsi_14 = 100 - (100 / (1 + (data['close'].diff().where(data['close'].diff() > 0, 0).rolling(window=14).mean() /
                                    -data['close'].diff().where(data['close'].diff() < 0, 0).rolling(
                                        window=14).mean()))).values[-1]
        volume = data['volume'].values[-1]

        # 상태 벡터 생성
        state = np.array([close_price, sma_60, sma_200, rsi_14, volume])

        # 벡터를 2차원 배열로 변환하여 반환
        return state.reshape(1, -1)

    def make_trading_decision(self, gpt4_advice: Dict[str, Any], ml_prediction: int, rl_action: int) -> Dict[str, Any]:
        """Make a trading decision based on GPT-4 advice, ML prediction, and RL action."""

        # GPT-4, ML, RL 결정을 숫자로 변환
        gpt4_decision = 1 if gpt4_advice['decision'] == 'buy' else -1 if gpt4_advice['decision'] == 'sell' else 0
        ml_decision = 1 if ml_prediction == 1 else -1 if ml_prediction == 0 else 0
        rl_decision = 1 if rl_action == 2 else -1 if rl_action == 0 else 0

        # 각 결정에 가중치를 부여하여 종합 결정 산출
        print(f'gpt4_decision:{gpt4_decision}/ ml_decision:{ml_decision} / rl_decision:{rl_decision}')
        weighted_decision = (gpt4_decision * 0.4) + (ml_decision * 0.3) + (rl_decision * 0.3)

        # 최종 결정 도출
        decision = 'buy' if weighted_decision > 0 else 'sell' if weighted_decision < 0 else 'hold'

        # 최종 결정 구조 반환
        return {
            'decision': decision,
            'percentage': gpt4_advice.get('percentage', 0),
            'target_price': gpt4_advice.get('target_price'),
            'stop_loss': gpt4_advice.get('stop_loss'),
            'take_profit': gpt4_advice.get('take_profit'),
            'reasoning': gpt4_advice.get('reasoning', 'Combined decision from GPT-4, ML, and RL.')
        }

    def record_performance(self, decision: Dict[str, Any]):
        try:
            current_price = self.upbit_client.get_current_price("KRW-BTC")
            balance = self.upbit_client.get_balance("KRW")
            btc_amount = self.upbit_client.get_balance("BTC")
            self.performance_monitor.record(
                decision, current_price, balance, btc_amount,
                config.get('trading_parameters', {}),
                self.regime_detector.detect_regime(self.data_manager.get_recent_data(1440)),
                any(self.anomaly_detector.detect_anomalies(self.data_manager.get_recent_data(1440))[0]),
                self.ml_predictor.get_accuracy(),
                self.ml_predictor.get_loss()
            )
        except Exception as e:
            logger.error(f"Error recording performance: {e}")


def main():
    data_manager = DataManager(upbit_client)
    ml_predictor = MLPredictor()
    rl_agent = RLAgent(state_size=5, action_size=3)
    auto_adjuster = AutoAdjustment(config.get('initial_params', {}))
    anomaly_detector = AnomalyDetector()
    regime_detector = MarketRegimeDetector()
    dynamic_adjuster = DynamicTradingFrequencyAdjuster()
    performance_monitor = PerformanceMonitor()
    position_manager = PositionManager(upbit_client)

    # PositionMonitor 시작
    position_monitor = PositionMonitor(upbit_client, position_manager, stop_loss_percentage=5,
                                       take_profit_percentage=10)
    position_monitor.start()

    try:
        # 초기 백테스트 실행
        historical_data = upbit_client.get_ohlcv("KRW-BTC", interval="minute10", count=1440)
        initial_backtest_results = run_backtest(historical_data)

        # 초기 ML 모델 학습
        ml_accuracy, _, _ = ml_predictor.train(historical_data)

        # 초기 전략 설정
        initial_strategy = auto_adjuster.generate_initial_strategy(initial_backtest_results, ml_accuracy)

        # TradingLoop 실행
        trading_loop = TradingLoop(
            upbit_client, openai_client, data_manager, ml_predictor, rl_agent,
            auto_adjuster, anomaly_detector, regime_detector, dynamic_adjuster,
            performance_monitor, position_monitor
        )
        trading_loop.run(initial_strategy, initial_backtest_results)
    finally:
        # 프로그램 종료 시 PositionMonitor 중지
        position_monitor.stop()
        position_monitor.join()


if __name__ == "__main__":
    main()