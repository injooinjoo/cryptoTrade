import logging

import pandas as pd
import schedule
import time
from dotenv import load_dotenv
import os
from typing import Dict, Any

from api_client import UpbitClient, OpenAIClient, OrderManager, PositionManager
from data_preparation import safe_fetch_multi_timeframe_data
from trading_logic import analyze_data_with_gpt4, execute_buy, execute_sell, trading_strategy, ml_predictor, rl_agent, \
    prepare_state, get_reward, execute_trade
from backtesting import run_backtest
from monitoring import MonitoringSystem
from discord_notifier import send_discord_message, send_performance_summary
from auto_adjustment import AutoAdjustment, AnomalyDetector, MarketRegimeDetector
from ml_models import MLPredictor, RLAgent
from performance_monitor import PerformanceMonitor
from config import load_config, setup_logging
from database import get_previous_decision, update_decision_accuracy, get_recent_decisions

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()
setup_logging()
# Upbit 및 OpenAI 클라이언트 초기화
upbit_client = UpbitClient(os.getenv("UPBIT_ACCESS_KEY"), os.getenv("UPBIT_SECRET_KEY"))
openai_client = OpenAIClient(os.getenv("OPENAI_API_KEY"))


def main():
    setup_logging()
    # 연결 확인
    if not upbit_client.check_connection():
        logger.error("Failed to connect to Upbit. Exiting.")
        return

    # 설정 로드
    config = load_config()

    # PerformanceMonitor 인스턴스 생성
    performance_monitor = PerformanceMonitor()
    position_manager = PositionManager(upbit_client)
    order_manager = OrderManager(upbit_client, position_manager)

    # 초기 파라미터 설정
    initial_params = {
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'sma_short': 10,
        'sma_long': 30
    }
    auto_adjuster = AutoAdjustment(initial_params)
    anomaly_detector = AnomalyDetector()
    regime_detector = MarketRegimeDetector()
    performance_monitor = PerformanceMonitor()

    # 백테스팅 실행
    logger.info("Starting backtesting...")
    historical_data = upbit_client.get_ohlcv("KRW-BTC", interval="day", count=365)  # 1년치 일별 데이터
    backtest_results = run_backtest(historical_data)
    logger.info(f"Backtest results: {backtest_results}")

    # 머신러닝 모델 학습
    ml_predictor.train(historical_data)

    # 강화학습 에이전트 학습
    # train_rl_agent(rl_agent, historical_data)

    # 트레이딩 루프
    def trading_loop():
        try:
            logger.info("Starting trading loop")

            position_manager.update_position()

            multi_timeframe_data = safe_fetch_multi_timeframe_data(upbit_client)
            data = multi_timeframe_data['short']

            decision = execute_trading_strategy(data, upbit_client, config, auto_adjuster, anomaly_detector,
                                                regime_detector)
            logger.info(f"Trading decision: {decision}")

            current_balance = upbit_client.get_balance("KRW")

            trade_executed = execute_trade(upbit_client, order_manager, decision, current_balance)

            if trade_executed:
                logger.info("Trade executed successfully")
                # stop_loss와 take_profit 업데이트
                order_manager.update_stop_loss_take_profit(decision['stop_loss'], decision['take_profit'])
            else:
                logger.warning("Trade execution failed or was not necessary")

            # 성능 모니터링 데이터 기록
            current_price = upbit_client.get_current_price("KRW-BTC")
            balance = upbit_client.get_balance("KRW")
            btc_amount = upbit_client.get_balance("BTC")
            performance_monitor.record(
                decision,
                current_price,
                balance,
                btc_amount,
                decision.get('param_adjustment', {}),
                decision.get('risk_assessment', 'unknown'),
                decision.get('anomalies', False)
            )

        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            logger.exception("Traceback:")

    # 성능 요약 보고 함수
    def report_performance():
        summary = performance_monitor.get_performance_summary()
        send_performance_summary(summary)
        logger.info("Performance summary sent to Discord")

    trading_loop()

    # 10분마다 트레이딩 로직 실행
    schedule.every(10).minutes.do(trading_loop)

    # 매일 자정에 성능 모니터링 데이터 저장
    schedule.every().day.at("00:00").do(performance_monitor.save_to_file)

    # 6시간마다 성능 요약 보고
    schedule.every(6).hours.do(report_performance)

    # 메인 루프
    logger.info("Starting main loop")
    while True:
        schedule.run_pending()
        time.sleep(1)


def train_rl_agent(rl_agent: RLAgent, historical_data: pd.DataFrame):
    """강화학습 에이전트 학습."""
    for episode in range(10):  # 에피소드 수는 조정 가능
        state = prepare_state(historical_data.iloc[:1])
        # print(historical_data)
        total_reward = 0
        # print(len(historical_data))
        for i in range(1, len(historical_data)):
            # print(i)
            action = rl_agent.act(state)
            next_state = prepare_state(historical_data.iloc[i:i + 1])
            reward = get_reward(action, historical_data['close'].iloc[i], historical_data['close'].iloc[i - 1])
            done = (i == len(historical_data) - 1)
            rl_agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if len(rl_agent.memory) > 32:
                rl_agent.replay(32)

        logger.info(f"Episode: {episode + 1}, Total Reward: {total_reward}")


def execute_trading_strategy(
        data: pd.DataFrame,
        upbit_client: UpbitClient,
        config: Dict[str, Any],
        auto_adjuster: AutoAdjustment,
        anomaly_detector: AnomalyDetector,
        regime_detector: MarketRegimeDetector
    ) -> Dict[str, Any]:
    """트레이딩 전략을 실행하고 결정을 반환합니다."""

    # 이전 결정 평가
    evaluate_previous_decision(upbit_client)

    # 최근 정확도를 반영한 데이터 분석
    average_accuracy = analyze_database_for_decision()

    # 기존 로직에 따라 트레이딩 결정 수행
    anomalies, anomaly_scores = anomaly_detector.detect_anomalies(data)
    current_regime = regime_detector.detect_regime(data)
    current_params = auto_adjuster.adjust_params()

    ml_prediction = ml_predictor.predict(data)
    state = prepare_state(data)
    rl_action = rl_agent.act(state)
    backtest_results = run_backtest(data)

    gpt4_advice = analyze_data_with_gpt4(
        data, openai_client, current_params, upbit_client, average_accuracy, anomalies, anomaly_scores, current_regime,
        ml_prediction, rl_action, backtest_results
    )

    # 결과에 따라 트레이딩 수행
    if gpt4_advice['decision'] == 'buy':
        execute_buy(upbit_client, gpt4_advice['percentage'], gpt4_advice['target_price'], current_params)
    elif gpt4_advice['decision'] == 'sell':
        execute_sell(upbit_client, gpt4_advice['percentage'], gpt4_advice['target_price'], current_params)

    gpt4_advice['ml_prediction'] = ml_prediction
    gpt4_advice['rl_action'] = rl_action

    return gpt4_advice


def evaluate_previous_decision(upbit_client: UpbitClient):
    """이전 결정과 실제 결과를 비교하여 데이터베이스에 저장."""
    previous_decision = get_previous_decision()
    if not previous_decision:
        logger.info("No previous decision found to evaluate.")
        return

    actual_price = upbit_client.get_current_price("KRW-BTC")
    accuracy = 0

    # 이전 결정의 타겟 가격과 현재 가격을 비교하여 결과 확인
    if previous_decision['decision'] == 'buy' and actual_price >= previous_decision['target_price']:
        accuracy = 1
    elif previous_decision['decision'] == 'sell' and actual_price <= previous_decision['target_price']:
        accuracy = 1

    # 데이터베이스에 결과와 분석 저장
    update_decision_accuracy(previous_decision['id'], accuracy)
    logger.info(f"Previous decision evaluated: {'Success' if accuracy == 1 else 'Failure'} with accuracy {accuracy}")

    return accuracy


def analyze_database_for_decision():
    """데이터베이스의 모든 기록을 참고하여 최근 기록을 더 비중 있게 반영."""
    recent_decisions = get_recent_decisions(days=30)
    total_accuracy = 0
    count = 0

    for decision in recent_decisions:
        if decision['accuracy'] is not None:
            total_accuracy += decision['accuracy']
            count += 1

    average_accuracy = total_accuracy / count if count > 0 else 0
    logger.info(f"Average accuracy over the last 30 days: {average_accuracy:.2f}")

    return average_accuracy



def update_and_report_performance(
        decision: Dict[str, Any],
        data: pd.DataFrame,
        upbit_client: UpbitClient,
        performance_monitor: PerformanceMonitor,
        # monitoring_system: MonitoringSystem
    ):
    """성능 모니터링 데이터를 업데이트하고 성능 보고서를 생성하여 전송합니다."""
    performance_monitor.record(
        decision=decision,
        current_price=data['close'].iloc[-1],
        balance=upbit_client.get_balance("KRW"),
        btc_amount=upbit_client.get_balance("BTC"),
        params=decision.get('param_adjustment', {}),
        regime=decision.get('risk_assessment', 'unknown'),
        anomalies=decision.get('anomalies', False)
    )
    print("Performance report")
    report = monitoring_system.get_performance_report()
    logger.info(f"Performance report: {report}")
    send_discord_message(f"Performance report: {report}")


if __name__ == "__main__":
    main()
