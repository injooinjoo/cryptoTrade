import logging
import os
import time
from typing import Dict, Any

import numpy as np
import pandas as pd
import schedule
from dotenv import load_dotenv

from api_client import UpbitClient, OpenAIClient, OrderManager, PositionManager
from auto_adjustment import AutoAdjustment, AnomalyDetector, MarketRegimeDetector
from backtesting import run_backtest
from config import load_config, setup_logging
from data_preparation import safe_fetch_multi_timeframe_data
from database import initialize_db, get_previous_decision, update_decision_accuracy, get_recent_decisions
from discord_notifier import send_discord_message
from ml_models import MLPredictor, RLAgent
from performance_monitor import PerformanceMonitor
from trading_logic import analyze_data_with_gpt4, execute_buy, execute_sell, prepare_state, get_reward

# 환경 변수 로드
load_dotenv()

# 로깅 설정
setup_logging()
logger = logging.getLogger(__name__)

# 전역 변수 및 객체 초기화
config = load_config()
upbit_client = UpbitClient(os.getenv("UPBIT_ACCESS_KEY"), os.getenv("UPBIT_SECRET_KEY"))
openai_client = OpenAIClient(os.getenv("OPENAI_API_KEY"))
performance_monitor = PerformanceMonitor()
position_manager = PositionManager(upbit_client)
order_manager = OrderManager(upbit_client, position_manager)
auto_adjuster = AutoAdjustment(config.get('initial_params', {}))
anomaly_detector = AnomalyDetector()
regime_detector = MarketRegimeDetector()
ml_predictor = MLPredictor()
rl_agent = RLAgent(state_size=5, action_size=3)  # 3 actions: buy, sell, hold



# 전역 변수로 학습 진행 상황 추적
learning_progress = {
    'ml_model': {'accuracy': 0, 'loss': 0},
    'rl_agent': {'epsilon': 1.0, 'average_reward': 0},
    'gpt4_agreement': 0,
    'total_trades': 0,
    'successful_trades': 0
}


def main():
    logger.info("Starting the trading bot")

    # 데이터베이스 초기화
    initialize_db()

    # Upbit 연결 확인
    if not upbit_client.check_connection():
        logger.error("Failed to connect to Upbit. Exiting.")
        return

    # 백테스팅 실행
    logger.info("Starting backtesting...")
    historical_data = upbit_client.get_ohlcv("KRW-BTC", interval="day", count=365)  # 1년치 일별 데이터
    backtest_results = run_backtest(historical_data)
    logger.info(f"Backtest results: {backtest_results}")

    # 머신러닝 모델 초기 학습
    ml_predictor.train(historical_data)

    trading_loop()
    # report_performance()

    # 주기적 작업 설정
    schedule.every(10).minutes.do(trading_loop)
    # schedule.every(10).minutes.do(report_performance)
    schedule.every().day.at("00:00").do(performance_monitor.save_to_file)

    # 메인 루프
    logger.info("Starting main loop")
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            logger.exception("Traceback:")


def trading_loop():
    try:
        logger.info("Starting trading loop")
        multi_timeframe_data = safe_fetch_multi_timeframe_data(upbit_client)

        # 가장 짧은 시간대의 데이터를 기본 데이터로 사용
        data = multi_timeframe_data['short']

        # 백테스팅 수행 및 결과 업데이트
        backtest_results = run_backtest(data)

        # ML 모델 재학습 및 예측
        ml_accuracy, ml_loss = ml_predictor.train(data)
        ml_prediction = ml_predictor.predict(data.iloc[-1:])

        # RL 에이전트 학습 및 행동 선택
        state = prepare_state(data.iloc[-1:])
        rl_action = rl_agent.act(state)

        # 이상 탐지 및 시장 체제 분석
        anomalies, anomaly_scores = anomaly_detector.detect_anomalies(data)
        current_regime = regime_detector.detect_regime(data)

        # 이전 결정 평가
        average_accuracy = evaluate_previous_decision(upbit_client)

        # GPT-4를 이용한 분석 및 조언 얻기
        gpt4_advice = analyze_data_with_gpt4(
            data, openai_client, config, upbit_client, average_accuracy,
            anomalies, anomaly_scores, current_regime, ml_prediction, rl_action, backtest_results
        )

        # 최종 결정 생성
        final_decision = generate_final_decision(gpt4_advice, ml_prediction, rl_action, current_regime, backtest_results)

        # 거래 실행
        if final_decision['decision'] == 'buy':
            execute_buy(upbit_client, final_decision['percentage'], final_decision['target_price'], config)
        elif final_decision['decision'] == 'sell':
            execute_sell(upbit_client, final_decision['percentage'], final_decision['target_price'], config)

        # RL 에이전트 학습
        if len(data) > 1:
            next_state = prepare_state(data.iloc[-1:])
            current_portfolio = {
                'btc_balance': upbit_client.get_balance("BTC"),
                'krw_balance': upbit_client.get_balance("KRW")
            }
            reward = calculate_reward(final_decision, data, current_portfolio)
            done = False
            rl_agent.remember(state[0], rl_action, reward, next_state[0], done)
            if len(rl_agent.memory) > rl_agent.batch_size:
                rl_agent.replay(rl_agent.batch_size)

        # 성능 모니터링 데이터 기록
        current_price = upbit_client.get_current_price("KRW-BTC")
        balance = upbit_client.get_balance("KRW")
        btc_amount = upbit_client.get_balance("BTC")
        performance_monitor.record(
            final_decision,
            current_price,
            balance,
            btc_amount,
            final_decision.get('param_adjustment', {}),
            final_decision.get('risk_assessment', 'unknown'),
            any(anomalies)
        )

        # GPT-4의 결정을 final_decision에 추가
        final_decision['gpt4_decision'] = gpt4_advice['decision']

        # 학습 진행 상황 업데이트
        update_learning_progress(
            ml_accuracy,
            ml_loss,
            rl_agent.epsilon,
            rl_agent.get_average_reward(),
            100 if final_decision['decision'] == final_decision['gpt4_decision'] else 0,
            calculate_trade_success(final_decision, data)
        )

        # 학습 진행 상황 보고
        report_learning_progress()

    except Exception as e:
        logger.error(f"Error in trading loop: {e}")
        logger.exception("Traceback:")


def calculate_trade_success(decision, data):
    if decision['decision'] == 'hold':
        return True  # HOLD 결정은 항상 성공으로 간주
    elif decision['decision'] == 'buy':
        return data['close'].iloc[-1] > data['close'].iloc[-2]
    elif decision['decision'] == 'sell':
        return data['close'].iloc[-1] < data['close'].iloc[-2]
    else:
        return False


def calculate_reward(decision: dict, data: pd.DataFrame, current_portfolio: dict) -> float:
    current_price = data['close'].iloc[-1]
    previous_price = data['close'].iloc[-2]
    price_change = (current_price - previous_price) / previous_price

    # 기본 보상: 결정의 정확성
    if decision['decision'] == 'buy' and price_change > 0:
        base_reward = 1
    elif decision['decision'] == 'sell' and price_change < 0:
        base_reward = 1
    elif decision['decision'] == 'hold' and abs(price_change) < 0.005:  # 0.5% 이내 변동
        base_reward = 0.5
    else:
        base_reward = -1

    # 수익률 보상
    profit_reward = 0
    if decision['decision'] == 'buy':
        profit_reward = price_change * decision['percentage'] / 100
    elif decision['decision'] == 'sell':
        profit_reward = -price_change * decision['percentage'] / 100

    # 리스크 관리 보상
    risk_reward = 0
    if 'stop_loss' in decision and 'take_profit' in decision:
        stop_loss_distance = (current_price - decision['stop_loss']) / current_price
        take_profit_distance = (decision['take_profit'] - current_price) / current_price
        if stop_loss_distance > 0 and take_profit_distance > 0:
            risk_reward = min(stop_loss_distance, take_profit_distance)

    # 거래 빈도 조절 보상
    frequency_penalty = -0.1 if decision['decision'] != 'hold' else 0

    # 포트폴리오 다각화 보상
    diversification_reward = 0
    if current_portfolio['btc_balance'] > 0 and current_portfolio['krw_balance'] > 0:
        btc_ratio = current_portfolio['btc_balance'] * current_price / (current_portfolio['btc_balance'] * current_price + current_portfolio['krw_balance'])
        diversification_reward = 1 - abs(0.5 - btc_ratio)  # 50:50 비율에 가까울수록 높은 보상

    # 총 보상 계산
    total_reward = (
        base_reward * 0.4 +
        profit_reward * 2 +
        risk_reward * 0.5 +
        frequency_penalty +
        diversification_reward * 0.3
    )

    # 보상 정규화 (-1에서 1 사이로)
    normalized_reward = np.clip(total_reward, -1, 1)

    return normalized_reward


def evaluate_previous_decision(upbit_client):
    previous_decision = get_previous_decision()
    if not previous_decision:
        logger.info("No previous decision found to evaluate.")
        return 0

    actual_price = upbit_client.get_current_price("KRW-BTC")
    accuracy = 0

    if previous_decision['decision'] == 'buy' and actual_price >= previous_decision['target_price']:
        accuracy = 1
    elif previous_decision['decision'] == 'sell' and actual_price <= previous_decision['target_price']:
        accuracy = 1

    update_decision_accuracy(previous_decision['id'], accuracy)
    logger.info(f"Previous decision evaluated: {'Success' if accuracy == 1 else 'Failure'} with accuracy {accuracy}")

    # 최근 결정들의 평균 정확도 계산
    recent_decisions = get_recent_decisions(days=7)  # 최근 7일간의 결정 가져오기
    if recent_decisions:
        average_accuracy = sum(d['accuracy'] for d in recent_decisions if d['accuracy'] is not None) / len(recent_decisions)
    else:
        average_accuracy = 0

    return average_accuracy


def generate_final_decision(gpt4_advice, ml_prediction, rl_action, current_regime, backtest_results):
    decision_weights = {
        'gpt4': 0.4,
        'ml': 0.3,
        'rl': 0.3
    }

    def decision_to_number(decision):
        if isinstance(decision, str):
            return {'buy': 1, 'hold': 0, 'sell': -1}.get(decision.lower(), 0)
        return decision

    gpt4_decision = decision_to_number(gpt4_advice.get('decision', 'hold'))
    ml_decision = 1 if ml_prediction == 1 else -1
    rl_decision = rl_action - 1  # Convert 0, 1, 2 to -1, 0, 1

    weighted_decision = (
        gpt4_decision * decision_weights['gpt4'] +
        ml_decision * decision_weights['ml'] +
        rl_decision * decision_weights['rl']
    )

    if weighted_decision > 0.1:
        final_decision = 'buy'
    elif weighted_decision < -0.1:
        final_decision = 'sell'
    else:
        final_decision = 'hold'

    # 백테스트 결과를 고려한 리스크 조정
    if backtest_results.get('sharpe_ratio', 0) < 0.5:
        final_decision = 'hold'  # 리스크가 높을 경우 홀드

    # 안전하게 값을 가져오고 기본값 설정
    def safe_float(value, default=0.0):
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    return {
        'decision': final_decision,
        'percentage': min(max(safe_float(gpt4_advice.get('percentage'), 0), 0), 100),
        'target_price': safe_float(gpt4_advice.get('target_price')),
        'stop_loss': safe_float(gpt4_advice.get('stop_loss')),
        'take_profit': safe_float(gpt4_advice.get('take_profit')),
        'reasoning': gpt4_advice.get('reasoning', 'No reasoning provided'),
        'risk_assessment': gpt4_advice.get('risk_assessment', 'unknown')
    }


def report_performance():
    summary = performance_monitor.get_performance_summary()
    accuracy = performance_monitor.get_prediction_accuracy()
    message = f"📊 Performance Summary (Last 10 minutes):\n{summary}\n\nPrediction Accuracy: {accuracy:.2f}%"
    # send_discord_message(message)


def update_learning_progress(ml_accuracy, ml_loss, rl_epsilon, rl_reward, gpt4_agreement, trade_success):
    global learning_progress
    learning_progress['ml_model']['accuracy'] = ml_accuracy
    learning_progress['ml_model']['loss'] = ml_loss
    learning_progress['rl_agent']['epsilon'] = rl_epsilon
    learning_progress['rl_agent']['average_reward'] = rl_reward
    learning_progress['gpt4_agreement'] = gpt4_agreement
    learning_progress['total_trades'] += 1
    if trade_success:
        learning_progress['successful_trades'] += 1


def report_learning_progress():
    global learning_progress

    cumulative_summary = performance_monitor.get_cumulative_summary()
    detailed_analysis = performance_monitor.get_detailed_analysis()
    success_rate = (cumulative_summary['successful_trades'] / cumulative_summary['total_trades'] * 100
                    if cumulative_summary['total_trades'] > 0 else 0)
    performance_comparison = performance_monitor.get_performance_comparison()

    report = f"""
    📊 트레이딩 성과 비교 📊
    트레이딩 수익률: {performance_comparison['trading_return']:.2f}%
    HODL 수익률: {performance_comparison['hodl_return']:.2f}%
    초과 성과: {performance_comparison['outperformance']:.2f}%

    📈 상세 학습 진행 보고서 📈

    머신러닝 모델: 정확도: {learning_progress['ml_model']['accuracy']:.2f}% | 손실: {learning_progress['ml_model']['loss']:.4f}
    강화학습 에이전트: 엡실론: {learning_progress['rl_agent']['epsilon']:.4f} | 평균 보상: {learning_progress['rl_agent']['average_reward']:.2f}
    GPT-4 일치율: {learning_progress['gpt4_agreement']:.2f}%

    누적 트레이딩 성과:
    - 총 거래 횟수: {cumulative_summary['total_trades']}회
    - 매수 거래: {cumulative_summary['buy_trades']}회
    - 매도 거래: {cumulative_summary['sell_trades']}회
    - 홀딩 결정: {cumulative_summary['hold_decisions']}회
    - 평균 거래 규모: {cumulative_summary['avg_trade_size']:.2f}%
    - 가격 변동: {cumulative_summary['price_change']:.2f}%
    - 잔고 변동: {cumulative_summary['balance_change']:.2f}%
    - 성공률: {success_rate:.2f}%

    예측 분석:
    상승 예측 (매수/BTC 보유 시 홀딩): 총 : {detailed_analysis['up_predictions']['total']}회 | 성공: {detailed_analysis['up_predictions']['successful']}회 | 정확도: {detailed_analysis['up_predictions']['accuracy']:.2f}%
    하락 예측 (매도/BTC 미보유 시 홀딩): 총 : {detailed_analysis['down_predictions']['total']}회 | 성공: {detailed_analysis['down_predictions']['successful']}회 | 정확도: {detailed_analysis['down_predictions']['accuracy']:.2f}%
    전체 예측 정확도: {detailed_analysis['overall_accuracy']:.2f}%
    실패 분석: 원인: {detailed_analysis['failure_reasons']}| 개선: {detailed_analysis['improvement_suggestions']}

    계속해서 학습하고 개선해 나가겠습니다! 🚀
    """

    send_discord_message(report)





if __name__ == "__main__":
    main()