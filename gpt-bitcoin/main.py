import logging
import os
import time
from typing import Dict, Any

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

    # 주기적 작업 설정
    schedule.every(10).minutes.do(trading_loop)
    schedule.every(10).minutes.do(report_performance)
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

        # 트레이딩 전략 실행 및 결정 생성
        decision = execute_trading_strategy(
            data, upbit_client, config, auto_adjuster, anomaly_detector,
            regime_detector, backtest_results, ml_prediction, rl_action
        )
        logger.info(f"Trading decision: {decision}")

        # GPT-4와의 일치도 계산
        gpt4_agreement = 100 if decision['decision'] == decision.get('gpt4_decision', '') else 0

        # 거래 성공 여부 (간단한 예시, 실제로는 더 복잡한 로직이 필요할 수 있음)
        trade_success = decision['decision'] == 'hold' or (
                    decision['decision'] == 'buy' and data['close'].iloc[-1] > data['close'].iloc[-2]) or (
                                    decision['decision'] == 'sell' and data['close'].iloc[-1] < data['close'].iloc[-2])

        # 학습 진행 상황 업데이트
        update_learning_progress(
            ml_accuracy,
            ml_loss,
            rl_agent.epsilon,
            rl_agent.get_average_reward(),
            gpt4_agreement,
            trade_success
        )

        # 학습 진행 상황 보고
        report_learning_progress()



        # RL 에이전트 학습
        if len(data) > 1:
            next_state = prepare_state(data.iloc[-1:])
            reward = calculate_reward(decision, data)
            done = False
            rl_agent.remember(state[0], rl_action, reward, next_state[0], done)
            if len(rl_agent.memory) > rl_agent.batch_size:
                rl_agent.replay(rl_agent.batch_size)

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
            decision.get('anomalies_detected', False)
        )

    except Exception as e:
        logger.error(f"Error in trading loop: {e}")
        logger.exception("Traceback:")


def execute_trading_strategy(
        data: pd.DataFrame,
        upbit_client: UpbitClient,
        config: Dict[str, Any],
        auto_adjuster: AutoAdjustment,
        anomaly_detector: AnomalyDetector,
        regime_detector: MarketRegimeDetector,
        backtest_results: Dict[str, Any],
        ml_prediction: int,
        rl_action: int
) -> Dict[str, Any]:
    """Execute the trading strategy and return a trading decision."""

    # 이전 결정 평가
    evaluate_previous_decision(upbit_client)

    # 최근 정확도를 반영한 데이터 분석
    average_accuracy = analyze_database_for_decision()

    # 현재 시장 상태 분석
    anomalies, anomaly_scores = anomaly_detector.detect_anomalies(data)
    current_regime = regime_detector.detect_regime(data)
    current_params = auto_adjuster.adjust_params()

    # GPT-4를 이용한 분석 및 조언 얻기
    gpt4_advice = analyze_data_with_gpt4(
        data, openai_client, current_params, upbit_client, average_accuracy,
        anomalies, anomaly_scores, current_regime, ml_prediction, rl_action, backtest_results
    )

    # 최종 결정 생성
    final_decision = generate_final_decision(gpt4_advice, ml_prediction, rl_action, current_regime, backtest_results)

    # 거래 실행
    if final_decision['decision'] == 'buy':
        execute_buy(upbit_client, final_decision['percentage'], final_decision['target_price'], config)
    elif final_decision['decision'] == 'sell':
        execute_sell(upbit_client, final_decision['percentage'], final_decision['target_price'], config)

    # 반환값에 anomalies 정보 추가
    final_decision['anomalies_detected'] = any(anomalies)
    # GPT-4의 결정을 final_decision에 추가
    final_decision['gpt4_decision'] = gpt4_advice['decision']
    return final_decision


def generate_final_decision(gpt4_advice, ml_prediction, rl_action, current_regime, backtest_results):
    """Generate the final trading decision based on all available information."""
    decision_weights = {
        'gpt4': 0.4,
        'ml': 0.3,
        'rl': 0.3
    }

    def decision_to_number(decision):
        if isinstance(decision, str):
            return {'buy': 1, 'hold': 0, 'sell': -1}.get(decision.lower(), 0)
        return decision

    gpt4_decision = decision_to_number(gpt4_advice['decision'])
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
    if backtest_results['sharpe_ratio'] < 0.5:
        final_decision = 'hold'  # 리스크가 높을 경우 홀드

    return {
        'decision': final_decision,
        'percentage': gpt4_advice['percentage'],
        'target_price': gpt4_advice['target_price'],
        'stop_loss': gpt4_advice['stop_loss'],
        'take_profit': gpt4_advice['take_profit'],
        'reasoning': f"GPT-4: {gpt4_advice['decision']}, ML: {ml_prediction}, RL: {rl_action}, Regime: {current_regime}",
        'risk_assessment': gpt4_advice['risk_assessment']
    }


def calculate_reward(decision, data):
    if decision['decision'] == 'buy':
        return (data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]
    elif decision['decision'] == 'sell':
        return (data['close'].iloc[-2] - data['close'].iloc[-1]) / data['close'].iloc[-2]
    else:
        return 0


def report_performance():
    summary = performance_monitor.get_performance_summary()
    accuracy = performance_monitor.get_prediction_accuracy()
    message = f"📊 Performance Summary (Last 10 minutes):\n{summary}\n\nPrediction Accuracy: {accuracy:.2f}%"
    send_discord_message(message)


def evaluate_previous_decision(upbit_client: UpbitClient):
    """이전 결정과 실제 결과를 비교하여 데이터베이스에 저장."""
    previous_decision = get_previous_decision()
    if not previous_decision:
        logger.info("No previous decision found to evaluate.")
        return

    actual_price = upbit_client.get_current_price("KRW-BTC")
    accuracy = 0

    if previous_decision['decision'] == 'buy' and actual_price >= previous_decision['target_price']:
        accuracy = 1
    elif previous_decision['decision'] == 'sell' and actual_price <= previous_decision['target_price']:
        accuracy = 1

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


# 전역 변수로 학습 진행 상황 추적
learning_progress = {
    'ml_model': {'accuracy': 0, 'loss': 0},
    'rl_agent': {'epsilon': 1.0, 'average_reward': 0},
    'gpt4_agreement': 0,
    'total_trades': 0,
    'successful_trades': 0
}


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

    success_rate = (learning_progress['successful_trades'] / learning_progress['total_trades'] * 100
                    if learning_progress['total_trades'] > 0 else 0)

    report = f"""
    📊 Learning Progress Report 📊
    
    ML Model:
    - Accuracy: {learning_progress['ml_model']['accuracy']:.2f}%
    - Loss: {learning_progress['ml_model']['loss']:.4f}
    
    RL Agent:
    - Epsilon: {learning_progress['rl_agent']['epsilon']:.4f}
    - Average Reward: {learning_progress['rl_agent']['average_reward']:.2f}
    
    GPT-4 Agreement Rate: {learning_progress['gpt4_agreement']:.2f}%
    
    Trading Performance:
    - Total Trades: {learning_progress['total_trades']}
    - Successful Trades: {learning_progress['successful_trades']}
    - Success Rate: {success_rate:.2f}%
    
    Keep learning and improving! 🚀
    """

    send_discord_message(report)



if __name__ == "__main__":
    main()