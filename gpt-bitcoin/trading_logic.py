import json
import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from api_client import UpbitClient, OpenAIClient
from auto_adjustment import AnomalyDetector, MarketRegimeDetector
from data_manager import DataManager
from backtesting_and_ml import RLAgent, MLPredictor
from config import load_config


logger = logging.getLogger(__name__)

# 설정 로드
config = load_config()

# UpbitClient 인스턴스 생성 (api_client.py에서 가져와야 함)
upbit_client = UpbitClient(config['upbit_access_key'], config['upbit_secret_key'])

# DataManager 인스턴스 생성
data_manager = DataManager(
    upbit_client=upbit_client,
    twitter_api_key=config.get('twitter_api_key'),
    twitter_api_secret=config.get('twitter_api_secret'),
    twitter_access_token=config.get('twitter_access_token'),
    twitter_access_token_secret=config.get('twitter_access_token_secret')
)

DataManager.initialize_db()  # 클래스 메서드로 호출
data_manager.initialize_twitter_api()  # Twitter API 초기화

ml_predictor = MLPredictor()
rl_agent = RLAgent(state_size=5, action_size=3)  # 3 actions: buy, sell, hold


def analyze_data_with_gpt4(
        data: pd.DataFrame,
        openai_client: OpenAIClient,
        params: Dict[str, Any],
        upbit_client: UpbitClient,
        average_accuracy: float,
        anomalies: List[bool],
        anomaly_scores: List[float],
        market_regime: str,
        ml_prediction: Optional[int],
        rl_action: Optional[int],
        backtest_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyze data with GPT-4 and return a trading decision."""
    # 최근 24시간 데이터만 사용
    recent_data = data.tail(144)  # 10분 간격으로 24시간 = 144 데이터 포인트

    # 주요 지표 계산 (안전하게)
    current_price = recent_data['close'].iloc[-1]
    sma_60 = recent_data['SMA'].iloc[-1] if 'SMA' in recent_data.columns else current_price
    ema_60 = recent_data['EMA'].iloc[-1] if 'EMA' in recent_data.columns else current_price
    rsi_14 = recent_data['RSI'].iloc[-1] if 'RSI' in recent_data.columns else 50
    bb_upper = recent_data['BB_Upper'].iloc[-1] if 'BB_Upper' in recent_data.columns else current_price * 1.02
    bb_lower = recent_data['BB_Lower'].iloc[-1] if 'BB_Lower' in recent_data.columns else current_price * 0.98

    # 추세 및 기술적 지표 해석
    trend = "Bullish" if current_price > sma_60 else "Bearish"
    rsi_state = "Oversold" if rsi_14 < 30 else "Overbought" if rsi_14 > 70 else "Neutral"
    bb_state = "Lower Band Touch" if current_price <= bb_lower else "Upper Band Touch" if current_price >= bb_upper else "Inside Bands"

    market_analysis = get_market_analysis(recent_data)
    decision_evaluation = evaluate_decisions(DataManager.get_recent_decisions(5), current_price)

    # 이상 징후 분석
    anomaly_analysis = {
        "anomalies_detected": sum(anomalies),
        "average_anomaly_score": sum(anomaly_scores) / len(anomaly_scores) if anomaly_scores else 0
    }

    # ML 및 RL 예측 결과
    ml_rl_predictions = {
        "ml_prediction": "Up" if ml_prediction == 1 else "Down" if ml_prediction == 0 else "Unknown",
        "rl_action": ["Sell", "Hold", "Buy"][rl_action] if rl_action is not None else "Unknown"
    }

    analysis_data = {
        "market_analysis": market_analysis,
        "decision_evaluation": decision_evaluation,
        "current_params": params,
        "average_accuracy": average_accuracy,
        "anomaly_analysis": anomaly_analysis,
        "market_regime": market_regime,
        "ml_rl_predictions": ml_rl_predictions,
        "backtest_summary": summarize_backtest_results(backtest_results),
        "current_price": current_price,
        "trend": trend,
        "rsi_state": rsi_state,
        "bb_state": bb_state
    }


    instructions_path = "instructions_v5.md"
    instructions = get_instructions(instructions_path)

    instructions2 = get_gpt4_instructions()
    final_instruction = f'{instructions} {instructions2}'
    try:
        response = openai_client.chat_completion(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": final_instruction},
                {"role": "user", "content": json.dumps(analysis_data)}
            ],
            max_tokens=500,
            response_format={"type": "json_object"}
        )

        gpt4_advice = json.loads(response.choices[0].message.content)
        logger.info(f"GPT-4 advice: {gpt4_advice}")
        return gpt4_advice
    except Exception as e:
        logger.error(f"Error in GPT-4 analysis: {e}")
        logger.exception("Traceback:")
        return default_hold_decision()


def summarize_backtest_results(backtest_results: Dict[str, Any]) -> Dict[str, Any]:
    """백테스트 결과를 요약합니다."""
    return {
        "total_return": backtest_results.get("total_return", 0),
        "sharpe_ratio": backtest_results.get("sharpe_ratio", 0),
        "max_drawdown": backtest_results.get("max_drawdown", 0),
        "win_rate": backtest_results.get("win_rate", 0),
    }


def get_market_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    current_price = data['close'].iloc[-1]
    sma_60 = data['SMA'].iloc[-1] if 'SMA' in data.columns else current_price
    ema_60 = data['EMA'].iloc[-1] if 'EMA' in data.columns else current_price
    rsi_14 = data['RSI'].iloc[-1] if 'RSI' in data.columns else 50

    return {
        "current_price": current_price,
        "sma_60_diff": (current_price - sma_60) / sma_60 * 100,
        "ema_60_diff": (current_price - ema_60) / ema_60 * 100,
        "rsi_14": rsi_14,
        "rsi_state": get_rsi_state(rsi_14),
        "bb_state": get_bb_state(data),
        "trend": "Bullish" if current_price > sma_60 else "Bearish",
        "volume_ratio": get_volume_ratio(data),
    }


def get_rsi_state(rsi: float) -> str:
    if rsi < 30:
        return "Oversold"
    elif rsi > 70:
        return "Overbought"
    else:
        return "Neutral"


def get_volume_ratio(data: pd.DataFrame) -> float:
    avg_volume = data['volume'].mean()
    current_volume = data['volume'].iloc[-1]
    return current_volume / avg_volume if avg_volume != 0 else 1.0


def get_bb_state(data: pd.DataFrame) -> str:
    """볼린저 밴드 상태를 확인합니다."""
    if data['close'].iloc[-1] <= data['BB_Lower'].iloc[-1]:
        return "Lower Band Touch"
    elif data['close'].iloc[-1] >= data['BB_Upper'].iloc[-1]:
        return "Upper Band Touch"
    return "Inside Bands"


def evaluate_decisions(recent_decisions: List[Dict[str, Any]], current_price: float) -> Dict[str, Any]:
    """최근 거래 결정을 평가합니다."""
    if not recent_decisions:
        return {"overall_assessment": "No recent decisions to evaluate"}

    correct_decisions = sum(1 for d in recent_decisions if is_decision_correct(d, current_price))
    accuracy = correct_decisions / len(recent_decisions)

    return {
        "overall_assessment": get_assessment(accuracy),
        "accuracy": accuracy
    }


def is_decision_correct(decision: Dict[str, Any], current_price: float) -> bool:
    """개별 거래 결정의 정확성을 평가합니다."""
    if decision['decision'] == 'buy':
        return current_price > decision['btc_krw_price']
    elif decision['decision'] == 'sell':
        return current_price < decision['btc_krw_price']
    return abs(current_price - decision['btc_krw_price']) / decision['btc_krw_price'] < 0.01


def get_assessment(accuracy: float) -> str:
    """정확도에 따른 전반적인 평가를 반환합니다."""
    if accuracy >= 0.7:
        return "Good performance"
    elif accuracy >= 0.5:
        return "Average performance"
    return "Needs improvement"


def get_gpt4_instructions() -> str:
    """GPT-4에 제공할 지시사항을 반환합니다."""
    return """
    분석 제공된 시장 데이터를 면밀히 검토하고 KRW-BTC 페어에 대한 구체적인 거래 결정을 내리세요.
    다음 사항들을 고려하여 상세한 분석과 함께 결정을 제시해 주세요:
    1. 단기(1시간), 중기(1일), 장기(1주일) 시장 동향
    2. 현재 시장 변동성과 그에 따른 리스크
    3. 기술적 지표 (RSI, MACD, 볼린저 밴드 등)의 신호
    4. 최근 거래 성과와 그 이유
    5. 현재 포트폴리오 상태와 리스크 노출도
    response in response_format json
    """
def get_instructions(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            instructions = file.read()
        return instructions
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("An error occurred while reading the file:", e)


def default_hold_decision() -> Dict[str, Any]:
    """기본 홀드 결정을 반환합니다."""
    return {
        "decision": "hold",
        "percentage": 0,
        "target_price": None,
        "stop_loss": None,
        "take_profit": None,
        "reasoning": "Default hold decision due to analysis error"
    }


def trading_strategy(data: pd.DataFrame, params: Dict[str, Any], openai_client: OpenAIClient,
                     upbit_client: UpbitClient) -> Dict[str, Any]:
    """전체 거래 전략을 실행합니다."""
    ml_prediction = ml_predictor.predict(data)
    rl_action = rl_agent.act(prepare_state(data))
    gpt4_advice = analyze_data_with_gpt4(data, openai_client, params, upbit_client)

    # 각 모델의 결정을 결합
    decisions = {
        'ml': 'buy' if ml_prediction == 1 else 'sell',
        'rl': ['buy', 'sell', 'hold'][rl_action],
        'gpt4': gpt4_advice['decision']
    }

    final_decision = max(set(decisions.values()), key=list(decisions.values()).count)

    return {
        'decision': final_decision,
        'percentage': gpt4_advice['percentage'],
        'target_price': gpt4_advice['target_price'],
        'stop_loss': gpt4_advice['stop_loss'],
        'take_profit': gpt4_advice['take_profit'],
        'reasoning': gpt4_advice['reasoning']
    }


def prepare_state(data: pd.DataFrame) -> np.ndarray:
    """RL 에이전트를 위한 상태를 준비합니다."""
    return data[['open', 'high', 'low', 'close', 'volume']].values[-1].reshape(1, -1)


def execute_trade(upbit_client: UpbitClient, decision: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """거래를 실행합니다."""
    try:
        # 결정을 확인하여 hold일 경우 percentage는 0이어야 함
        if decision['decision'] == 'hold':
            if decision.get('percentage', 0) != 0:
                logger.error("Hold decision cannot have a non-zero percentage.")
                return False
            logger.info("Holding position, no trade executed.")
            return True  # hold decision

        # 결정을 확인하여 buy 또는 sell일 경우 percentage가 0이 아니어야 함
        if decision['decision'] in ['buy', 'sell']:
            if decision.get('percentage', 0) == 0:
                logger.error(f"{decision['decision'].capitalize()} decision cannot have a zero percentage.")
                return False

        if decision['decision'] == 'buy':
            return execute_buy(upbit_client, decision['percentage'], decision['target_price'], config)
        elif decision['decision'] == 'sell':
            return execute_sell(upbit_client, decision['percentage'], decision['target_price'], config)

        return False  # 예상치 못한 decision 값에 대해서는 False 반환

    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return False


def execute_buy(upbit_client: UpbitClient, percentage: float, target_price: Optional[float], config: Dict[str, Any]) -> bool:
    """매수 주문을 실행합니다."""
    try:
        krw_balance = upbit_client.get_balance("KRW")
        if krw_balance is None or krw_balance == 0:
            logger.warning("KRW 잔고가 없습니다.")
            return False

        if target_price is None:
            logger.warning("목표 가격이 설정되지 않았습니다. 현재 시장 가격으로 매수를 시도합니다.")
            target_price = upbit_client.get_current_price("KRW-BTC")

        if target_price is None:
            logger.error("현재 시장 가격을 가져올 수 없습니다.")
            return False

        amount_to_buy = (krw_balance * percentage / 100) / target_price

        # min_trade_amount 기본값을 설정 (예: 5000 KRW)
        min_trade_amount = config.get('min_trade_amount', 5000)

        if amount_to_buy * target_price < min_trade_amount:
            logger.info(f"매수 금액이 최소 거래 금액({min_trade_amount} KRW)보다 작습니다.")
            return False

        result = upbit_client.buy_limit_order("KRW-BTC", target_price, amount_to_buy)
        if result:
            logger.info(f"매수 주문 성공: {amount_to_buy:.8f} BTC at {target_price} KRW")
            return True
        else:
            logger.warning("매수 주문 실패")
            return False
    except Exception as e:
        logger.error(f"매수 주문 실행 중 오류 발생: {e}")
        return False


def execute_sell(upbit_client: UpbitClient, percentage: float, target_price: Optional[float], config: Dict[str, Any]) -> bool:
    """매도 주문을 실행합니다."""
    try:
        btc_balance = upbit_client.get_balance("BTC")
        if btc_balance is None or btc_balance == 0:
            logger.warning("BTC 잔고가 없습니다.")
            return False

        if target_price is None:
            logger.warning("목표 가격이 설정되지 않았습니다. 현재 시장 가격으로 매도를 시도합니다.")
            target_price = upbit_client.get_current_price("KRW-BTC")

        if target_price is None:
            logger.error("현재 시장 가격을 가져올 수 없습니다.")
            return False

        amount_to_sell = btc_balance * percentage / 100

        # min_trade_amount 기본값을 설정 (예: 5000 KRW)
        min_trade_amount = config.get('min_trade_amount', 5000)

        if amount_to_sell * target_price < min_trade_amount:
            logger.info(f"매도 금액이 최소 거래 금액({min_trade_amount} KRW)보다 작습니다.")
            return False

        result = upbit_client.sell_limit_order("KRW-BTC", target_price, amount_to_sell)
        if result:
            logger.info(f"매도 주문 성공: {amount_to_sell:.8f} BTC at {target_price} KRW")
            return True
        else:
            logger.warning("매도 주문 실패")
            return False
    except Exception as e:
        logger.error(f"매도 주문 실행 중 오류 발생: {e}")
        return False