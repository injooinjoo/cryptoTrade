import json
import logging
import math
import time
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from api_client import UpbitClient, OpenAIClient
from backtesting_and_ml import RLAgent, MLPredictor
from config import load_config
from data_manager import DataManager

logger = logging.getLogger(__name__)

# 설정 로드
config = load_config()

# UpbitClient 인스턴스 생성 (api_client.py에서 가져와야 함)
upbit_client = UpbitClient(config['upbit_access_key'], config['upbit_secret_key'])

# DataManager 인스턴스 생성
data_manager = DataManager(
    upbit_client=upbit_client,
)

ml_predictor = MLPredictor()
rl_agent = RLAgent(state_size=5, action_size=3)  # 3 actions: buy, sell, hold


def numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    else:
        return obj


def analyze_data_with_gpt4(data: pd.DataFrame, openai_client: OpenAIClient, params: Dict[str, Any],
                           upbit_client: UpbitClient, average_accuracy: float, anomalies: List[bool],
                           anomaly_scores: List[float], market_regime: str, ml_prediction: Optional[int],
                           xgboost_prediction: Optional[int], rl_action: Optional[int],
                           lstm_prediction: Optional[int], backtest_results: Dict[str, Any],
                           market_analysis: Dict[str, Any], current_balance: float,
                           current_btc_balance: float, hodl_performance: float,
                           current_performance: float, trading_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    recent_data = data.tail(144)
    current_price = float(recent_data['close'].iloc[-1])

    sma_60 = float(recent_data['sma'].fillna(current_price).infer_objects(copy=False).iloc[-1])
    ema_60 = float(recent_data['ema'].fillna(current_price).infer_objects(copy=False).iloc[-1])
    rsi_14 = float(recent_data['rsi'].fillna(50).infer_objects(copy=False).iloc[-1])
    bb_upper = float(recent_data['BB_Upper'].fillna(current_price).infer_objects(copy=False).iloc[-1])
    bb_lower = float(recent_data['BB_Lower'].fillna(current_price).infer_objects(copy=False).iloc[-1])

    trend = "Bullish" if current_price > sma_60 and current_price > ema_60 else "Bearish" if current_price < sma_60 and current_price < ema_60 else "Neutral"
    rsi_state = "Oversold" if rsi_14 < 30 else "Overbought" if rsi_14 > 70 else "Neutral"
    bb_state = "Lower Band Touch" if current_price <= bb_lower else "Upper Band Touch" if current_price >= bb_upper else "Inside Bands"

    market_analysis.update({
        "current_price": current_price,
        "sma_60": sma_60,
        "ema_60": ema_60,
        "rsi_14": rsi_14,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "trend": trend,
        "rsi_state": rsi_state,
        "bb_state": bb_state,
        "sma_diff": float((current_price - sma_60) / sma_60 * 100),
        "ema_diff": float((current_price - ema_60) / ema_60 * 100),
    })

    decision_evaluation = evaluate_decisions(data_manager.get_recent_decisions(5), current_price)

    anomaly_analysis = {
        "anomalies_detected": int(sum(anomalies)),
        "average_anomaly_score": float(sum(anomaly_scores) / len(anomaly_scores) if anomaly_scores else 0)
    }

    ml_rl_predictions = {
        "ml_prediction": "Up" if ml_prediction == 1 else "Down" if ml_prediction == 0 else "Unknown",
        "xgboost_prediction": "Up" if xgboost_prediction == 1 else "Down" if xgboost_prediction == 0 else "Unknown",
        "rl_action": ["Sell", "Hold", "Buy"][rl_action] if rl_action is not None else "Unknown",
        "lstm_prediction": "Up" if lstm_prediction == 1 else "Down" if lstm_prediction == 0 else "Unknown"
    }

    # Aggregate all analysis data
    analysis_data = {
        "market_analysis": market_analysis,
        "ml_prediction": ml_prediction,
        "xgboost_prediction": xgboost_prediction,
        "rl_action": rl_action,
        "lstm_prediction": lstm_prediction,
        "decision_evaluation": decision_evaluation,
        "current_params": params,
        "average_accuracy": float(average_accuracy),
        "anomaly_analysis": anomaly_analysis,
        "market_regime": market_regime,
        "ml_rl_predictions": ml_rl_predictions,
        "backtest_summary": summarize_backtest_results(backtest_results),
        "current_balance": current_balance,
        "current_btc_balance": current_btc_balance,
        "hodl_performance": hodl_performance,
        "current_performance": current_performance,
        "trading_history": trading_history,
    }

    # numpy 타입을 Python 기본 타입으로 변환
    analysis_data = numpy_to_python(analysis_data)
    instruction = get_instructions('instructions_v5.md')
    # pprint(f'{instruction}{get_gpt4_instructions()}')
    # pprint(analysis_data)
    try:
        response = openai_client.chat_completion(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f'{instruction}{get_gpt4_instructions()}'},
                {"role": "user", "content": json.dumps(analysis_data)}
            ],
            max_tokens=500,
            response_format={"type": "json_object"}
        )

        try:
            gpt4_advice = json.loads(response.choices[0].message.content)
            print(gpt4_advice)
            # 'decision' 키가 없는 경우 기본값 설정
            if 'decision' not in gpt4_advice:
                gpt4_advice['decision'] = 'hold'

            return gpt4_advice
        except json.JSONDecodeError as json_error:
            logger.error(f"JSON 파싱 오류: {json_error}")
            logger.error(f"GPT-4 응답 내용: {response.choices[0].message.content}")
            gpt4_advice = default_hold_decision()

            # GPT-4 응답 구조 변환
        if 'final_decision' in gpt4_advice and isinstance(gpt4_advice['final_decision'], dict):
            gpt4_advice['decision'] = gpt4_advice['final_decision'].get('action', 'hold')
        else:
            gpt4_advice['decision'] = 'hold'

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
    if 'BB_Lower' not in data.columns or 'BB_Upper' not in data.columns:
        return "BB_Lower or BB_Upper not found in data"

    current_price = data['close'].iloc[-1]
    bb_lower = data['BB_Lower'].iloc[-1]
    bb_upper = data['BB_Upper'].iloc[-1]

    if pd.isna(bb_lower) or pd.isna(bb_upper):
        return "BB values are NaN"

    if current_price <= bb_lower:
        return "Lower Band Touch"
    elif current_price >= bb_upper:
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

        결정이 'hold'인 경우에도 잠재적인 'buy' 또는 'sell' 시나리오에 대비하여 매수/매도 가격과 비율을 제공해 주세요.

        response JSON format 
        {
        "decision": "buy" or "sell" or "hold",
        "percentage": number between 0 and 100 (cannot be 0 for buy/sell decisions),
        "target_price": number (cannot be null for buy/sell decisions),
        "stop_loss": number (must be realistic for the next 10 minutes),
        "take_profit": number (must be realistic for the next 10 minutes),
        "reasoning": "Detailed explanation of your decision",
        "risk_assessment": "low", "medium", or "high",
        "short_term_prediction": "increase", "decrease", or "stable",
        "param_adjustment": {
            "param1": new_value,
            "param2": new_value
        },
        "potential_buy": {
            "percentage": number between 0 and 100,
            "target_price": number
        },
        "potential_sell": {
            "percentage": number between 0 and 100,
            "target_price": number
        }
        }
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
    return {
        "decision": "hold",
        "percentage": 0,
        "target_price": None,
        "stop_loss": None,
        "take_profit": None,
        "reasoning": "기본 홀드 결정 (GPT-4 분석 오류)",
        "final_decision": {
            "action": "hold",
            "rationale": "GPT-4 분석 중 오류가 발생하여 기본적으로 홀드 결정을 내렸습니다."
        }
    }


def trading_strategy(data: pd.DataFrame, params: Dict[str, Any], openai_client: OpenAIClient,
                     upbit_client: UpbitClient) -> Dict[str, Any]:
    end_time = int(time.time())
    start_time = end_time - (60 * 60 * 24)  # 24시간 데이터
    data = data_manager.get_data_for_training(start_time, end_time)

    ml_prediction = ml_predictor.predict(data)
    rl_action = rl_agent.act(prepare_state(data))
    gpt4_advice = analyze_data_with_gpt4(data, openai_client, params, upbit_client,
                                         data_manager.get_average_accuracy(), [], [], "", ml_prediction, rl_action, {})

    ml_accuracy = ml_predictor.get_accuracy()
    rl_reward = rl_agent.get_average_reward()
    gpt_accuracy = data_manager.get_average_accuracy()

    weights = calculate_weights(ml_accuracy, rl_reward, gpt_accuracy)

    decisions = {
        'ml': 1 if ml_prediction == 1 else -1,
        'rl': 1 if rl_action == 2 else -1 if rl_action == 0 else 0,
        'gpt4': 1 if gpt4_advice['decision'] == 'buy' else -1 if gpt4_advice['decision'] == 'sell' else 0
    }

    weighted_decision = sum(decisions[model] * weight for model, weight in weights.items())

    final_decision = 'buy' if weighted_decision > 0 else 'sell' if weighted_decision < 0 else 'hold'

    return {
        'decision': final_decision,
        'percentage': gpt4_advice['percentage'],
        'target_price': gpt4_advice['target_price'],
        'stop_loss': gpt4_advice['stop_loss'],
        'take_profit': gpt4_advice['take_profit'],
        'reasoning': f"Weighted decision based on ML ({weights['ml']:.2f}), RL ({weights['rl']:.2f}), and GPT-4 ({weights['gpt4']:.2f}). {gpt4_advice['reasoning']}"
    }


def prepare_state(data: pd.DataFrame) -> np.ndarray:
    """RL 에이전트를 위한 상태를 준비합니다."""
    return data[['open', 'high', 'low', 'close', 'volume']].values[-1].reshape(1, -1)


def execute_trade(upbit_client: UpbitClient, decision: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if decision['decision'] == 'hold':
            return {"success": True, "message": "Hold position", "uuid": None}

        current_price = upbit_client.get_current_price("KRW-BTC")
        if current_price is None:
            return {"success": False, "message": "Failed to get current price", "uuid": None}

        min_trade_amount = config.get('min_trade_amount', 5000)

        if decision['decision'] == 'buy':
            krw_balance = upbit_client.get_balance("KRW")
            if krw_balance is None:
                return {"success": False, "message": "Failed to get KRW balance", "uuid": None}

            amount_to_buy = (krw_balance * decision['percentage'] / 100) / current_price
            if amount_to_buy * current_price < min_trade_amount:
                logger.warning(
                    f"Buy amount ({amount_to_buy * current_price:.2f} KRW) is less than minimum trade amount ({min_trade_amount} KRW)")
                return {"success": False, "message": f"매수 금액이 최소 거래 금액({min_trade_amount} KRW)보다 작습니다.", "uuid": None}

            result = execute_buy(upbit_client, amount_to_buy, current_price, config)
        else:  # sell
            btc_balance = upbit_client.get_balance("BTC")
            if btc_balance is None:
                return {"success": False, "message": "Failed to get BTC balance", "uuid": None}

            amount_to_sell = btc_balance * decision['percentage'] / 100
            if amount_to_sell * current_price < min_trade_amount:
                logger.warning(
                    f"Sell amount ({amount_to_sell * current_price:.2f} KRW) is less than minimum trade amount ({min_trade_amount} KRW)")
                return {"success": False, "message": f"매도 금액이 최소 거래 금액({min_trade_amount} KRW)보다 작습니다.", "uuid": None}

            result = execute_sell(upbit_client, amount_to_sell, current_price, config)

        if result['success']:
            logger.info(
                f"{decision['decision'].capitalize()} order executed: {result['amount']:.8f} BTC at {result['price']} KRW")
            return {
                "success": True,
                "message": f"{decision['decision'].capitalize()} order executed: {result['amount']:.8f} BTC at {result['price']} KRW",
                "uuid": result['uuid'],
                "amount": result['amount'],
                "price": result['price'],
                "stop_loss": decision.get('stop_loss'),
                "take_profit": decision.get('take_profit')
            }
        else:
            return {"success": False, "message": result['message'], "uuid": None}

    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return {"success": False, "message": f"Error: {str(e)}", "uuid": None}


def execute_buy(upbit_client: UpbitClient, amount_to_buy: float, current_price: float, config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        min_trade_amount = config.get('min_trade_amount', 5000)

        # BTC 주문 개수를 소수점 8자리까지 내림
        amount_to_buy = math.floor(amount_to_buy * 100000000) / 100000000
        # KRW 가격을 정수로 변환
        current_price = int(current_price)

        if amount_to_buy * current_price < min_trade_amount:
            logger.warning(f"매수 금액이 최소 거래 금액({min_trade_amount} KRW)보다 작습니다. 금액: {int(amount_to_buy * current_price)} KRW")
            return {"success": False, "message": f"매수 금액이 최소 거래 금액({min_trade_amount} KRW)보다 작습니다."}

        # KRW 잔고 확인
        krw_balance = int(upbit_client.get_balance("KRW"))
        if krw_balance < int(amount_to_buy * current_price):
            logger.warning(f"KRW 잔고 부족. 필요: {int(amount_to_buy * current_price)} KRW, 보유: {krw_balance} KRW")
            return {"success": False, "message": f"KRW 잔고 부족. 필요: {int(amount_to_buy * current_price)} KRW, 보유: {krw_balance} KRW"}

        print(f'BUY: {current_price} | {amount_to_buy:.8f}')
        result = upbit_client.buy_limit_order("KRW-BTC", current_price, amount_to_buy)
        if result and isinstance(result, dict) and 'uuid' in result:
            logger.info(f"매수 주문 성공: {amount_to_buy:.8f} BTC at {current_price} KRW")
            return {"success": True, "amount": amount_to_buy, "price": current_price, "uuid": result['uuid']}
        else:
            logger.warning(f"매수 주문 실패. 반환된 결과: {result}")
            return {"success": False, "message": f"매수 주문 실패. 상세 정보: {result}"}
    except Exception as e:
        logger.error(f"매수 주문 실행 중 오류 발생: {e}", exc_info=True)
        return {"success": False, "message": str(e)}


def execute_sell(upbit_client: UpbitClient, amount_to_sell: float, current_price: float, config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        min_trade_amount = config.get('min_trade_amount', 5000)

        # BTC 판매 개수를 소수점 8자리까지 내림
        amount_to_sell = math.floor(amount_to_sell * 100000000) / 100000000
        # KRW 가격을 정수로 변환
        current_price = int(current_price)

        sell_amount_krw = int(amount_to_sell * current_price)
        if sell_amount_krw < min_trade_amount:
            logger.warning(f"매도 금액이 최소 거래 금액({min_trade_amount} KRW)보다 작습니다. 금액: {sell_amount_krw} KRW")
            return {"success": False, "message": f"매도 금액이 최소 거래 금액({min_trade_amount} KRW)보다 작습니다."}

        # BTC 잔고 확인
        btc_balance = upbit_client.get_balance("BTC")
        if btc_balance < amount_to_sell:
            logger.warning(f"BTC 잔고 부족. 필요: {amount_to_sell:.8f} BTC, 보유: {btc_balance:.8f} BTC")
            return {"success": False, "message": f"BTC 잔고 부족. 필요: {amount_to_sell:.8f} BTC, 보유: {btc_balance:.8f} BTC"}

        print(f'SELL: {current_price} | {amount_to_sell:.8f}')
        result = upbit_client.sell_limit_order("KRW-BTC", current_price, amount_to_sell)
        if result and isinstance(result, dict) and 'uuid' in result:
            logger.info(f"매도 주문 성공: {amount_to_sell:.8f} BTC at {current_price} KRW")
            return {"success": True, "amount": amount_to_sell, "price": current_price, "uuid": result['uuid']}
        else:
            logger.warning(f"매도 주문 실패. 반환된 결과: {result}")
            return {"success": False, "message": f"매도 주문 실패. 상세 정보: {result}"}
    except Exception as e:
        logger.error(f"매도 주문 실행 중 오류 발생: {e}", exc_info=True)
        return {"success": False, "message": str(e)}

def calculate_weights(ml_accuracy: float, rl_reward: float, gpt_accuracy: float) -> Dict[str, float]:
    total = ml_accuracy + rl_reward + gpt_accuracy
    if total == 0:
        return {'ml': 1 / 3, 'rl': 1 / 3, 'gpt4': 1 / 3}

    weights = {
        'ml': ml_accuracy / total,
        'rl': rl_reward / total,
        'gpt4': gpt_accuracy / total
    }
    return weights