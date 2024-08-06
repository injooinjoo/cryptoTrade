import json
import logging
import math
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from api_client import UpbitClient, OpenAIClient
from data_preparation import safe_fetch_multi_timeframe_data
from database import initialize_db
from ml_models import MLPredictor, RLAgent
from auto_adjustment import AnomalyDetector, MarketRegimeDetector

logger = logging.getLogger(__name__)
initialize_db()
ml_predictor = MLPredictor()
rl_agent = RLAgent(state_size=5, action_size=3)  # 3 actions: buy, sell, hold


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)

def preprocess_data_for_json(data):
    if isinstance(data, dict):
        return {k: preprocess_data_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [preprocess_data_for_json(v) for v in data]
    elif isinstance(data, (np.integer, np.floating)):
        return data.item()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, pd.DataFrame):
        return data.to_dict(orient='records')
    elif isinstance(data, pd.Timestamp):
        return data.isoformat()
    else:
        return data


def analyze_data_with_gpt4(
        data: pd.DataFrame,
        openai_client: OpenAIClient,
        params: Dict[str, Any],
        upbit_client: UpbitClient,
        average_accuracy: float,
        anomalies: List[bool],
        anomaly_scores: List[float],
        market_regime: str,
        ml_prediction: int,
        rl_action: int,
        backtest_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyze data with GPT-4 and return a trading decision."""
    multi_timeframe_data = safe_fetch_multi_timeframe_data(upbit_client)
    signals = analyze_multi_timeframe_data(multi_timeframe_data)

    # 지표가 계산되지 않았을 경우 대체 전략 사용
    if 'SMA_10' not in data.columns or 'RSI_14' not in data.columns:
        logger.warning("Not enough data for technical indicators. Using alternative strategy.")
        return alternative_strategy(data, upbit_client)

    analysis_data = {
        "market_data": data.to_dict(orient='records'),
        "multi_timeframe_signals": signals,
        "portfolio_state": get_portfolio_state(upbit_client),
        "current_params": params,
        "anomalies": anomalies,
        "anomaly_scores": anomaly_scores,
        "market_regime": market_regime,
        "average_accuracy": average_accuracy,
        "ml_prediction": int(ml_prediction),
        "rl_action": int(rl_action),
        "backtest_results": backtest_results
    }

    instructions = """
    Analyze the provided market data, including multi-timeframe signals, current trading parameters,
    anomaly detection results, market regime information, machine learning predictions, reinforcement learning actions,
    and backtesting results to make a trading decision for Bitcoin (BTC).
    Your response should be a JSON object with the following structure:
    {
        "decision": "buy" or "sell" or "hold",
        "percentage": a number between 0 and 100 representing the percentage of the total portfolio value to use for the trade,
        "target_price": the target price for the trade (use null for hold decisions),
        "stop_loss": a suggested stop-loss price to limit potential losses,
        "take_profit": a suggested take-profit price to secure gains,
        "reasoning": a brief explanation of your decision, including how all provided data influenced your decision,
        "risk_assessment": an evaluation of the current market risk (low, medium, high),
        "param_adjustment": suggestions for parameter adjustments based on current market conditions
    }
    """

    try:
        json_data = json.dumps(analysis_data, cls=NumpyEncoder)
        response = openai_client.chat_completion(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": json_data}
            ],
            max_tokens=500,
            response_format={"type": "json_object"}
        )

        gpt4_advice = json.loads(response.choices[0].message.content)
        logger.info(f"GPT-4o-mini advice: {gpt4_advice}")
        return gpt4_advice
    except Exception as e:
        logger.error(f"Error in GPT-4 analysis: {e}")
        logger.exception("Traceback:")
        return default_hold_decision()


def alternative_strategy(data: pd.DataFrame, upbit_client: UpbitClient) -> Dict[str, Any]:
    """충분한 데이터가 없을 때 사용할 대체 전략"""
    current_price = data['close'].iloc[-1]
    previous_price = data['close'].iloc[-2] if len(data) > 1 else current_price

    if current_price > previous_price:
        decision = 'buy'
        percentage = 10  # 보수적인 매수
    elif current_price < previous_price:
        decision = 'sell'
        percentage = 10  # 보수적인 매도
    else:
        decision = 'hold'
        percentage = 0

    return {
        'decision': decision,
        'percentage': percentage,
        'target_price': current_price,
        'stop_loss': current_price * 0.98,
        'take_profit': current_price * 1.02,
        'reasoning': "Insufficient data for technical analysis. Using simple price comparison.",
        'risk_assessment': 'high',
        'param_adjustment': {}
    }


def round_price_to_tick_size(price: float, tick_size: float = 1000) -> float:
    """Round the price to the nearest tick size."""
    return math.floor(price / tick_size) * tick_size


def execute_trade(upbit_client, order_manager, decision: Dict, current_balance: float):
    trade_type = decision['decision']
    percentage = decision['percentage'] / 100
    target_price = decision['target_price']
    stop_loss = decision['stop_loss']
    take_profit = decision['take_profit']

    current_price = upbit_client.get_current_price("KRW-BTC")

    upbit_client.cancel_existing_orders()

    if trade_type == 'buy':
        # 수수료를 고려한 실제 사용 가능한 금액 계산 (예: 0.05% 수수료)
        available_balance = current_balance * 0.9995
        amount_to_trade = min(available_balance * percentage, available_balance)
        btc_amount = amount_to_trade / current_price

        # 최소 주문 가능 금액 확인 (예: 5000원)
        if amount_to_trade < 5000:
            logger.warning(
                f"Buy order not placed: Amount ({amount_to_trade:.2f} KRW) is less than minimum order amount")
            return False

        try:
            order = upbit_client.buy_limit_order("KRW-BTC", target_price, btc_amount)
            logger.info(f"Buy order placed: {order}")

            if order and 'uuid' in order:
                order_manager.add_order(order['uuid'], 'buy', target_price, stop_loss, take_profit)
                order_manager.start_monitoring()
            return True
        except Exception as e:
            logger.error(f"Error placing buy order: {e}")
            return False

    elif trade_type == 'sell':
        btc_balance = upbit_client.get_balance("BTC")
        btc_amount = btc_balance * percentage

        # 최소 주문 가능 금액 확인 (예: 5000원)
        if btc_amount * current_price < 5000:
            logger.warning(
                f"Sell order not placed: Amount ({btc_amount * current_price:.2f} KRW) is less than minimum order amount")
            return False

        try:
            order = upbit_client.sell_limit_order("KRW-BTC", target_price, btc_amount)
            logger.info(f"Sell order placed: {order}")

            if order and 'uuid' in order:
                order_manager.add_order(order['uuid'], 'sell', target_price, stop_loss, take_profit)
                order_manager.start_monitoring()
            return True
        except Exception as e:
            logger.error(f"Error placing sell order: {e}")
            return False

    else:  # Hold
        logger.info("Decision is to hold. No trade executed.")
        return True



def execute_buy(upbit_client, percentage: float, target_price: float, config: dict,
                use_market_order: bool = False) -> None:
    """Execute a buy order."""
    execute_trade(upbit_client, percentage, target_price, config, "buy", use_market_order)


def execute_sell(upbit_client, percentage: float, target_price: float, config: dict,
                 use_market_order: bool = False) -> None:
    """Execute a sell order."""
    execute_trade(upbit_client, percentage, target_price, config, "sell", use_market_order)


def analyze_multi_timeframe_data(data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, str]]:
    """Analyze data across multiple timeframes and return signals."""
    signals = {}
    for tf, df in data.items():
        if not df.empty:
            signals[tf] = {
                'trend': get_trend(df),
                'volatility': get_volatility(df),
                'momentum': get_momentum(df)
            }
        else:
            signals[tf] = {
                'trend': 'unknown',
                'volatility': 'unknown',
                'momentum': 'unknown'
            }
    return signals


def get_trend(df: pd.DataFrame) -> str:
    """Determine the trend based on SMA and EMA."""
    if 'SMA_10' not in df.columns or 'EMA_10' not in df.columns:
        return 'unknown'

    last_close = df['close'].iloc[-1]
    sma_10 = df['SMA_10'].iloc[-1]
    ema_10 = df['EMA_10'].iloc[-1]

    if pd.isna(sma_10) or pd.isna(ema_10):
        return 'unknown'

    if last_close > sma_10 and last_close > ema_10:
        return 'bullish'
    elif last_close < sma_10 and last_close < ema_10:
        return 'bearish'
    else:
        return 'neutral'


def get_volatility(df: pd.DataFrame) -> str:
    """Determine the volatility based on ATR."""
    if 'ATR_14' not in df.columns:
        return 'unknown'

    current_volatility = df['ATR_14'].iloc[-1]
    avg_volatility = df['ATR_14'].mean()

    if pd.isna(current_volatility) or pd.isna(avg_volatility):
        return 'unknown'

    if current_volatility > avg_volatility * 1.2:
        return 'high'
    elif current_volatility < avg_volatility * 0.8:
        return 'low'
    else:
        return 'normal'


def get_momentum(df: pd.DataFrame) -> str:
    """Determine the momentum based on RSI."""
    if 'RSI_14' not in df.columns:
        return 'unknown'

    rsi = df['RSI_14'].iloc[-1]

    if pd.isna(rsi):
        return 'unknown'

    if rsi > 70:
        return 'overbought'
    elif rsi < 30:
        return 'oversold'
    else:
        return 'neutral'


def process_gpt4_advice(advice: Dict[str, Any], upbit_client: UpbitClient) -> Dict[str, Any]:
    """Process GPT-4 advice and adjust it based on the current portfolio state."""
    current_price = upbit_client.get_current_price("KRW-BTC")
    portfolio_state = get_portfolio_state(upbit_client)

    decision = {
        "decision": advice.get("decision", "hold"),
        "percentage": min(max(float(advice.get("percentage", 0)), 0), 100),
        "target_price": float(advice.get("target_price", current_price) or current_price),
        "stop_loss": float(advice.get("stop_loss", current_price * 0.95) or current_price * 0.95),
        "take_profit": float(advice.get("take_profit", current_price * 1.05) or current_price * 1.05),
        "reasoning": advice.get("reasoning", "No specific reason provided"),
        "risk_assessment": advice.get("risk_assessment", "medium")
    }

    # Adjust percentage based on portfolio state
    if decision["decision"] == "buy":
        max_buy_amount = portfolio_state["krw_balance"]
        adjusted_percentage = (decision["percentage"] / 100) * portfolio_state[
            "total_balance_krw"] / max_buy_amount * 100 if max_buy_amount > 0 else 0
        decision["percentage"] = min(adjusted_percentage, 100)
    elif decision["decision"] == "sell":
        max_sell_amount = portfolio_state["btc_balance"] * current_price
        adjusted_percentage = (decision["percentage"] / 100) * portfolio_state[
            "total_balance_krw"] / max_sell_amount * 100 if max_sell_amount > 0 else 0
        decision["percentage"] = min(adjusted_percentage, 100)

    return decision


def get_portfolio_state(upbit_client: UpbitClient) -> Dict[str, float]:
    """Retrieve the current portfolio state."""
    btc_balance = upbit_client.get_balance("BTC")
    krw_balance = upbit_client.get_balance("KRW")
    btc_current_price = upbit_client.get_current_price("KRW-BTC")
    return {
        "btc_balance": btc_balance,
        "krw_balance": krw_balance,
        "btc_current_price": btc_current_price,
        "total_balance_krw": (btc_balance * btc_current_price) + krw_balance
    }


def default_hold_decision() -> Dict[str, Any]:
    """Return a default hold decision in case of errors."""
    return {
        "decision": "hold",
        "percentage": 0,
        "target_price": None,
        "stop_loss": None,
        "take_profit": None,
        "reasoning": "Default hold decision due to analysis error",
        "risk_assessment": "high",
        "param_adjustment": {}
    }


def prepare_state(data: pd.DataFrame) -> np.ndarray:
    """Prepare the state for the RL agent."""
    return data[['open', 'high', 'low', 'close', 'volume']].values[-1].reshape(1, -1)


def get_reward(action: int, next_price: float, current_price: float) -> float:
    """Calculate the reward for the RL agent based on the action taken."""
    if action == 0:  # buy
        return (next_price - current_price) / current_price
    elif action == 1:  # sell
        return (current_price - next_price) / current_price
    else:  # hold
        return 0


def trading_strategy(
        data: pd.DataFrame,
        params: Dict[str, Any],
        openai_client: OpenAIClient,
        upbit_client: UpbitClient,
        strategy_weights: Dict[str, float],
        anomaly_detector: AnomalyDetector,
        regime_detector: MarketRegimeDetector,
        rl_agent: RLAgent
) -> Dict[str, Any]:
    """Combine ML, RL, and GPT-4 to form a trading strategy."""
    ml_prediction = ml_predictor.predict(data)

    # Prepare state for RL agent
    state = prepare_state(data)
    rl_action = rl_agent.act(state)

    # Get next state and reward for RL learning
    next_data = data.shift(-1).dropna()
    if not next_data.empty:
        next_state = prepare_state(next_data)
        next_price = next_data['close'].iloc[0]
        current_price = data['close'].iloc[-1]
        reward = get_reward(rl_action, next_price, current_price)
        done = False
    else:
        next_state = state
        reward = 0
        done = True

    # Learn from this experience
    rl_agent.remember(state[0], rl_action, reward, next_state[0], done)

    # Perform batch learning if enough experiences are collected
    if len(rl_agent.memory) >= rl_agent.batch_size:
        rl_agent.replay(rl_agent.batch_size)

    anomalies, anomaly_scores = anomaly_detector.detect_anomalies(data)
    current_regime = regime_detector.detect_regime(data)

    gpt4_advice = analyze_data_with_gpt4(
        data, openai_client, params, upbit_client, strategy_weights, anomalies, anomaly_scores, current_regime
    )

    # Combine ML, RL, and GPT-4 decisions
    ml_decision = 'buy' if ml_prediction == 1 else 'sell'
    rl_decision = ['buy', 'sell', 'hold'][rl_action]
    gpt4_decision = gpt4_advice['decision']

    # Simple majority voting with weighted influence
    decisions = [
        ml_decision * strategy_weights.get('ml', 1),
        rl_decision * strategy_weights.get('rl', 1),
        gpt4_decision * strategy_weights.get('gpt4', 1)
    ]
    final_decision = max(set(decisions), key=decisions.count)

    if final_decision == 'buy':
        percentage = min(gpt4_advice['percentage'], 100)
        target_price = gpt4_advice['target_price'] or data['close'].iloc[-1] * 1.02
    elif final_decision == 'sell':
        percentage = min(gpt4_advice['percentage'], 100)
        target_price = gpt4_advice['target_price'] or data['close'].iloc[-1] * 0.98
    else:  # hold
        percentage = 0
        target_price = data['close'].iloc[-1]

    return {
        'decision': final_decision,
        'percentage': percentage,
        'target_price': target_price,
        'stop_loss': gpt4_advice['stop_loss'],
        'take_profit': gpt4_advice['take_profit'],
        'ml_prediction': ml_prediction,
        'rl_action': rl_action,
        'gpt4_advice': gpt4_advice,
        'reasoning': gpt4_advice['reasoning'],
        'risk_assessment': gpt4_advice['risk_assessment'],
        'param_adjustment': gpt4_advice['param_adjustment']
    }
