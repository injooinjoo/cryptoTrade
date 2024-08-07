import json
import logging
import math
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from api_client import UpbitClient, OpenAIClient
from auto_adjustment import AnomalyDetector, MarketRegimeDetector
from database import initialize_db, get_recent_decisions
from discord_notifier import send_discord_message
from ml_models import MLPredictor, RLAgent

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


def evaluate_decisions(recent_decisions, current_price):
    """
    Evaluate the performance of recent trading decisions.
    """
    if not recent_decisions:
        return {
            "overall_assessment": "No recent decisions to evaluate",
            "patterns_identified": "N/A",
            "reasons_for_performance": "N/A",
            "future_improvements": "Start making trading decisions",
            "adjustments_needed": "N/A"
        }

    correct_decisions = 0
    total_decisions = len(recent_decisions)
    profit_loss = 0

    for decision in recent_decisions:
        if 'decision' not in decision or 'btc_krw_price_at_decision' not in decision:
            logger.warning(f"Invalid decision format: {decision}")
            continue

        if decision['decision'] == 'buy':
            if current_price > decision['btc_krw_price_at_decision']:
                correct_decisions += 1
                profit_loss += (current_price - decision['btc_krw_price_at_decision']) / decision['btc_krw_price_at_decision']
            else:
                profit_loss -= (decision['btc_krw_price_at_decision'] - current_price) / decision['btc_krw_price_at_decision']
        elif decision['decision'] == 'sell':
            if current_price < decision['btc_krw_price_at_decision']:
                correct_decisions += 1
                profit_loss += (decision['btc_krw_price_at_decision'] - current_price) / decision['btc_krw_price_at_decision']
            else:
                profit_loss -= (current_price - decision['btc_krw_price_at_decision']) / decision['btc_krw_price_at_decision']



    accuracy = correct_decisions / total_decisions if total_decisions > 0 else 0
    avg_profit_loss = profit_loss / total_decisions if total_decisions > 0 else 0

    if accuracy >= 0.7:
        overall_assessment = "Good performance"
    elif accuracy >= 0.5:
        overall_assessment = "Average performance"
    else:
        overall_assessment = "Needs improvement"

    patterns = "Consistently profitable" if avg_profit_loss > 0 else "Consistently unprofitable" if avg_profit_loss < 0 else "No clear pattern"

    evaluation = {
        "overall_assessment": overall_assessment,
        "patterns_identified": patterns,
        "reasons_for_performance": f"Accuracy: {accuracy:.2f}, Average profit/loss: {avg_profit_loss:.2f}",
        "future_improvements": "Focus on improving accuracy" if accuracy < 0.7 else "Maintain current strategy",
        "adjustments_needed": "Reassess risk management" if avg_profit_loss < 0 else "Continue current approach"
    }

    return evaluation


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
    # 최근 24시간 데이터만 사용
    recent_data = data.tail(24)

    # 주요 지표 계산
    current_price = recent_data['close'].iloc[-1]
    sma_10 = recent_data['SMA_10'].iloc[-1]
    ema_10 = recent_data['EMA_10'].iloc[-1]
    rsi_14 = recent_data['RSI_14'].iloc[-1]
    bb_upper = recent_data['BB_Upper'].iloc[-1]
    bb_lower = recent_data['BB_Lower'].iloc[-1]

    # 추세 및 기술적 지표 해석
    trend = "Bullish" if current_price > sma_10 else "Bearish"
    rsi_state = "Oversold" if rsi_14 < 30 else "Overbought" if rsi_14 > 70 else "Neutral"
    bb_state = "Lower Band Touch" if current_price <= bb_lower else "Upper Band Touch" if current_price >= bb_upper else "Inside Bands"

    # 거래량 분석
    avg_volume = recent_data['volume'].mean()
    current_volume = recent_data['volume'].iloc[-1]
    volume_ratio = current_volume / avg_volume

    # 이상치 요약
    significant_anomalies = [score for score in anomaly_scores if abs(score) > 0.5]

    market_analysis = {
        "current_price": current_price,
        "sma_10_diff": (current_price - sma_10) / sma_10 * 100,
        "ema_10_diff": (current_price - ema_10) / ema_10 * 100,
        "rsi_14": rsi_14,
        "rsi_state": rsi_state,
        "bb_state": bb_state,
        "trend": trend,
        "volume_ratio": volume_ratio,
        "significant_anomalies": significant_anomalies
    }

    # 최근 결정 평가
    recent_decisions = get_recent_decisions(5)  # 최근 5개 결정 가져오기
    decision_evaluation = evaluate_decisions(recent_decisions, current_price)

    analysis_data = {
        "market_analysis": market_analysis,
        "decision_evaluation": decision_evaluation,
        "current_regime": market_regime,
        "current_params": params,
        "ml_prediction": ml_prediction,
        "rl_action": rl_action,
        "backtest_summary": summarize_backtest_results(backtest_results)
    }
    # 데이터 전처리 및 JSON 직렬화
    preprocessed_data = preprocess_data_for_json(analysis_data)
    json_data = json.dumps(preprocessed_data, cls=NumpyEncoder, indent=2)

    # Discord로 요청 내용 전송
    send_discord_message(f"GPT-4 Request Data:\n```json\n{json_data}\n```")

    # instructions-v5.md 파일 읽기
    with open('instructions_v5.md', 'r') as file:
        instructions_v5 = file.read()

    instructions = f"""
    {instructions_v5}

    Based on the provided data and instructions, analyze the market conditions and make a trading decision for the KRW-BTC pair. Your response should be a JSON object with the following structure:
    {{
        "decision": "buy" or "sell" or "hold",
        "percentage": a number between 0 and 100 representing the percentage of the total portfolio value to use for the trade,
        "target_price": the target price for the trade (use null for hold decisions),
        "stop_loss": a suggested stop-loss price to limit potential losses,
        "take_profit": a suggested take-profit price to secure gains,
        "reasoning": a detailed explanation of your decision, including:
            - Which technical indicators were most influential in your decision
            - How you considered multiple timeframes
            - How you applied risk management principles
            - How past decision performance influenced your current decision
            - Your interpretation of the current market regime and anomalies
            - How you balanced the ML prediction and RL action with other factors
        "risk_assessment": an evaluation of the current market risk (low, medium, high),
        "short_term_prediction": your prediction for the price movement in the next 10 minutes (increase, decrease, or stable),
        "param_adjustment": suggestions for parameter adjustments based on current market conditions and past performance
    }}
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


def summarize_backtest_results(backtest_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize the backtest results for easier interpretation.
    """
    summary = {
        "total_return": backtest_results.get("total_return", 0),
        "sharpe_ratio": backtest_results.get("sharpe_ratio", 0),
        "max_drawdown": backtest_results.get("max_drawdown", 0),
        "win_rate": backtest_results.get("win_rate", 0),
        "average_trade_duration": backtest_results.get("average_trade_duration", 0),
        "total_trades": backtest_results.get("total_trades", 0),
    }

    # Interpret the results
    if summary["total_return"] > 0:
        summary["performance"] = "Positive"
    elif summary["total_return"] < 0:
        summary["performance"] = "Negative"
    else:
        summary["performance"] = "Neutral"

    if summary["sharpe_ratio"] > 1:
        summary["risk_adjusted_performance"] = "Good"
    elif summary["sharpe_ratio"] > 0:
        summary["risk_adjusted_performance"] = "Moderate"
    else:
        summary["risk_adjusted_performance"] = "Poor"

    return summary


def calculate_stochastic_oscillator(data, k_period=14, d_period=3):
    """
    Calculate the Stochastic Oscillator for the given data.
    """
    low_min = data['low'].rolling(window=k_period).min()
    high_max = data['high'].rolling(window=k_period).max()

    k = 100 * (data['close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()

    return k.iloc[-1], d.iloc[-1]  # Return the most recent K and D values


def calculate_market_sentiment(data):
    """
    Calculate a simple market sentiment indicator based on price momentum.
    """
    returns = data['close'].pct_change()
    sentiment = returns.rolling(window=10).mean() / returns.rolling(window=10).std()
    return sentiment.iloc[-1] * 100  # Scale to a 0-100 range


def calculate_price_divergence(data):
    """
    Calculate price divergence from a simple moving average.
    """
    sma = data['close'].rolling(window=20).mean()
    divergence = (data['close'] - sma) / sma * 100
    return divergence.iloc[-1]


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


def execute_trade(
    upbit_client,
    percentage: float,
    target_price: float,
    config: dict,
    trade_type: str,
    use_market_order: bool = False
) -> bool:
    """Execute a buy or sell order based on the trade type."""
    try:
        krw_balance = upbit_client.get_balance("KRW")
        btc_balance = upbit_client.get_balance("BTC")
        btc_current_price = upbit_client.get_current_price("KRW-BTC")
        total_balance_krw = (btc_balance * btc_current_price) + krw_balance

        amount_to_invest = total_balance_krw * (percentage / 100)
        amount_to_invest = min(amount_to_invest, krw_balance * 0.9995)  # Adjust for potential fees

        min_krw_balance = config.get('min_krw_balance', 10000)  # Default to 10,000 KRW if not specified
        min_transaction_amount = config.get('min_transaction_amount', 5000)  # Default to 5,000 KRW if not specified

        if trade_type == "buy" and krw_balance < min_krw_balance:
            logger.info(
                f"Buy order not placed: KRW balance ({krw_balance}) is less than {min_krw_balance} KRW")
            return False

        amount_to_buy = amount_to_invest / target_price if trade_type == "buy" else amount_to_invest / btc_current_price

        if use_market_order:
            logger.info(f"Placing market {trade_type} order for {amount_to_invest} KRW")
            if trade_type == "buy":
                result = upbit_client.buy_market_order("KRW-BTC", amount_to_invest)
            else:
                result = upbit_client.sell_market_order("KRW-BTC", amount_to_buy)
        else:
            if target_price is None:
                logger.info(f"{trade_type.capitalize()} order not placed: No target price specified for limit order")
                return False
            target_price = round_price_to_tick_size(target_price)

            if trade_type == "buy" and amount_to_invest > min_transaction_amount:
                result = upbit_client.buy_limit_order("KRW-BTC", target_price, amount_to_buy)
            elif trade_type == "sell" and amount_to_buy * target_price > min_transaction_amount:
                result = upbit_client.sell_limit_order("KRW-BTC", target_price, amount_to_buy)
            else:
                logger.info(f"{trade_type.capitalize()} order not placed: Amount too small.")
                return False

        if result is None:
            logger.warning(f"{trade_type.capitalize()} order placement returned None.")
            return False
        else:
            logger.info(f"{trade_type.capitalize()} order placed successfully: {result}")
            return True

    except Exception as e:
        logger.error(f"Failed to execute {trade_type} order: {e}")
        logger.exception("Traceback:")
        return False


def execute_buy(upbit_client, percentage: float, target_price: float, config: dict,
                use_market_order: bool = False) -> bool:
    """Execute a buy order."""
    return execute_trade(upbit_client, percentage, target_price, config, "buy", use_market_order)

def execute_sell(upbit_client, percentage: float, target_price: float, config: dict,
                 use_market_order: bool = False) -> bool:
    """Execute a sell order."""
    return execute_trade(upbit_client, percentage, target_price, config, "sell", use_market_order)


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
