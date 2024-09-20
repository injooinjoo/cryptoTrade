import json
import logging
from typing import Dict, Any
from typing import List, Optional

import numpy as np
import pandas as pd

from api_client import UpbitClient, OpenAIClient
from config import load_config
from data_manager import DataManager

logger = logging.getLogger(__name__)

config = load_config()
upbit_client = UpbitClient(config['upbit_access_key'], config['upbit_secret_key'])
data_manager = DataManager(upbit_client=upbit_client)


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


def calculate_weighted_price(predictions: Dict[str, float], weights: Dict[str, float], current_price: float) -> float:
    total_weight = sum(weights.values())
    weighted_diff = sum(weights[model] * (price - current_price) for model, price in predictions.items())
    return current_price + (weighted_diff / total_weight)


def determine_trade_ratio(expected_price: float, current_price: float) -> float:
    price_diff_ratio = abs(expected_price - current_price) / current_price

    if price_diff_ratio < 0.01:
        return 0.1
    elif price_diff_ratio < 0.02:
        return 0.2
    elif price_diff_ratio < 0.03:
        return 0.3
    elif price_diff_ratio < 0.04:
        return 0.4
    else:
        return 0.5


def make_weighted_decision(predictions: Dict[str, float], weights: Dict[str, float], current_price: float) -> Dict[
    str, Any]:
    expected_price = calculate_weighted_price(predictions, weights, current_price)
    trade_ratio = determine_trade_ratio(expected_price, current_price)

    if expected_price > current_price:
        decision = 'buy'
    elif expected_price < current_price:
        decision = 'sell'
    else:
        decision = 'hold'

    return {
        'decision': decision,
        'percentage': trade_ratio * 100,
        'target_price': expected_price,
        'stop_loss': current_price * 0.98 if decision == 'buy' else current_price * 1.02,
        'take_profit': current_price * 1.02 if decision == 'buy' else current_price * 0.98,
        'reasoning': f"가중 평균 예상 가격 {expected_price:.0f}원 기반 결정. 현재 가격: {current_price:.0f}원",
        'risk_assessment': "medium",
        'short_term_prediction': "increase" if expected_price > current_price else "decrease",
    }


def analyze_data_with_gpt4(data: pd.DataFrame, openai_client: OpenAIClient, params: Dict[str, Any],
                           upbit_client: UpbitClient, average_accuracy: float, anomalies: List[bool],
                           anomaly_scores: List[float], market_regime: str, ml_prediction: Optional[float],
                           xgboost_prediction: Optional[float], rl_action: Optional[float],
                           lstm_prediction: Optional[float], arima_prediction: Optional[float],
                           prophet_prediction: Optional[float], transformer_prediction: Optional[float],
                           backtest_results: Dict[str, Any], market_analysis: Dict[str, Any],
                           current_balance: float, current_btc_balance: float, hodl_performance: float,
                           current_performance: float, trading_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    recent_data = data.tail(144)
    current_price = float(recent_data['close'].iloc[-1])

    market_analysis.update({
        "current_price": current_price,
        "sma_60": float(recent_data['sma'].fillna(current_price).iloc[-1]),
        "ema_60": float(recent_data['ema'].fillna(current_price).iloc[-1]),
        "rsi_14": float(recent_data['rsi'].fillna(50).iloc[-1]),
        "bb_upper": float(recent_data['BB_Upper'].fillna(current_price).iloc[-1]),
        "bb_lower": float(recent_data['BB_Lower'].fillna(current_price).iloc[-1]),
    })

    # rl_action 처리
    rl_prediction = current_price
    if rl_action is not None:
        if isinstance(rl_action, (float, int)):
            rl_prediction = rl_action
        elif isinstance(rl_action, (np.ndarray, list)) and len(rl_action) > 0:
            if rl_action[0] == 2:
                rl_prediction = current_price * 1.01
            elif rl_action[0] == 0:
                rl_prediction = current_price * 0.99

    predictions = {
        'gpt': current_price * 1.01,  # GPT의 예측은 별도로 처리해야 할 수 있습니다.
        'ml': ml_prediction if ml_prediction is not None else current_price,
        'xgboost': xgboost_prediction if xgboost_prediction is not None else current_price,
        'rl': rl_prediction,
        'lstm': lstm_prediction if lstm_prediction is not None else current_price,
        'arima': arima_prediction if arima_prediction is not None else current_price,
        'prophet': prophet_prediction if prophet_prediction is not None else current_price,
        'transformer': transformer_prediction if transformer_prediction is not None else current_price
    }

    weights = {
        'gpt': 0.2, 'ml': 0.1, 'xgboost': 0.1, 'rl': 0.1, 'lstm': 0.2,
        'arima': 0.1, 'prophet': 0.1, 'transformer': 0.1
    }

    final_decision = make_weighted_decision(predictions, weights, current_price)

    analysis_data = {
        "market_analysis": market_analysis,
        "predictions": predictions,
        "weights": weights,
        "current_balance": current_balance,
        "current_btc_balance": current_btc_balance,
        "hodl_performance": hodl_performance,
        "current_performance": current_performance,
        "trading_history": trading_history,
    }

    return final_decision


def get_instructions(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            instructions = file.read()
        return instructions
    except FileNotFoundError:
        logger.error("Instructions file not found.")
    except Exception as e:
        logger.error(f"An error occurred while reading the instructions file: {e}")
    return ""


class TradingDecisionMaker:
    def __init__(self, config: Dict[str, Any], weight_file: str = 'model_weights.json'):
        self.config = config
        self.weight_file = weight_file
        self.weights = self.load_weights()
        self.decision_history = []

    def load_weights(self) -> Dict[str, float]:
        try:
            with open(self.weight_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'gpt': 0.2, 'ml': 0.1, 'xgboost': 0.1, 'rl': 0.1, 'lstm': 0.2,
                'arima': 0.1, 'prophet': 0.1, 'transformer': 0.1
            }

    def make_decision(self, predictions: Dict[str, float], current_price: float) -> Dict[str, Any]:
        valid_predictions = self.validate_predictions(predictions, current_price)
        weighted_diff = sum(self.weights[model] * (price - current_price) for model, price in valid_predictions.items())
        total_weight = sum(self.weights[model] for model in valid_predictions)
        expected_price = current_price + (weighted_diff / total_weight if total_weight > 0 else 0)

        decision = 'buy' if expected_price > current_price else 'sell'
        percentage = self.calculate_trade_percentage(expected_price, current_price)

        decision_info = {
            'decision': decision,
            'percentage': percentage,
            'target_price': expected_price,
            'current_price': current_price,
            'predictions': valid_predictions,
            'weights': {model: self.weights[model] for model in valid_predictions}
        }
        self.decision_history.append(decision_info)

        return decision_info

    def validate_predictions(self, predictions: Dict[str, float], current_price: float) -> Dict[str, float]:
        valid_predictions = {}
        for model, price in predictions.items():
            if isinstance(price, (int, float)) and 0.5 * current_price <= price <= 2 * current_price:
                valid_predictions[model] = price
            else:
                logger.warning(f"Invalid prediction from {model}: {price}. Using current price instead.")
                valid_predictions[model] = current_price
        return valid_predictions

    def calculate_trade_percentage(self, expected_price: float, current_price: float) -> float:
        price_diff_ratio = abs(expected_price - current_price) / current_price
        if price_diff_ratio < 0.005:
            return 10
        elif price_diff_ratio < 0.01:
            return 20
        elif price_diff_ratio < 0.02:
            return 30
        elif price_diff_ratio < 0.03:
            return 40
        else:
            return 50

    def save_weights(self):
        with open(self.weight_file, 'w') as f:
            json.dump(self.weights, f)
