import logging
from typing import Dict, Any, List, Optional

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
                           anomaly_scores: List[float], market_regime: str, ml_prediction: Optional[int],
                           xgboost_prediction: Optional[int], rl_action: Optional[int],
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

    predictions = {
        'gpt': current_price * 1.01,  # GPT의 예측은 별도로 처리해야 할 수 있습니다.
        'ml': current_price * (1 + 0.01 if ml_prediction == 1 else -0.01 if ml_prediction == 0 else 0),
        'xgboost': current_price * (1 + 0.01 if xgboost_prediction == 1 else -0.01 if xgboost_prediction == 0 else 0),
        'rl': current_price * (1 + 0.01 if rl_action == 2 else -0.01 if rl_action == 0 else 0),
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


def execute_trade(upbit_client: UpbitClient, decision: Dict[str, Any], config: Dict[str, Any],
                  current_btc_holding: bool) -> Dict[str, Any]:
    try:
        if decision['decision'] == 'hold':
            return {"success": True, "message": "Hold position", "uuid": None, "has_btc": current_btc_holding}

        current_price = upbit_client.get_current_price("KRW-BTC")
        logger.info(f"Current BTC price: {current_price}")

        if current_price is None or current_price == 0:
            logger.error(f"Invalid current price: {current_price}")
            return {"success": False, "message": "Invalid current price", "uuid": None, "has_btc": current_btc_holding}

        min_trade_amount = config.get('min_trade_amount', 5000)
        max_trade_ratio = config.get('max_trade_ratio', 0.99)
        fee_rate = config.get('fee_rate', 0.0005)

        krw_balance = upbit_client.get_balance("KRW")
        btc_balance = upbit_client.get_balance("BTC")
        total_asset_value = krw_balance + (btc_balance * current_price)

        logger.info(f'총 자산 가치: {total_asset_value:.2f} KRW (KRW: {krw_balance:.2f}, BTC: {btc_balance:.8f})')

        trade_amount = min(total_asset_value * max_trade_ratio, total_asset_value * (decision['percentage'] / 100))
        trade_amount = trade_amount * (1 - fee_rate)  # 수수료를 고려한 거래 금액 계산

        if decision['decision'] == 'buy':
            max_buy_amount = min(trade_amount, krw_balance * (1 - fee_rate))
            amount_to_buy = max_buy_amount / current_price
            amount_to_buy = round(amount_to_buy, 8)  # Upbit BTC 거래 단위에 맞춰 조정

            logger.info(f'매수 시도: {amount_to_buy:.8f} BTC (약 {max_buy_amount:.2f} KRW)')

            if max_buy_amount < min_trade_amount:
                return {"success": False, "message": f"매수 금액이 최소 거래 금액({min_trade_amount} KRW)보다 작습니다.", "uuid": None, "has_btc": current_btc_holding}

            result = upbit_client.buy_limit_order("KRW-BTC", current_price, amount_to_buy)
            logger.info(f"Buy order result: {result}")
            if result and 'uuid' in result:
                print('Buy order placed')
                return {"success": True, "message": "Buy order placed", "uuid": result['uuid'], "amount": amount_to_buy, "price": current_price, "has_btc": True}
            else:
                print('Failed to place buy order')
                error_message = result.get('error', {}).get('message', 'Unknown error occurred')
                logger.error(f"Failed to place buy order: {error_message}")
                return {"success": False, "message": f"Failed to place buy order: {error_message}", "uuid": None, "has_btc": current_btc_holding}

        elif decision['decision'] == 'sell':
            amount_to_sell = min(btc_balance, trade_amount / current_price)
            amount_to_sell = round(amount_to_sell, 8)  # Upbit BTC 거래 단위에 맞춰 조정

            logger.info(f'매도 시도: {amount_to_sell:.8f} BTC (약 {amount_to_sell * current_price:.2f} KRW)')

            if amount_to_sell * current_price < min_trade_amount:
                return {"success": False, "message": f"매도 금액이 최소 거래 금액({min_trade_amount} KRW)보다 작습니다.", "uuid": None, "has_btc": current_btc_holding}

            result = upbit_client.sell_limit_order("KRW-BTC", current_price, amount_to_sell)
            logger.info(f"Sell order result: {result}")
            if result and 'uuid' in result:
                print('Sell order placed')
                return {"success": True, "message": "Sell order placed", "uuid": result['uuid'], "amount": amount_to_sell, "price": current_price, "has_btc": btc_balance > amount_to_sell}
            else:
                print('Failed to place sell order')
                error_message = result.get('error', {}).get('message', 'Unknown error occurred')
                logger.error(f"Failed to place sell order: {error_message}")
                return {"success": False, "message": f"Failed to place sell order: {error_message}", "uuid": None, "has_btc": current_btc_holding}

    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        logger.exception("Traceback:")
        return {"success": False, "message": f"Error: {str(e)}", "uuid": None, "has_btc": current_btc_holding}


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


import json
from typing import Dict, Any
import numpy as np


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

    def save_weights(self):
        with open(self.weight_file, 'w') as f:
            json.dump(self.weights, f)

    def make_decision(self, predictions: Dict[str, float], current_price: float) -> Dict[str, Any]:
        weighted_diff = sum(self.weights[model] * (price - current_price) for model, price in predictions.items())
        expected_price = current_price + weighted_diff

        decision = 'buy' if expected_price > current_price else 'sell'
        percentage = self.calculate_trade_percentage(expected_price, current_price)

        decision_info = {
            'decision': decision,
            'percentage': percentage,
            'target_price': expected_price,
            'current_price': current_price,
            'predictions': predictions
        }
        self.decision_history.append(decision_info)

        return decision_info

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

    def update_weights(self, actual_price: float):
        if not self.decision_history:
            return

        last_decision = self.decision_history[-1]
        predictions = last_decision['predictions']

        for model, predicted_price in predictions.items():
            if (predicted_price > last_decision['current_price'] and actual_price > last_decision['current_price']) or \
                    (predicted_price < last_decision['current_price'] and actual_price < last_decision[
                        'current_price']):
                self.weights[model] = min(self.weights[model] + 0.02, 1.0)
            else:
                self.weights[model] = max(self.weights[model] - 0.02, 0.0)

        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {model: weight / total_weight for model, weight in self.weights.items()}

        self.save_weights()

    def execute_trade(self, upbit_client: UpbitClient, decision: Dict[str, Any]) -> Dict[str, Any]:
        try:
            current_price = upbit_client.get_current_price("KRW-BTC")
            krw_balance = upbit_client.get_balance("KRW")
            btc_balance = upbit_client.get_balance("BTC")

            trade_amount = min(krw_balance, btc_balance * current_price) * (decision['percentage'] / 100)
            trade_amount = trade_amount * (1 - self.config['fee_rate'])

            if decision['decision'] == 'buy':
                amount_to_buy = trade_amount / current_price
                amount_to_buy = round(amount_to_buy, 8)  # Upbit BTC 거래 단위에 맞춰 조정

                if trade_amount < self.config['min_trade_amount']:
                    return {"success": False, "message": f"매수 금액이 최소 거래 금액보다 작습니다.", "uuid": None}

                result = upbit_client.buy_limit_order("KRW-BTC", current_price, amount_to_buy)
            else:  # sell
                amount_to_sell = min(btc_balance, trade_amount / current_price)
                amount_to_sell = round(amount_to_sell, 8)

                if amount_to_sell * current_price < self.config['min_trade_amount']:
                    return {"success": False, "message": f"매도 금액이 최소 거래 금액보다 작습니다.", "uuid": None}

                result = upbit_client.sell_limit_order("KRW-BTC", current_price, amount_to_sell)

            if result and 'uuid' in result:
                return {"success": True, "message": f"{decision['decision']} order placed", "uuid": result['uuid']}
            else:
                error_message = result.get('error', {}).get('message', 'Unknown error occurred')
                return {"success": False, "message": f"Failed to place order: {error_message}", "uuid": None}

        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}", "uuid": None}