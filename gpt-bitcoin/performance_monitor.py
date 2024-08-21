import json
import logging
import time
from typing import Dict, Any
import pandas as pd
from tabulate import tabulate

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    def __init__(self, upbit_client, file_path='performance_data.csv'):
        self.upbit_client = upbit_client
        self.file_path = file_path

        # 초기 잔액 설정
        self.initial_balance = self.upbit_client.get_balance("KRW")
        self.initial_btc_balance = self.upbit_client.get_balance("BTC")
        self.initial_btc_price = self.upbit_client.get_current_price("KRW-BTC")

        # 현재 잔액 초기화
        self.current_balance = self.initial_balance
        self.current_btc_balance = self.initial_btc_balance

        # 초기 총 가치 계산
        self.initial_total_value = self.calculate_total_value()

        self.total_trades = 0
        self.successful_trades = 0
        self.total_profit = 0

        self.model_accuracies = {model: {'correct': 0, 'total': 0, 'accuracy': 0.0} for model in
                                 ['gpt', 'ml', 'xgboost', 'rl', 'lstm', 'arima', 'prophet', 'transformer']}

        self.trade_history = []
        self.prediction_history = []

        self.load_data()

    def calculate_total_value(self):
        btc_price = self.upbit_client.get_current_price("KRW-BTC")
        return self.current_balance + (self.current_btc_balance * btc_price)

    def update_balances(self):
        self.current_balance = self.upbit_client.get_balance("KRW")
        self.current_btc_balance = self.upbit_client.get_balance("BTC")

    def record_trade(self, trade_info: Dict[str, Any]):
        self.trade_history.append(trade_info)
        self.total_trades += 1
        if trade_info.get('success', False):
            self.successful_trades += 1
        self.total_profit += trade_info.get('profit', 0)
        self.update_balances()
        self.save_data()

    def update_prediction_accuracy(self, model: str, is_correct: bool):
        if model not in self.model_accuracies:
            self.model_accuracies[model] = {'correct': 0, 'total': 0, 'accuracy': 0.0}

        self.model_accuracies[model]['total'] += 1
        if is_correct:
            self.model_accuracies[model]['correct'] += 1

        total = self.model_accuracies[model]['total']
        correct = self.model_accuracies[model]['correct']
        self.model_accuracies[model]['accuracy'] = (correct / total) * 100 if total > 0 else 0.0

        self.save_data()

    def get_all_model_accuracies(self) -> Dict[str, float]:
        return {model: data['accuracy'] for model, data in self.model_accuracies.items()}

    def record_prediction(self, predictions: Dict[str, float], actual_price: float):
        self.prediction_history.append((predictions, actual_price))
        if len(self.prediction_history) > 100:  # Keep only last 100 predictions
            self.prediction_history.pop(0)
        self.save_data()

    def calculate_returns(self):
        current_total_value = self.calculate_total_value()
        return ((current_total_value - self.initial_total_value) / self.initial_total_value) * 100

    def calculate_sharpe_ratio(self):
        if len(self.trade_history) < 2:
            return 0

        returns = [(trade['profit'] / trade['amount']) for trade in self.trade_history]
        return (pd.Series(returns).mean() / pd.Series(returns).std()) * (252 ** 0.5)  # Annualized

    def get_performance_summary(self) -> str:
        current_total_value = self.calculate_total_value()
        returns = self.calculate_returns()
        sharpe_ratio = self.calculate_sharpe_ratio()

        summary = "\n📊 트레이딩 성능 요약 📊\n"
        summary += "=" * 50 + "\n\n"

        summary += "1. 수익률\n"
        returns_data = [
            ["총 수익률", f"{returns:.2f}%"],
            ["Sharpe Ratio", f"{sharpe_ratio:.2f}"]
        ]
        summary += tabulate(returns_data, headers=["지표", "값"], tablefmt="grid") + "\n\n"

        summary += "2. 모델별 성능\n"
        model_data = []
        for model, data in self.model_accuracies.items():
            model_data.append([model.upper(), f"{data['accuracy']:.2f}%", f"{data['correct']}/{data['total']}"])
        summary += tabulate(model_data, headers=["모델", "정확도", "예측성공/총예측"], tablefmt="grid") + "\n\n"

        summary += "3. 자산 현황\n"
        asset_data = [
            ["초기 자산", f"{self.initial_total_value:,.0f} KRW"],
            ["현재 자산", f"{current_total_value:,.0f} KRW"],
            ["현재 KRW", f"{self.current_balance:,.0f} KRW"],
            ["현재 BTC", f"{self.current_btc_balance:.8f} BTC"]
        ]
        summary += tabulate(asset_data, headers=["항목", "값"], tablefmt="grid") + "\n\n"

        summary += "4. 거래 통계\n"
        trade_data = [
            ["총 거래 횟수", self.total_trades],
            ["성공한 거래", self.successful_trades],
            ["성공률", f"{(self.successful_trades / self.total_trades * 100):.2f}%" if self.total_trades > 0 else "N/A"],
            ["총 수익", f"{self.total_profit:,.0f} KRW"]
        ]
        summary += tabulate(trade_data, headers=["지표", "값"], tablefmt="grid") + "\n\n"

        summary += f"\n마지막 업데이트: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"

        return summary

    def save_data(self):
        data = {
            'initial_balance': self.initial_balance,
            'initial_btc_balance': self.initial_btc_balance,
            'initial_btc_price': self.initial_btc_price,
            'initial_total_value': self.initial_total_value,
            'current_balance': self.current_balance,
            'current_btc_balance': self.current_btc_balance,
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'total_profit': self.total_profit,
            'model_accuracies': self.model_accuracies,
            'trade_history': self.trade_history,
            'prediction_history': self.prediction_history
        }
        with open(self.file_path, 'w') as f:
            json.dump(data, f)

    def load_data(self):
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            self.initial_balance = data['initial_balance']
            self.initial_btc_balance = data['initial_btc_balance']
            self.initial_btc_price = data['initial_btc_price']
            self.initial_total_value = data['initial_total_value']
            self.current_balance = data['current_balance']
            self.current_btc_balance = data['current_btc_balance']
            self.total_trades = data['total_trades']
            self.successful_trades = data['successful_trades']
            self.total_profit = data['total_profit']
            self.model_accuracies = data['model_accuracies']
            self.trade_history = data['trade_history']
            self.prediction_history = data['prediction_history']
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("No existing performance data found. Starting with fresh data.")

    def log_prediction_stats(self):
        logger.info("현재 모델별 예측 횟수 및 정확도:")
        for model, data in self.model_accuracies.items():
            logger.info(f"  {model}: {data['total']}회 예측, 정확도 {data['accuracy']:.2f}%")