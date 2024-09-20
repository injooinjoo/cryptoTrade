import json
import logging
import time
from collections import deque
from typing import Dict, Any, List
from tabulate import tabulate

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    def __init__(self, upbit_client, file_path='performance_data.json'):
        self.upbit_client = upbit_client
        self.file_path = file_path
        self.prediction_stats = {}
        self.model_weights = {}
        self.prediction_history = deque(maxlen=1000)
        self.weight_history = deque(maxlen=100)

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
        self.total_profit = 0

        self.load_data()

    def calculate_total_value(self):
        btc_price = self.upbit_client.get_current_price("KRW-BTC")
        return (self.current_btc_balance * btc_price) + self.current_balance

    def update_prediction_stats(self, model: str, is_correct: bool):
        if model not in self.prediction_stats:
            self.prediction_stats[model] = {'total': 0, 'correct': 0}

        self.prediction_stats[model]['total'] += 1
        if is_correct:
            self.prediction_stats[model]['correct'] += 1

        # 업데이트 후 즉시 로깅
        total = self.prediction_stats[model]['total']
        correct = self.prediction_stats[model]['correct']
        accuracy = (correct / total * 100) if total > 0 else 0
        logger.info(f"{model.upper()} 모델 예측 업데이트: 정확도 {accuracy:.2f}%, 적중/전체: {correct}/{total}")

        self.save_data()

    def reset_prediction_stats(self, model: str):
        if model in self.prediction_stats:
            self.prediction_stats[model] = {'total': 0, 'correct': 0}
            self.save_data()
            logger.info(f"{model} 모델의 예측 통계가 초기화되었습니다.")
        else:
            logger.warning(f"모델 {model}의 예측 통계를 초기화하려 했지만, 해당 모델이 존재하지 않습니다.")

    def get_prediction_stats(self, model: str):
        stats = self.prediction_stats.get(model, {'total': 0, 'correct': 0})
        total = stats['total']
        correct = stats['correct']
        accuracy = (correct / total * 100) if total > 0 else 0
        return total, correct, accuracy

    def update_balances(self):
        self.current_balance = self.upbit_client.get_balance("KRW")
        self.current_btc_balance = self.upbit_client.get_balance("BTC")
        self.save_data()

    def record_trade(self, trade_info: Dict[str, Any]):
        self.total_trades += 1
        self.total_profit += trade_info.get('profit', 0)
        self.current_balance = trade_info.get('current_balance', self.current_balance)
        self.current_btc_balance = trade_info.get('current_btc_balance', self.current_btc_balance)
        self.save_data()

    def record_prediction(self, predictions: Dict[str, float], weighted_prediction: float, current_price: float):
        self.prediction_history.append({
            'timestamp': time.time(),
            'predictions': predictions,
            'weighted_prediction': weighted_prediction,
            'current_price': current_price
        })
        self.save_data()

    def calculate_returns(self):
        current_total_value = self.calculate_total_value()
        return ((current_total_value - self.initial_total_value) / self.initial_total_value) * 100

    def get_recent_predictions(self, count: int = 10) -> List[Dict[str, Any]]:
        return list(self.prediction_history)[-count:]

    def get_recent_weights(self, count: int = 3) -> List[Dict[str, float]]:
        return list(self.weight_history)[-count:]

    def get_performance_summary(self) -> str:
        current_total_value = self.calculate_total_value()
        current_btc_price = self.upbit_client.get_current_price("KRW-BTC")

        total_return = ((current_total_value - self.initial_total_value) / self.initial_total_value) * 100
        btc_return = ((current_btc_price - self.initial_btc_price) / self.initial_btc_price) * 100

        summary = "\n📊 트레이딩 성능 요약 📊\n"
        summary += "=" * 30 + "\n\n"

        summary += "1. 수익률\n"
        summary += f"총 수익률: {total_return:.2f}%\n"
        summary += f"BTC 등락률 (Buy & Hold): {btc_return:.2f}%\n\n"

        summary += "2. 모델별 성능\n"
        for model, weight in self.model_weights.items():
            total, correct, accuracy = self.get_prediction_stats(model)

            recent_accuracy = ''
            for pred in list(self.prediction_history)[-30:]:
                if model in pred['predictions']:
                    predicted_price = pred['predictions'][model]
                    actual_price = pred['current_price']
                    next_price = self.prediction_history[self.prediction_history.index(pred) + 1][
                        'current_price'] if self.prediction_history.index(pred) < len(
                        self.prediction_history) - 1 else current_btc_price
                    is_correct = (predicted_price > actual_price and next_price > actual_price) or (
                                predicted_price < actual_price and next_price < actual_price)
                    recent_accuracy += 'O' if is_correct else 'X'

            summary += f"{model.upper()}: 정확도 {accuracy:.2f}%, 가중치 {weight:.4f}, 적중/전체 {correct}/{total}\n"
            summary += f"최근 30개 예측: {recent_accuracy}\n\n"

        summary += "3. 자산 현황\n"
        summary += f"초기 자산: {self.initial_total_value:,.0f} KRW\n"
        summary += f"현재 자산: {current_total_value:,.0f} KRW\n"
        summary += f"현재 KRW: {self.current_balance:,.0f} KRW\n"
        summary += f"현재 BTC: {self.current_btc_balance:.8f} BTC\n\n"

        summary += "4. 거래 통계\n"
        total_predictions = sum(stats['total'] for stats in self.prediction_stats.values())
        total_correct = sum(stats['correct'] for stats in self.prediction_stats.values())
        prediction_accuracy = (total_correct / total_predictions * 100) if total_predictions > 0 else 0

        summary += f"총 거래 횟수: {self.total_trades}\n"
        summary += f"총 예측 횟수: {total_predictions}\n"
        summary += f"적중한 예측: {total_correct}\n"
        summary += f"예측 적중률: {prediction_accuracy:.2f}%\n"
        summary += f"총 수익: {self.total_profit:,.0f} KRW\n\n"

        summary += "5. 최근 예측 결과\n"
        for pred in reversed(list(self.prediction_history)[-5:]):
            pred_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(pred['timestamp']))
            direction = "상승 예측" if pred['weighted_prediction'] > pred['current_price'] else "하락 예측"
            summary += f"{pred_time}: 예측 {pred['weighted_prediction']:.0f}, 실제 {pred['current_price']:.0f}, {direction}\n"

        summary += f"\n마지막 업데이트: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"

        return summary

    def save_data(self):
        data = {
            'prediction_stats': self.prediction_stats,
            'model_weights': self.model_weights,
            'prediction_history': list(self.prediction_history),
            'weight_history': list(self.weight_history),
            'initial_total_value': self.initial_total_value,
            'current_balance': self.current_balance,
            'current_btc_balance': self.current_btc_balance,
            'total_trades': self.total_trades,
            'total_profit': self.total_profit
        }
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def update_weights(self, weights: Dict[str, float]):
        self.model_weights = weights
        self.weight_history.append(weights)
        self.save_data()

    def load_data(self):
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            self.prediction_stats = data.get('prediction_stats', {})
            self.model_weights = data.get('model_weights', {})
            self.prediction_history = deque(data.get('prediction_history', []), maxlen=1000)
            self.weight_history = deque(data.get('weight_history', []), maxlen=100)
            self.initial_total_value = data.get('initial_total_value', self.initial_total_value)
            self.current_balance = data.get('current_balance', self.current_balance)
            self.current_btc_balance = data.get('current_btc_balance', self.current_btc_balance)
            self.total_trades = data.get('total_trades', 0)
            self.total_profit = data.get('total_profit', 0)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"성능 데이터 파일을 불러오는 데 실패했습니다. 새로운 데이터를 시작합니다.")

    def log_model_performance(self):
        logger.info("현재 모델별 성능 요약:")
        for model, stats in self.prediction_stats.items():
            total = stats['total']
            correct = stats['correct']
            accuracy = (correct / total * 100) if total > 0 else 0
            logger.info(f"{model.upper()} 모델: 정확도 {accuracy:.2f}%, 적중/전체: {correct}/{total}, "
                        f"현재 가중치: {self.model_weights.get(model, 0):.4f}")