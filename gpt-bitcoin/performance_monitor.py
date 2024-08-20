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

        # 초기 잔액 및 가격 설정
        self.initial_balance = self.upbit_client.get_balance("KRW")
        self.initial_btc_balance = self.upbit_client.get_balance("BTC")
        self.initial_btc_price = self.upbit_client.get_current_price("KRW-BTC")

        # 현재 상태 변수들
        self.current_balance = self.initial_balance
        self.current_btc_balance = self.initial_btc_balance

        # 거래 관련 변수들
        self.total_trades = 0
        self.total_buy_trades = 0
        self.total_sell_trades = 0
        self.total_hold_trades = 0
        self.total_successful_trades = 0
        self.total_successful_buys = 0
        self.total_successful_sells = 0
        self.total_successful_holds = 0
        self.buy_hold_trades = 0
        self.successful_buy_hold_trades = 0
        self.sell_trades = 0
        self.successful_sell_trades = 0

        # 최근 거래 정보
        self.last_decision = 'hold'
        self.last_trade_price = 0
        self.last_trade_percentage = 0
        self.last_trade_success = False
        self.last_trade_profit = 0

        # 성과 관련 변수들
        self.total_profit = 0
        self.strategy_performance = 0
        self.hodl_performance = 0
        self.buy_hold_success_rate = 0.0
        self.sell_success_rate = 0.0

        # 모델 성능 관련 변수들
        self.total_predictions = 0

        self.model_weights = {
            'gpt': 0.2, 'ml': 0.2, 'xgboost': 0.2, 'rl': 0.2, 'lstm': 0.2,
            'arima': 0.2, 'prophet': 0.2, 'transformer': 0.2
        }

        # LSTM 관련 변수들
        self.lstm_accuracy_history = []
        self.lstm_loss_history = []
        self.lstm_mse_history = []
        self.lstm_mae_history = []

        # 기타 메트릭스
        self.mse = 0
        self.mae = 0

        # 거래 이력
        self.trades = []
        self.weights = {}  # 이 줄을 추가

        # 초기 총 자산 가치 계산
        if self.initial_balance is None or self.initial_balance < 1000:
            logger.warning("초기 KRW 잔액이 너무 적거나 가져오는 데 실패했습니다. 기본값 500,000 KRW를 사용합니다.")
            self.initial_balance = 500_000

        if self.initial_btc_price is None or self.initial_btc_price == 0:
            logger.warning("초기 BTC 가격을 가져오는 데 실패했습니다. 현재 가격을 사용합니다.")
            self.initial_btc_price = self.upbit_client.get_current_price("KRW-BTC")

        self.initial_total_value = self.initial_balance + (self.initial_btc_balance * self.initial_btc_price)
        logger.info(f"초기 총 자산 가치: {self.initial_total_value} KRW")

        # 데이터 프레임 초기화
        self.data = pd.DataFrame()

        # 파일에서 기존 데이터 로드 (있는 경우)
        try:
            self.data = pd.read_csv(file_path)
            logger.info(f"Loaded existing data from {file_path}")
        except FileNotFoundError:
            logger.info(f"No existing data file found at {file_path}. Starting with empty DataFrame.")

        self.accuracy_history = {model: [] for model in
                                 ['gpt', 'ml', 'xgboost', 'rl', 'lstm', 'arima', 'prophet', 'transformer']}
        self.prediction_history = []

        self.correct_predictions = {model: 0 for model in
                                    ['gpt', 'ml', 'xgboost', 'rl', 'lstm', 'arima', 'prophet', 'transformer']}
        self.total_predictions = {model: 0 for model in
                                  ['gpt', 'ml', 'xgboost', 'rl', 'lstm', 'arima', 'prophet', 'transformer']}

        self.last_known_btc_price = None

        # 기존 초기화 코드
        self.weight_adjustment_rate = 0.05  # 5%의 가중치 조정률
        self.prediction_stats_file = 'prediction_stats.json'
        self.model_accuracies = {
            model: {'correct': 0, 'total': 0, 'accuracy': 0.0}
            for model in ['gpt', 'ml', 'xgboost', 'rl', 'lstm', 'arima', 'prophet', 'transformer']
        }
        self.load_prediction_stats()

    def load_prediction_stats(self):
        try:
            with open(self.prediction_stats_file, 'r') as f:
                stats = json.load(f)
            for model, data in stats.get('accuracies', {}).items():
                if isinstance(data, dict):
                    self.model_accuracies[model] = data
                else:
                    self.model_accuracies[model] = {'correct': 0, 'total': 1, 'accuracy': float(data)}
            logger.info("예측 통계를 성공적으로 불러왔습니다.")
            logger.info(f"Loaded model accuracies: {self.model_accuracies}")
        except FileNotFoundError:
            logger.info("예측 통계 파일이 없습니다. 새로운 통계를 시작합니다.")
        except json.JSONDecodeError:
            logger.error("예측 통계 파일이 손상되었습니다. 새로운 통계를 시작합니다.")


    def save_prediction_stats(self):
        with open(self.prediction_stats_file, 'w') as f:
            json.dump({'accuracies': self.model_accuracies}, f, indent=2)
        logger.info("예측 통계를 성공적으로 저장했습니다.")
        logger.info(f"Saved model accuracies: {self.model_accuracies}")

    def adjust_weight(self, model: str, success: bool):
        if model not in self.weights:
            return

        total_weight = sum(self.weights.values())
        current_weight = self.weights[model]

        if success:
            # 성공 시 가중치 증가
            adjustment = current_weight * self.weight_adjustment_rate
            self.weights[model] += adjustment
        else:
            # 실패 시 가중치 감소
            adjustment = current_weight * self.weight_adjustment_rate
            self.weights[model] = max(0.01, self.weights[model] - adjustment)  # 최소 1%의 가중치 보장

        # 가중치 정규화
        new_total = sum(self.weights.values())
        for model in self.weights:
            self.weights[model] /= new_total

    def get_prediction_count(self, model: str) -> int:
        if model not in self.model_accuracies:
            logger.warning(f"No prediction count data for model: {model}")
            return 0
        data = self.model_accuracies[model]
        if isinstance(data, dict) and 'total' in data:
            return data['total']
        else:
            logger.warning(f"Unexpected data structure for model {model}: {data}")
            return 0

    def reset_prediction_count(self, model):
        if model in self.model_accuracies:
            self.model_accuracies[model]['total'] = 0
            self.model_accuracies[model]['correct'] = 0
            self.model_accuracies[model]['accuracy'] = 0.0
        logger.info(f"{model} 모델의 예측 횟수와 정확도가 초기화되었습니다.")

    def update_prediction_accuracy(self, model: str, is_correct: bool):
        if model not in self.model_accuracies or not isinstance(self.model_accuracies[model], dict):
            self.model_accuracies[model] = {'correct': 0, 'total': 0, 'accuracy': 0.0}

        self.model_accuracies[model]['total'] += 1
        if is_correct:
            self.model_accuracies[model]['correct'] += 1

        total = self.model_accuracies[model]['total']
        correct = self.model_accuracies[model]['correct']
        self.model_accuracies[model]['accuracy'] = (correct / total) if total > 0 else 0.0

        logger.info(
            f"Updated accuracy for {model}: Total={total}, Correct={correct}, Is_correct={is_correct}, New Accuracy={self.model_accuracies[model]['accuracy']:.4f}")

        self.save_prediction_stats()

    def update_weights(self, weights):
        self.weights = weights

    def get_all_prediction_counts(self):
        return {model: self.get_prediction_count(model) for model in self.model_accuracies}

    def get_current_btc_price(self):
        price = self.upbit_client.get_current_price("KRW-BTC")
        if price is not None:
            self.last_known_btc_price = price
        return price

    def update_model_accuracy(self, model: str, is_correct: bool):
        if model not in self.model_accuracies:
            self.model_accuracies[model] = {'correct': 0, 'total': 0}

        self.model_accuracies[model]['total'] += 1
        if is_correct:
            self.model_accuracies[model]['correct'] += 1

        total = self.model_accuracies[model]['total']
        correct = self.model_accuracies[model]['correct']
        self.model_accuracies[model]['accuracy'] = (correct / total) * 100 if total > 0 else 0

    def get_model_accuracy(self, model: str) -> float:
        if model not in self.model_accuracies:
            logger.warning(f"No accuracy data for model: {model}")
            return 0.0
        data = self.model_accuracies[model]
        if isinstance(data, dict) and 'accuracy' in data:
            return data['accuracy']
        elif isinstance(data, (int, float)):
            return float(data)
        else:
            logger.warning(f"Unexpected data structure for model {model}: {data}")
            return 0.0

    def get_all_model_accuracies(self) -> Dict[str, float]:
        return {model: self.get_model_accuracy(model) for model in self.model_accuracies}

    def log_prediction_stats(self):
        logger.info("현재 모델별 예측 횟수 및 정확도:")
        for model, data in self.model_accuracies.items():
            logger.info(f"  {model}: {data['total']}회 예측, 정확도 {data['accuracy']:.2f}%, 정답 수: {data['correct']}")

    def reset_accuracy_data(self):
        for model in self.model_accuracies:
            self.model_accuracies[model] = {'correct': 0, 'total': 0}
        logger.info("All model accuracy data has been reset.")

    def log_accuracy_summary(self):
        logger.info("Current model accuracy summary:")
        for model, accuracy in self.get_all_model_accuracies().items():
            logger.info(f"{model}: {accuracy:.2%}")

    def record(self, decision: dict, current_price: float, balance: float, btc_amount: float,
               params: dict, regime: str, anomalies: bool, ml_accuracy: float, ml_loss: float):
        new_record = pd.DataFrame([{
            'timestamp': pd.Timestamp.now(),
            'decision': decision.get('decision', 'unknown'),
            'percentage': decision.get('percentage', 0),
            'target_price': decision.get('target_price', None),
            'current_price': current_price,
            'balance': balance,
            'btc_amount': btc_amount,
            'params': str(params),
            'regime': regime,
            'anomalies': anomalies,
            'dynamic_target': decision.get('dynamic_target', 0),
            'weighted_decision': decision.get('weighted_decision', 0),
            'ml_accuracy': ml_accuracy,
            'ml_loss': ml_loss,
            'success': self._is_trade_successful(decision, current_price)
        }])

        new_record_cleaned = new_record.dropna(axis=1, how='all')
        self.data = pd.concat([self.data, new_record_cleaned], ignore_index=True)
        self.total_trades += 1

        # 전체 성과 업데이트
        if new_record_cleaned['decision'].iloc[0] == 'buy':
            self.total_buy_trades += 1
            if new_record_cleaned['success'].iloc[0]:
                self.total_successful_buys += 1
        elif new_record_cleaned['decision'].iloc[0] == 'sell':
            self.total_sell_trades += 1
            if new_record_cleaned['success'].iloc[0]:
                self.total_successful_sells += 1
        else:  # hold
            self.total_hold_trades += 1
            if new_record_cleaned['success'].iloc[0]:
                self.total_successful_holds += 1

        if new_record_cleaned['success'].iloc[0]:
            self.total_successful_trades += 1

        self.save_to_file()

    def get_recent_prediction_accuracy(self):
        return {model: sum(history[-50:]) / len(history[-50:]) if history else 0
                for model, history in self.accuracy_history.items()}

    def _is_trade_successful(self, decision, current_price):
        if self.data.empty or len(self.data) < 2:
            return False
        prev_price = self.data['current_price'].iloc[-1]
        if decision['decision'] == 'buy':
            return current_price > prev_price
        elif decision['decision'] == 'sell':
            return current_price < prev_price
        else:  # hold
            return abs(current_price - prev_price) / prev_price < 0.01

    def get_current_balances(self):
        krw_balance = self.upbit_client.get_balance("KRW")
        btc_balance = self.upbit_client.get_balance("BTC")
        btc_price = self.upbit_client.get_current_price("KRW-BTC")
        total_balance = krw_balance + (btc_balance * btc_price)
        return krw_balance, btc_balance, total_balance

    def get_performance_summary(self, weights: Dict[str, float]) -> str:
        current_btc_price = self.get_current_btc_price()
        if current_btc_price is None:
            current_btc_price = self.last_known_btc_price or 0

        total_asset_value = self.current_balance + (self.current_btc_balance * current_btc_price)

        strategy_return, hodl_return = self.calculate_returns()

        summary = "\n📊 트레이딩 성능 요약 📊\n"
        summary += "=" * 50 + "\n\n"

        # 1. 수익률 비교
        summary += "1. 수익률 비교\n"
        returns_data = [
            ["전략 진행 수익률", f"{strategy_return:.2f}%"],
            ["HODL 수익률", f"{hodl_return:.2f}%"]
        ]
        summary += tabulate(returns_data, headers=["지표", "값"], tablefmt="grid") + "\n\n"

        # 2. 모델별 성능
        summary += "2. 모델별 성능\n"
        model_data = []
        for model, weight in weights.items():
            try:
                accuracy = self.get_model_accuracy(model) * 100
                model_data.append([model.upper(), f"{accuracy:.2f}%", f"{weight * 100:.2f}%"])
            except Exception as e:
                logger.error(f"Error getting accuracy for model {model}: {e}")
                model_data.append([model.upper(), "N/A", f"{weight * 100:.2f}%"])
        summary += tabulate(model_data, headers=["모델", "정확도", "가중치"], tablefmt="grid") + "\n\n"

        # 3. 자산 현황
        summary += "3. 자산 현황\n"
        asset_data = [
            ["시작 금액", f"{self.initial_balance:,.0f} KRW"],
            ["현재 KRW", f"{self.current_balance:,.0f} KRW"],
            ["현재 BTC", f"{self.current_btc_balance:.8f} BTC"],
            ["현재 총 자산 가치", f"{total_asset_value:,.0f} KRW"]
        ]
        summary += tabulate(asset_data, headers=["항목", "값"], tablefmt="grid") + "\n\n"

        # 4. 거래 통계
        summary += "4. 거래 통계\n"
        trade_data = [
            ["총 거래 횟수", self.total_trades],
            ["성공한 거래", self.total_successful_trades],
            ["매수(+홀드) 성공률", f"{self.buy_hold_success_rate:.2f}%"],
            ["매도 성공률", f"{self.sell_success_rate:.2f}%"]
        ]
        summary += tabulate(trade_data, headers=["지표", "값"], tablefmt="grid") + "\n\n"

        # 5. 최근 거래 정보
        summary += "5. 최근 거래 정보\n"
        recent_trade_data = [
            ["결정", self.last_decision],
            ["거래가", f"{self.last_trade_price:,.0f} KRW"],
            ["거래 비율", f"{self.last_trade_percentage:.2f}%"],
            ["성공 여부", '성공' if self.last_trade_success else '실패'],
            ["수익", f"{self.last_trade_profit:,.0f} KRW"]
        ]
        summary += tabulate(recent_trade_data, headers=["항목", "값"], tablefmt="grid") + "\n\n"

        # 전체 성능 요약
        overall_performance = "양호" if strategy_return > hodl_return else "개선 필요"
        summary += f"📌 전체 성능 평가: {overall_performance}\n"
        if strategy_return > hodl_return:
            summary += "   현재 전략이 HODL 전략보다 좋은 성과를 보이고 있습니다.\n"
        else:
            summary += "   현재 전략의 성과가 HODL 전략에 미치지 못하고 있습니다. 전략 개선이 필요할 수 있습니다.\n"

        summary += f"\n마지막 업데이트: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"

        return summary

    def get_accuracy(self, model):
        total = self.total_predictions.get(model, 0)
        if total > 0:
            return self.correct_predictions.get(model, 0) / total
        return 0

    def load_model_weights(self):
        try:
            with open('model_weights.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'gpt': 0.2, 'ml': 0.2, 'xgboost': 0.2, 'rl': 0.2, 'lstm': 0.2,
                'arima': 0.2, 'prophet': 0.2, 'transformer': 0.2
            }

    def save_to_file(self):
        new_record = {
            'timestamp': pd.Timestamp.now(),
            'strategy_performance': self.strategy_performance,
            'hodl_performance': self.hodl_performance,
            'current_balance': self.current_balance,
            'total_trades': self.total_trades,
            'trades': json.dumps(self.trades),
            'total_profit': self.total_profit,
            'buy_hold_success_rate': self.buy_hold_success_rate,
            'sell_success_rate': self.sell_success_rate,
            'last_decision': self.last_decision if isinstance(self.last_decision, str) else str(self.last_decision),
            'last_trade_price': self.last_trade_price,
            'last_trade_percentage': self.last_trade_percentage,
            'last_trade_success': self.last_result.get('success', False) if isinstance(self.last_result,
                                                                                       dict) else False,
            'last_trade_profit': self.last_result.get('profit', 0) if isinstance(self.last_result, dict) else 0
        }

        # NaN 값을 None으로 대체
        new_record = {k: (v if pd.notna(v) else None) for k, v in new_record.items()}

        new_df = pd.DataFrame([new_record])

        if self.data.empty:
            self.data = new_df
        else:
            # 기존 데이터와 새 데이터의 컬럼을 일치시킵니다
            all_columns = set(self.data.columns) | set(new_df.columns)
            for col in all_columns:
                if col not in self.data.columns:
                    self.data[col] = None
                if col not in new_df.columns:
                    new_df[col] = None

            # 데이터 타입을 일치시킵니다
            for col in all_columns:
                if self.data[col].dtype != new_df[col].dtype:
                    # 문자열(object) 타입으로 통일
                    self.data[col] = self.data[col].astype(str)
                    new_df[col] = new_df[col].astype(str)

            # 데이터 연결
            self.data = pd.concat([self.data, new_df], ignore_index=True)

        # 데이터 저장
        self.data.to_csv(self.file_path, index=False)

        # 로깅 추가
        logger.info(f"Data saved to {self.file_path}. Total records: {len(self.data)}")

    def set_file_path(self, file_path):
        self.file_path = file_path

    def get_file_path(self):
        return self.file_path

    def update_lstm_metrics(self, accuracy, loss=None, mse=None, mae=None):
        self.lstm_accuracy_history.append(accuracy)
        if loss is not None:
            self.lstm_loss_history.append(loss)
        if mse is not None:
            self.lstm_mse_history.append(mse)
            self.mse = mse  # 현재 MSE 값 업데이트
        if mae is not None:
            self.lstm_mae_history.append(mae)
            self.mae = mae  # 현재 MAE 값 업데이트

    def update_arima_metrics(self, accuracy, loss=None):
        """ARIMA 모델의 정확도와 손실을 업데이트하는 메서드"""
        self.arima_accuracy = accuracy
        if loss is not None:
            self.arima_loss = loss

    def update_ml_metrics(self, accuracy, loss):
        self.ml_accuracy = accuracy
        self.ml_loss = loss

    def update_xgboost_metrics(self, accuracy, loss=None):
        self.xgboost_accuracy = accuracy
        if loss is not None:
            self.xgboost_loss = loss

    def update(self, strategy_performance, hodl_performance, model_accuracies, model_weights,
               current_balance, decision, trade_price, trade_percentage):
        self.strategy_performance = strategy_performance
        self.hodl_performance = hodl_performance
        self.model_accuracies = model_accuracies
        self.model_weights = model_weights
        self.current_balance = current_balance
        self.last_decision = decision
        self.last_trade_price = trade_price
        self.last_trade_percentage = trade_percentage
        self.current_btc_balance = self.upbit_client.get_balance("BTC")

    def update_trade_result(self, success, profit):
        self.last_result = {'success': success, 'profit': profit}
        self.total_trades += 1
        self.total_profit += profit
        if success:
            self.total_successful_trades += 1
        self.save_to_file()

    def update_success_rates(self, buy_hold_success_rate, sell_success_rate):
        self.buy_hold_success_rate = buy_hold_success_rate
        self.sell_success_rate = sell_success_rate
        if self.buy_hold_trades > 0:
            self.buy_hold_success_rate = (self.successful_buy_hold_trades / self.buy_hold_trades) * 100
        else:
            self.buy_hold_success_rate = 0.0

        if self.sell_trades > 0:
            self.sell_success_rate = (self.successful_sell_trades / self.sell_trades) * 100
        else:
            self.sell_success_rate = 0.0

    def record_trade(self, trade_info):
        self.trades.append(trade_info)
        self.total_trades += 1

        if trade_info['decision'] in ['buy', 'hold']:
            self.buy_hold_trades += 1
            if trade_info.get('success', False):
                self.successful_buy_hold_trades += 1
        elif trade_info['decision'] == 'sell':
            self.sell_trades += 1
            if trade_info.get('success', False):
                self.successful_sell_trades += 1

        if trade_info.get('success', False):
            self.total_successful_trades += 1
            self.total_profit += trade_info.get('profit', 0)

        # 성공률 계산
        self.buy_hold_success_rate = (
                    self.successful_buy_hold_trades / self.buy_hold_trades * 100) if self.buy_hold_trades > 0 else 0
        self.sell_success_rate = (self.successful_sell_trades / self.sell_trades * 100) if self.sell_trades > 0 else 0

    def calculate_returns(self):
        initial_value = self.initial_balance
        current_value = self.current_balance + (self.current_btc_balance * self.get_current_btc_price())

        strategy_return = ((current_value - initial_value) / initial_value) * 100

        initial_btc_price = self.initial_btc_price or self.get_current_btc_price()
        current_btc_price = self.get_current_btc_price()
        hodl_return = ((current_btc_price - initial_btc_price) / initial_btc_price) * 100

        return strategy_return, hodl_return

    def update_trade_success_rates(self, decision, success):
        if decision in ['buy', 'hold']:
            self.buy_hold_trades += 1
            if success:
                self.successful_buy_hold_trades += 1
        elif decision == 'sell':
            self.sell_trades += 1
            if success:
                self.successful_sell_trades += 1

        self.buy_hold_success_rate = (
                                                 self.successful_buy_hold_trades / self.buy_hold_trades) * 100 if self.buy_hold_trades > 0 else 0
        self.sell_success_rate = (self.successful_sell_trades / self.sell_trades) * 100 if self.sell_trades > 0 else 0

    def update_last_trade_result(self, success, profit):
        self.last_trade_success = success
        self.last_trade_profit = profit

