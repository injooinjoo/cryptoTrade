import json
import logging
import os
import shutil
import threading
import time
import warnings
from collections import deque
from typing import Dict, Any

import numpy as np
import pandas as pd
import schedule
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from statsmodels.tools.sm_exceptions import ValueWarning
from xgboost import XGBClassifier

from api_client import UpbitClient, OpenAIClient, PositionManager
from auto_adjustment import AutoAdjustment, AnomalyDetector, MarketRegimeDetector, DynamicTradingFrequencyAdjuster
from backtesting_and_ml import MLPredictor, RLAgent, run_backtest, LSTMPredictor, XGBoostPredictor, ARIMAPredictor, \
    ProphetPredictor, TransformerPredictor
from config import load_config, setup_logging
from data_manager import DataManager
from discord_notifier import send_discord_message
from performance_monitor import PerformanceMonitor
from trading_logic import analyze_data_with_gpt4, execute_trade, data_manager

warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")

warnings.filterwarnings("ignore", category=ValueWarning)

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)
config = load_config()
pd.set_option('future.no_silent_downcasting', True)

def update_data():
    data_manager.update_data()

class PositionMonitor(threading.Thread):
    def __init__(self, upbit_client, position_manager, stop_loss_percentage: float, take_profit_percentage: float):
        threading.Thread.__init__(self)
        self.upbit_client = upbit_client
        self.position_manager = position_manager
        self.stop_loss_percentage = stop_loss_percentage
        self.take_profit_percentage = take_profit_percentage
        self.running = True
        self.active_orders = {}
        self.highest_price = 0
        self.trailing_stop_percentage = 0.02

    def run(self):
        while self.running:
            try:
                position = self.position_manager.get_position()
                btc_balance = position['btc_balance']
                current_price = self.upbit_client.get_current_price("KRW-BTC")

                if btc_balance > 0:
                    avg_buy_price = self.upbit_client.get_avg_buy_price("BTC")

                    for order_id, order in list(self.active_orders.items()):
                        if current_price <= order['stop_loss']:
                            self._execute_sell(order_id, current_price, btc_balance, "Stop Loss")
                        elif current_price >= order['take_profit']:
                            self._execute_sell(order_id, current_price, btc_balance, "Take Profit")

                    profit_percentage = (current_price - avg_buy_price) / avg_buy_price * 100

                    if profit_percentage <= -self.stop_loss_percentage:
                        self._execute_sell(None, current_price, btc_balance, "Dynamic Stop Loss")
                    elif profit_percentage >= self.take_profit_percentage:
                        self._execute_sell(None, current_price, btc_balance, "Dynamic Take Profit")

                    self.highest_price = max(self.highest_price, current_price)
                    if current_price <= self.highest_price * (1 - self.trailing_stop_percentage):
                        self._execute_sell(None, current_price, btc_balance, "Trailing Stop")

                time.sleep(10)
            except Exception as e:
                logger.error(f"Error in position monitoring: {e}")
                time.sleep(60)

    def _execute_sell(self, order_id, current_price, amount, reason):
        try:
            logger.info(
                f"Executing sell for {'order ' + order_id if order_id else 'dynamic trigger'} at price {current_price} due to {reason}")
            result = self.upbit_client.sell_market_order("KRW-BTC", amount)
            if result:
                logger.info(f"Sell order executed successfully: {result}")
                if order_id:
                    self.remove_order(order_id)
                self.position_manager.update_position()
                send_discord_message(f"🚨 {reason} 실행: {amount:.8f} BTC sold at ₩{current_price:,}")
            else:
                logger.warning(
                    f"Sell order execution failed for {'order ' + order_id if order_id else 'dynamic trigger'}")
        except Exception as e:
            logger.error(f"Error executing sell for {'order ' + order_id if order_id else 'dynamic trigger'}: {e}")

    def remove_order(self, order_id: str):
        if order_id in self.active_orders:
            del self.active_orders[order_id]
            logger.info(f"Removed order: {order_id}")

    def stop(self):
        self.running = False


class TradingLoop:
    def __init__(self, upbit_client, openai_client, data_manager, ml_predictor, rl_agent,
                 auto_adjuster, anomaly_detector, regime_detector, dynamic_adjuster,
                 performance_monitor, position_monitor):
        self.upbit_client = upbit_client
        self.openai_client = openai_client
        self.data_manager = data_manager
        self.ml_predictor = ml_predictor
        self.rl_agent = rl_agent
        self.auto_adjuster = auto_adjuster
        self.anomaly_detector = anomaly_detector
        self.regime_detector = regime_detector
        self.dynamic_adjuster = dynamic_adjuster
        self.performance_monitor = performance_monitor
        self.position_monitor = position_monitor
        self.xgboost_predictor = XGBoostPredictor()
        self.lstm_predictor = LSTMPredictor()
        self.counter = 0
        self.last_trade_time = None
        self.trading_interval = 600
        self.last_decision = None
        self.buy_and_hold_performance = None
        self.config = load_config()
        self.initial_balance = None
        self.trading_history = []
        self.trade_history = []
        self.pending_evaluations = deque()
        self.evaluation_delay = 600
        self.arima_predictor = ARIMAPredictor()
        self.prophet_predictor = ProphetPredictor()
        self.transformer_predictor = TransformerPredictor(input_dim=5, hidden_dim=64, output_dim=1, num_layers=2)
        self.weights = self.get_default_weights()
        self.prediction_accuracy = {model: 0.5 for model in self.weights}
        self.weight_adjustment = 0.01
        self.prediction_history = []
        self.strategy_performance = {}
        self.weight_history = []
        self.accuracy_history = {model: [] for model in self.weights.keys()}
        self.report_interval = 50
        self.has_btc = False
        self.prediction_history_file = 'prediction_history.json'
        self.model_update_interval = 100
        self.model_update_threshold = 0.4
        self.performance_monitor = performance_monitor

        # 파일 경로 초기화
        self.model_weights_file = 'model_weights.json'
        self.model_performance_file = 'model_performance.json'

        self.load_model_weights()
        self.load_prediction_history()
        self.performance_evaluation_interval = 50  # 100회 반복마다 성능 평가
        self.min_predictions_for_update = 20  # 최소 50개의 예측 후 모델 업데이트 고려
        self.model_performance = {model: {'predictions': [], 'actual_values': []} for model in ['gpt', 'ml', 'xgboost', 'rl', 'lstm', 'arima', 'prophet', 'transformer']}
        self.initialize_model_performance()

    def run(self, initial_strategy, initial_backtest_results, historical_data):
        global data
        self.data_manager.check_table_structure()
        self.config['trading_parameters'] = initial_strategy
        last_backtest_results = initial_backtest_results
        self.data_manager.fetch_extended_historical_data(days=365)

        initial_price = historical_data['close'].iloc[0]
        final_price = historical_data['close'].iloc[-1]
        self.hodl_performance = (final_price - initial_price) / initial_price

        self.lstm_predictor.train(historical_data)

        X, y = self.data_manager.prepare_data_for_ml(historical_data)
        self.xgboost_predictor.train(X, y)

        while True:
            try:
                current_time = time.time()

                self.evaluate_pending_trades()
                data = self.data_manager.ensure_sufficient_data()
                if len(data) < 2:
                    logger.warning("데이터가 충분하지 않습니다. 다음 주기를 기다립니다.")
                    time.sleep(60)
                    continue

                if self.last_trade_time is None or (current_time - self.last_trade_time) >= self.trading_interval:
                    print('RUNNING NOW')
                    self.cancel_existing_orders()
                    data = self.data_manager.ensure_sufficient_data()
                    data['date'] = pd.to_datetime(data['date'])
                    data.set_index('date', inplace=True)

                    if self.counter % self.model_update_interval == 0:
                        worst_model, worst_accuracy = self.identify_worst_performing_model()
                        if worst_model and worst_accuracy < self.model_update_threshold:
                            logger.info(f"모델 업데이트 시작: {worst_model} (정확도: {worst_accuracy:.2%})")
                            self.update_model(worst_model)

                            # 모델 업데이트 후 성능 재평가
                            updated_accuracy = self.calculate_all_time_accuracy().get(worst_model)
                            if updated_accuracy is not None:
                                logger.info(f"{worst_model} 모델 업데이트 완료. 새 정확도: {updated_accuracy:.2%}")

                                if updated_accuracy > worst_accuracy:
                                    improvement = (updated_accuracy - worst_accuracy) / worst_accuracy * 100
                                    logger.info(f"{worst_model} 모델 성능이 {improvement:.2f}% 향상되었습니다.")
                                else:
                                    logger.warning(f"{worst_model} 모델 성능이 개선되지 않았습니다. 추가적인 분석이 필요할 수 있습니다.")
                            else:
                                logger.warning(f"{worst_model} 모델의 업데이트된 정확도를 계산할 수 없습니다.")
                        else:
                            logger.info("모든 모델의 성능이 충분히 좋거나, 성능 데이터가 없어 업데이트가 필요하지 않습니다.")

                    self.arima_predictor.train(data)
                    self.prophet_predictor.train(data)

                    ml_prediction = self.ml_predictor.predict(data)
                    xgboost_prediction = self.xgboost_predictor.predict(data)
                    rl_action = self.rl_agent.act(self.prepare_state(data))
                    lstm_prediction = self.lstm_predictor.predict(data)
                    arima_prediction = self.arima_predictor.predict()
                    if isinstance(arima_prediction, (pd.Series, np.ndarray)):
                        arima_prediction = arima_prediction[0]

                    prophet_prediction = self.prophet_predictor.predict()
                    transformer_prediction = self.transformer_predictor.predict(data.tail(1))

                    latest_data = self.data_manager.ensure_sufficient_data()
                    latest_data['date'] = pd.to_datetime(latest_data['date'])
                    latest_data.set_index('date', inplace=True)
                    historical_data = pd.concat([historical_data, latest_data])
                    historical_data = historical_data[~historical_data.index.duplicated(keep='last')]
                    historical_data.sort_index(inplace=True)
                    print(len(historical_data))
                    market_analysis = self.analyze_market(data)
                    anomalies, anomaly_scores = self.anomaly_detector.detect_anomalies(data)
                    current_regime = self.regime_detector.detect_regime(data)
                    average_accuracy = self.data_manager.get_average_accuracy()
                    market_volatility = self.calculate_market_volatility(data)
                    self.dynamic_adjuster.adjust_threshold(market_volatility)
                    self.trading_interval = max(300, int(600 * self.dynamic_adjuster.decision_threshold))

                    gpt4_advice = analyze_data_with_gpt4(
                        data=data,
                        openai_client=self.openai_client,
                        params=self.config['trading_parameters'],
                        upbit_client=self.upbit_client,
                        average_accuracy=self.data_manager.get_average_accuracy(),
                        anomalies=anomalies,
                        anomaly_scores=anomaly_scores,
                        market_regime=current_regime,
                        ml_prediction=ml_prediction,
                        xgboost_prediction=xgboost_prediction,
                        rl_action=rl_action,
                        lstm_prediction=lstm_prediction,
                        backtest_results=last_backtest_results,
                        market_analysis=market_analysis,
                        current_balance=self.upbit_client.get_balance("KRW"),
                        current_btc_balance=self.upbit_client.get_balance("BTC"),
                        hodl_performance=self.hodl_performance,
                        current_performance=self.calculate_current_performance(),
                        trading_history=self.get_recent_trading_history()
                    )
                    predictions = {
                        'gpt': 1 if gpt4_advice['decision'] == 'buy' else -1 if gpt4_advice[
                                                                                    'decision'] == 'sell' else 0,
                        'ml': ml_prediction,
                        'xgboost': xgboost_prediction,
                        'rl': rl_action,
                        'lstm': lstm_prediction,
                        'arima': arima_prediction,
                        'prophet': prophet_prediction,
                        'transformer': transformer_prediction
                    }

                    # 각 모델의 예측 및 실제 값 업데이트
                    current_price = self.upbit_client.get_current_price("KRW-BTC")
                    for model_name, prediction in predictions.items():
                        self.update_model_performance(model_name, prediction, current_price)

                    # 주기적으로 모델 성능 평가
                    if self.counter % self.performance_evaluation_interval == 0:
                        self.evaluate_model_performances()

                    self.record_predictions(gpt4_advice['decision'], ml_prediction, xgboost_prediction, rl_action, lstm_prediction,
                    arima_prediction, prophet_prediction, transformer_prediction)

                    self.evaluate_previous_predictions()
                    model_weights = self.calculate_model_weights()
                    self.adjust_weights_based_on_performance()
                    self.save_model_weights()
                    self.generate_prediction_report()

                    self.log_predictions_and_weights()

                    if len(self.prediction_history) >= 2:
                        previous_predictions, previous_price = self.prediction_history[-2]
                        current_price = self.upbit_client.get_current_price("KRW-BTC")
                        price_change = current_price - previous_price

                        for model, prediction in previous_predictions.items():
                            is_correct = (prediction > 0 and price_change > 0) or (
                                        prediction < 0 and price_change < 0) or (
                                                     prediction == 0 and abs(price_change) < previous_price * 0.005)
                            self.performance_monitor.update_prediction_accuracy(model, is_correct)

                    if self.counter % 10 == 0:
                        self.adjust_weights_based_on_performance()

                        # 가중치 기반 최종 결정
                    decision = self.make_weighted_decision(gpt4_advice, ml_prediction, xgboost_prediction, rl_action,
                                                           lstm_prediction, arima_prediction, prophet_prediction,
                                                           transformer_prediction)

                    strategy_performance = self.calculate_strategy_performance()
                    hodl_performance = self.calculate_hodl_performance(historical_data)
                    current_balance = self.upbit_client.get_balance("KRW") + self.upbit_client.get_balance(
                        "BTC") * self.upbit_client.get_current_price("KRW-BTC")
                    self.performance_monitor.update(
                        strategy_performance,
                        hodl_performance,
                        self.prediction_accuracy,
                        self.calculate_model_weights(),
                        current_balance,
                        decision['decision'],
                        decision['target_price'],
                        decision['percentage']
                    )

                    if decision['decision'] == 'hold':
                        self.performance_monitor.update_trade_result(True, 0)
                        self.performance_monitor.update_success_rates()

                    if decision['decision'] in ['buy', 'sell']:
                        result = execute_trade(self.upbit_client, decision, self.config, self.has_btc)

                        if result['success']:
                            self.has_btc = result['has_btc']
                            self.add_pending_evaluation(decision, result)

                            trade_info = {
                                'timestamp': time.time(),
                                'decision': decision['decision'],
                                'price': result['price'],
                                'amount': result['amount']
                            }
                            self.performance_monitor.record_trade(trade_info)

                    self.last_trade_time = current_time
                    all_time_accuracy = self.calculate_all_time_accuracy()
                    accuracy_report = "전체 예측 성공률:\n" + "\n".join(
                        [f"{model}: {acc:.2%}" for model, acc in all_time_accuracy.items()])
                    logger.info(accuracy_report)
                    send_discord_message(accuracy_report)

                    performance_summary = self.performance_monitor.get_performance_summary(self.weights)
                    send_discord_message(performance_summary)

                    self.save_model_weights()
                    self.last_trade_time = current_time
                    self.counter += 1

                # 주기적 업데이트 (예: 1시간마다)
                if self.counter % 360 == 0:  # 6시간마다 (10분 * 36)
                    self.periodic_update(historical_data)

                time.sleep(10)  # 10초마다 체크

            except ValueError as ve:
                logger.error(f"ValueError in trading loop: {ve}")
                logger.info("Attempting to retrain XGBoost model...")
                X, y = self.data_manager.prepare_data_for_ml(data)
                self.xgboost_predictor.train(X, y)
                continue

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                logger.exception("Traceback:")
                time.sleep(60)

    def numpy_to_python(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.numpy_to_python(item) for item in obj]
        else:
            return obj

    def initialize_model_performance(self):
        self.model_performance = {
            model: {
                'predictions': [],
                'actual_values': [],
                'accuracy': 0.0
            } for model in ['gpt', 'ml', 'xgboost', 'rl', 'lstm', 'arima', 'prophet', 'transformer']
        }

    def identify_worst_performing_model(self):
        models_to_consider = {}
        for model_name, data in self.model_performance.items():
            if len(data['predictions']) >= self.min_predictions_for_update:
                models_to_consider[model_name] = data.get('accuracy', 0.0)

        if not models_to_consider:
            return None, None

        worst_model = min(models_to_consider, key=models_to_consider.get)
        worst_accuracy = models_to_consider[worst_model]
        return worst_model, worst_accuracy

    def update_model(self, model_name):
        if model_name == 'xgboost':
            self.update_xgboost_model()
        elif model_name == 'lstm':
            self.update_lstm_model()
        elif model_name == 'ml':
            self.update_ml_model()
        elif model_name == 'rl':
            self.update_rl_model()
        # 다른 모델들에 대한 업데이트 로직 추가

        logger.info(f"{model_name} 모델이 업데이트되었습니다.")

        # 모델 업데이트 후 성능 재평가
        updated_accuracy = self.calculate_model_accuracy(model_name)
        previous_accuracy = self.model_performance[model_name]['accuracy']

        if updated_accuracy > previous_accuracy:
            if previous_accuracy > 0:
                improvement = (updated_accuracy - previous_accuracy) / previous_accuracy * 100
                logger.info(f"{model_name} 모델 성능이 {improvement:.2f}% 향상되었습니다.")
            else:
                logger.info(f"{model_name} 모델 성능이 향상되었습니다. 새 정확도: {updated_accuracy:.2%}")
        else:
            logger.warning(f"{model_name} 모델 성능이 개선되지 않았습니다. 추가적인 분석이 필요할 수 있습니다.")

        # 모델 성능 정보 업데이트
        self.model_performance[model_name]['accuracy'] = updated_accuracy
        self.save_model_weights()  # 업데이트된 성능 정보 저장

    def calculate_model_accuracy(self, model_name):
        if model_name not in self.model_performance:
            logger.warning(f"Model {model_name} not found in performance data.")
            return 0.0

        predictions = self.model_performance[model_name]['predictions']
        actual_values = self.model_performance[model_name]['actual_values']

        if len(predictions) < self.min_predictions_for_update:
            return 0.0

        if model_name in ['gpt', 'ml', 'xgboost', 'rl']:
            # 분류 모델의 경우
            accuracy = accuracy_score([1 if v > 0 else 0 for v in actual_values], [1 if p > 0 else 0 for p in predictions])
        elif model_name in ['lstm', 'arima', 'prophet', 'transformer']:
            # 회귀 모델의 경우
            mse = mean_squared_error(actual_values, predictions)
            accuracy = 1 / (1 + mse)  # 정규화된 정확도
        else:
            logger.warning(f"Unknown model: {model_name}")
            return 0.0

        self.model_performance[model_name]['accuracy'] = accuracy
        return accuracy


    def update_xgboost_model(self):
        data = self.data_manager.ensure_sufficient_data()
        X, y = self.data_manager.prepare_data_for_ml(data)

        param_dist = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 4, 5],
            'min_child_weight': [1, 3, 5]
        }

        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        random_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=10, cv=3, random_state=42)
        random_search.fit(X, y)

        self.xgboost_predictor.model = random_search.best_estimator_

    def update_lstm_model(self):
        data = self.data_manager.ensure_sufficient_data()
        self.lstm_predictor.train(data)

    def update_ml_model(self):
        data = self.data_manager.ensure_sufficient_data()
        self.ml_predictor.train(self.data_manager)

    def update_rl_model(self):
        # RL 모델 업데이트 로직 구현
        # 예: 새로운 에피소드로 학습 또는 파라미터 조정
        pass

    def save_model_weights(self):
        self._safe_save(self.model_weights_file, self.weights)
        self._safe_save(self.model_performance_file, self.model_performance)
        logger.info("Model weights and performance data saved successfully.")

    def _safe_save(self, file_path, data):
        # 임시 파일에 저장
        temp_file = file_path + '.temp'
        with open(temp_file, 'w') as f:
            json.dump(self.numpy_to_python(data), f, indent=2)

        # 기존 파일 백업 (있는 경우)
        if os.path.exists(file_path):
            backup_file = file_path + '.bak'
            shutil.copy2(file_path, backup_file)

        # 임시 파일을 실제 파일로 이동
        os.replace(temp_file, file_path)

    def save_prediction_history(self):
        try:
            # 저장 전 데이터 유효성 검사
            if not isinstance(self.prediction_history, list):
                raise ValueError("prediction_history는 리스트 형태여야 합니다.")

            for item in self.prediction_history:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ValueError("prediction_history의 각 항목은 2개 요소를 가진 튜플이어야 합니다.")

                predictions, price = item
                if not isinstance(predictions, dict) or not isinstance(price, (int, float)):
                    raise ValueError("prediction_history 항목의 형식이 올바르지 않습니다.")

            with open(self.prediction_history_file, 'w') as f:
                json.dump(self.prediction_history, f, default=self.json_default)
            logger.info(f"예측 기록 {len(self.prediction_history)}개가 성공적으로 저장되었습니다.")
        except Exception as e:
            logger.error(f"예측 기록을 저장하는 중 오류 발생: {e}")

    def json_default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)

    def load_prediction_history(self):
        try:
            if os.path.exists(self.prediction_history_file):
                with open(self.prediction_history_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        try:
                            self.prediction_history = json.loads(content)
                            logger.info(f"예측 기록을 성공적으로 로드했습니다. 기록 수: {len(self.prediction_history)}")
                        except json.JSONDecodeError as json_err:
                            logger.error(f"JSON 디코딩 오류: {json_err}")
                            logger.error(f"파일 내용 (처음 100자): {content[:100]}")
                            logger.info("손상된 prediction_history.json 파일을 백업하고 새로운 파일을 생성합니다.")
                            self._backup_and_create_new_file()
                            self.prediction_history = []
                    else:
                        logger.warning("prediction_history.json 파일이 비어 있습니다. 새로운 예측 기록을 시작합니다.")
                        self.prediction_history = []
            else:
                logger.info("prediction_history.json 파일이 없습니다. 새 파일을 생성합니다.")
                self._create_new_file()
                self.prediction_history = []
        except Exception as e:
            logger.error(f"예측 기록을 로드하는 중 예외 발생: {e}")
            logger.exception("상세 오류:")
            self._backup_and_create_new_file()
            self.prediction_history = []

    def _backup_and_create_new_file(self):
        if os.path.exists(self.prediction_history_file):
            backup_file = f"{self.prediction_history_file}.bak"
            os.rename(self.prediction_history_file, backup_file)
            logger.info(f"기존 파일을 {backup_file}으로 백업했습니다.")
        self._create_new_file()

    def _create_new_file(self):
        with open(self.prediction_history_file, 'w') as f:
            json.dump([], f)
        logger.info(f"새로운 {self.prediction_history_file} 파일을 생성했습니다.")

    def load_model_weights(self):
        self.weights = self._safe_load(self.model_weights_file, self.get_default_weights)
        self.model_performance = self._safe_load(self.model_performance_file, self.initialize_model_performance)

    def _safe_load(self, file_path, default_function):
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                if not content.strip():
                    logger.warning(f"File {file_path} is empty. Using default values.")
                    return default_function()
                data = json.loads(content)
            logger.info(f"Data loaded successfully from {file_path}.")
            return data
        except FileNotFoundError:
            logger.warning(f"File {file_path} not found. Using default values.")
            return default_function()
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {file_path}: {e}")
            logger.error(f"Content of {file_path}: {content[:100]}...")  # Log first 100 characters
            backup_file = file_path + '.error'
            shutil.copy2(file_path, backup_file)
            logger.info(f"Backup of problematic file created: {backup_file}")
            return default_function()
    def get_default_weights(self):
        return {model: 1.0 / 8 for model in ['gpt', 'ml', 'xgboost', 'rl', 'lstm', 'arima', 'prophet', 'transformer']}

    def adjust_model_weights(self):
        total_accuracy = sum(self.prediction_accuracy.values())
        if total_accuracy > 0:
            for model in self.weights:
                self.weights[model] = self.prediction_accuracy[model] / total_accuracy
        else:
            for model in self.weights:
                self.weights[model] = 1 / len(self.weights)

        self.save_model_weights()

    def add_pending_evaluation(self, decision, result):
        evaluation_time = time.time() + self.evaluation_delay
        self.pending_evaluations.append({
            'decision': decision,
            'result': result,
            'evaluation_time': evaluation_time,
            'strategy': decision.get('strategy', 'unknown')
        })

    def evaluate_pending_trades(self):
        current_time = time.time()
        while self.pending_evaluations and self.pending_evaluations[0]['evaluation_time'] <= current_time:
            try:
                trade = self.pending_evaluations.popleft()
                success = self.evaluate_trade_success(trade['decision'], trade['result'])
                self.update_strategy_performance(trade['decision'], success)
            except Exception as e:
                logger.error(f"Error evaluating trade: {e}")
                logger.exception("Traceback:")

    def calculate_market_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(window=window).std().iloc[-1]
        return volatility * np.sqrt(252)  # 연간화된 변동성

    def update_strategy_performance(self, decision, success):
        strategy = decision.get('strategy', 'unknown')
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {'success': 0, 'failure': 0}

        if success:
            self.strategy_performance[strategy]['success'] += 1
        else:
            self.strategy_performance[strategy]['failure'] += 1

    def evaluate_previous_predictions(self):
        if len(self.prediction_history) < 2:
            return

        previous_predictions, previous_price = self.prediction_history[-2]
        current_price = self.upbit_client.get_current_price("KRW-BTC")
        price_change_ratio = (current_price - previous_price) / previous_price

        for model, prediction in previous_predictions.items():
            if isinstance(prediction, np.ndarray):
                prediction = prediction.item()

            if prediction == 0:  # 홀드 예측
                is_correct = abs(price_change_ratio) < 0.005  # 0.5% 미만의 변동은 홀드로 간주
            else:
                is_correct = (prediction > 0 and price_change_ratio > 0) or (prediction < 0 and price_change_ratio < 0)

            self.update_model_performance(model, prediction, current_price)
            self.performance_monitor.update_prediction_accuracy(model, is_correct)  # 여기를 수정했습니다

        logger.info(f"가격 변화율: {price_change_ratio:.2%}")
        logger.info(f"예측 결과: {previous_predictions}")
        logger.info(f"현재 예측 정확도: {self.performance_monitor.get_all_model_accuracies()}")

    def update_model_performance(self, model_name, prediction, actual_value):
        if model_name not in self.model_performance:
            self.model_performance[model_name] = {'predictions': [], 'actual_values': [], 'accuracy': 0.0}

        # NumPy 배열을 스칼라로 변환
        if isinstance(prediction, np.ndarray):
            prediction = prediction.item()
        if isinstance(actual_value, np.ndarray):
            actual_value = actual_value.item()

        self.model_performance[model_name]['predictions'].append(float(prediction))
        self.model_performance[model_name]['actual_values'].append(float(actual_value))

        # 저장된 데이터 개수 제한 (예: 최근 1000개만 유지)
        max_history = 1000
        if len(self.model_performance[model_name]['predictions']) > max_history:
            self.model_performance[model_name]['predictions'] = self.model_performance[model_name]['predictions'][
                                                                -max_history:]
            self.model_performance[model_name]['actual_values'] = self.model_performance[model_name]['actual_values'][
                                                                  -max_history:]

        # 정확도 갱신
        self.calculate_model_accuracy(model_name)

    def evaluate_model_performances(self):
        for model_name in self.model_performance.keys():
            accuracy = self.calculate_model_accuracy(model_name)
            logger.info(f"{model_name} 모델 정확도: {accuracy:.4f}")

    def generate_prediction_report(self):
        if len(self.prediction_history) % self.report_interval == 0:
            report = "예측 성공률 및 가중치 보고서:\n"
            for model in self.weights.keys():
                accuracy = sum(self.accuracy_history[model][-self.report_interval:]) / self.report_interval
                avg_weight = sum(w[model] for w in self.weight_history[-self.report_interval:]) / self.report_interval
                report += f"{model.upper()}: 성공률 {accuracy:.2%}, 평균 가중치 {avg_weight:.4f}\n"

            logger.info(report)
            send_discord_message(report)

    def log_predictions_and_weights(self):
        logger.info("모델별 가중치:")
        for model, weight in self.weights.items():
            logger.info(f"{model.upper()}: 가중치 {weight:.4f}")

        logger.info("최근 예측 결과 및 성공 여부:")

        cumulative_predictions = {model: {'correct': 0, 'total': 0} for model in self.weights.keys()}

        recent_predictions = self.prediction_history[-5:] if len(self.prediction_history) >= 5 else self.prediction_history

        for i, (predictions, price) in enumerate(recent_predictions):
            logger.info(f"예측 {i + 1}:")
            for model in self.weights.keys():
                prediction = predictions[model]

                if i < len(recent_predictions) - 1:
                    next_price = recent_predictions[i + 1][1]
                    price_change_ratio = (next_price - price) / price

                    if prediction == 0:  # 홀드 예측
                        success = (self.has_btc and price_change_ratio > 0) or (not self.has_btc and price_change_ratio < 0)
                    else:
                        success = (prediction > 0 and price_change_ratio > 0) or (prediction < 0 and price_change_ratio < 0)

                    cumulative_predictions[model]['total'] += 1
                    if success:
                        cumulative_predictions[model]['correct'] += 1

                    logger.info(
                        f"  {model.upper()}: 예측 {prediction}, 실제 변동률 {price_change_ratio:.2%}, 성공 여부: {'성공' if success else '실패'}")
                else:
                    logger.info(f"  {model.upper()}: 예측 {prediction}, 실제 변동 아직 알 수 없음")

        logger.info("누적 예측 성공률:")
        for model, results in cumulative_predictions.items():
            if results['total'] > 0:
                success_rate = results['correct'] / results['total'] * 100
                logger.info(f"  {model.upper()}: {success_rate:.2f}% ({results['correct']}/{results['total']})")
            else:
                logger.info(f"  {model.upper()}: 아직 충분한 데이터가 없습니다.")

    def analyze_market(self, data: pd.DataFrame) -> Dict[str, Any]:
        current_price = data['close'].iloc[-1]
        sma_20 = data['close'].rolling(window=20).mean().iloc[-1]
        rsi_14 = self.calculate_rsi(data['close'], period=14)

        return {
            'current_price': current_price,
            'sma_20': sma_20,
            'rsi_14': rsi_14,
            'trend': 'Bullish' if current_price > sma_20 else 'Bearish',
            'overbought_oversold': 'Overbought' if rsi_14 > 70 else 'Oversold' if rsi_14 < 30 else 'Neutral'
        }

    def prepare_state(self, data: pd.DataFrame) -> np.ndarray:
        state = np.array([
            data['close'].iloc[-1],
            data['volume'].iloc[-1],
            data['close'].pct_change().iloc[-1],
            data['close'].rolling(window=20).mean().iloc[-1],
            self.calculate_rsi(data['close'], period=14)
        ])
        return state.reshape(1, -1)

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))

    def get_recent_trading_history(self, n=5):
        return self.trading_history[-n:]

    def adjust_weights_based_on_performance(self):
        total_accuracy = sum(model_data.get('accuracy', 0.0) for model_data in self.model_performance.values())
        if total_accuracy > 0:
            for model in self.weights:
                self.weights[model] = self.model_performance[model].get('accuracy', 0.0) / total_accuracy
        else:
            for model in self.weights:
                self.weights[model] = 1 / len(self.weights)

        self.save_model_weights()

    def periodic_update(self, historical_data):
        last_backtest_results = run_backtest(self.data_manager, historical_data)
        ml_accuracy, _, ml_performance = self.ml_predictor.train(self.data_manager)

        X, y = self.data_manager.prepare_data_for_ml(historical_data)
        xgboost_accuracy, _, _ = self.xgboost_predictor.train(X, y)
        lstm_accuracy = self.lstm_predictor.train(historical_data)

        self.arima_predictor.train(historical_data)
        self.prophet_predictor.train(historical_data)
        self.transformer_predictor.train(historical_data)

        logger.info(f"ML 정확도: {ml_accuracy:.2%}")
        logger.info(f"XGBoost 정확도: {xgboost_accuracy:.2%}")
        logger.info(f"LSTM 정확도: {lstm_accuracy:.2%}")
        logger.info(f"ARIMA 정확도: {self.arima_predictor.get_accuracy():.2%}")
        logger.info(f"Prophet 정확도: {self.prophet_predictor.get_accuracy():.2%}")
        logger.info(f"Transformer 정확도: {self.transformer_predictor.get_accuracy():.2%}")

        self.save_model_weights()

    def calculate_current_performance(self):
        try:
            recent_data = self.data_manager.get_recent_trades(days=30)
            if recent_data.empty:
                logger.warning("최근 거래 데이터가 없습니다.")
                return 0

            initial_price = recent_data['close'].iloc[0]
            final_price = recent_data['close'].iloc[-1]
            performance = (final_price - initial_price) / initial_price

            daily_returns = recent_data['close'].pct_change().dropna()
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()

            cumulative_returns = (1 + daily_returns).cumprod()
            max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()

            current_balance = self.upbit_client.get_balance("KRW") + \
                              self.upbit_client.get_balance("BTC") * self.upbit_client.get_current_price("KRW-BTC")

            if self.initial_balance is None:
                self.initial_balance = current_balance

            actual_performance = (current_balance - self.initial_balance) / self.initial_balance if self.initial_balance else 0

            logger.info(f"현재 전략 성능: 시장 수익률 {performance:.2%}, 실제 수익률 {actual_performance:.2%}, "
                        f"샤프 비율 {sharpe_ratio:.2f}, 최대 손실폭 {max_drawdown:.2%}")

            performance_score = 0.3 * performance + 0.3 * actual_performance + 0.2 * sharpe_ratio - 0.2 * max_drawdown

            return performance_score

        except Exception as e:
            logger.error(f"성능 계산 중 오류 발생: {e}")
            return 0

    def cancel_existing_orders(self):
        try:
            orders = self.upbit_client.get_order("KRW-BTC")
            for order in orders:
                self.upbit_client.cancel_order(order['uuid'])
            logger.info("모든 미체결 주문이 취소되었습니다.")
        except Exception as e:
            logger.error(f"미체결 주문 취소 중 오류 발생: {e}")

    def record_predictions(self, gpt4_decision, ml_prediction, xgboost_prediction, rl_action, lstm_prediction,
                           arima_prediction, prophet_prediction, transformer_prediction):
        current_price = self.upbit_client.get_current_price("KRW-BTC")
        predictions = {
            'gpt': 1 if gpt4_decision == 'buy' else -1 if gpt4_decision == 'sell' else 0,
            'ml': ml_prediction,
            'xgboost': xgboost_prediction,
            'rl': 1 if rl_action == 2 else -1 if rl_action == 0 else 0,
            'lstm': 1 if lstm_prediction > current_price else -1 if lstm_prediction < current_price else 0,
            'arima': 1 if arima_prediction > current_price else -1 if arima_prediction < current_price else 0,
            'prophet': 1 if prophet_prediction > current_price else -1 if prophet_prediction < current_price else 0,
            'transformer': 1 if transformer_prediction > current_price else -1 if transformer_prediction < current_price else 0
        }
        self.prediction_history.append((predictions, current_price))
        if len(self.prediction_history) > 100:
            self.prediction_history.pop(0)

        logger.info(f"현재 예측: {predictions}")

    def calculate_prediction_strength(self, predictions):
        strength = sum(abs(pred) for pred in predictions.values())
        return strength / len(predictions)

    def adjust_trade_ratio(self, base_ratio, prediction_strength):
        max_ratio = 1.0  # 최대 거래 비율
        min_ratio = 0.1  # 최소 거래 비율
        return min(max(base_ratio * prediction_strength, min_ratio), max_ratio)

    def make_weighted_decision(self, gpt4_advice, ml_prediction, xgboost_prediction, rl_action, lstm_prediction,
                               arima_prediction, prophet_prediction, transformer_prediction):
        current_price = self.upbit_client.get_current_price("KRW-BTC")

        decisions = {
            'buy': 0,
            'sell': 0,
            'hold': 0
        }

        gpt_decision = 'buy' if gpt4_advice['decision'] == 'buy' else 'sell' if gpt4_advice['decision'] == 'sell' else 'hold'
        decisions[gpt_decision] += self.weights['gpt']

        ml_decision = 'buy' if ml_prediction == 1 else 'sell' if ml_prediction == 0 else 'hold'
        decisions[ml_decision] += self.weights['ml']

        xgboost_decision = 'buy' if xgboost_prediction == 1 else 'sell' if xgboost_prediction == 0 else 'hold'
        decisions[xgboost_decision] += self.weights['xgboost']

        rl_decision = 'buy' if rl_action == 2 else 'sell' if rl_action == 0 else 'hold'
        decisions[rl_decision] += self.weights['rl']

        lstm_decision = 'buy' if lstm_prediction > current_price else 'sell' if lstm_prediction < current_price else 'hold'
        decisions[lstm_decision] += self.weights['lstm']

        arima_decision = 'buy' if arima_prediction > current_price else 'sell' if arima_prediction < current_price else 'hold'
        decisions[arima_decision] += self.weights.get('arima', 0.1)

        prophet_decision = 'buy' if prophet_prediction > current_price else 'sell' if prophet_prediction < current_price else 'hold'
        decisions[prophet_decision] += self.weights.get('prophet', 0.1)

        transformer_decision = 'buy' if transformer_prediction > current_price else 'sell' if transformer_prediction < current_price else 'hold'
        decisions[transformer_decision] += self.weights.get('transformer', 0.1)

        logger.info(f"Weighted votes: {decisions}")

        final_decision = max(decisions, key=decisions.get)

        prediction_strength = self.calculate_prediction_strength(decisions)
        base_ratio = 0.5  # 기본 거래 비율
        adjusted_ratio = self.adjust_trade_ratio(base_ratio, prediction_strength)

        if final_decision == 'buy':
            target_price = gpt4_advice['potential_buy']['target_price'] if gpt4_advice['decision'] == 'hold' else \
            gpt4_advice['target_price']
            percentage = adjusted_ratio * 100
        elif final_decision == 'sell':
            target_price = gpt4_advice['potential_sell']['target_price'] if gpt4_advice['decision'] == 'hold' else \
            gpt4_advice['target_price']
            percentage = adjusted_ratio * 100
        else:
            target_price = None
            percentage = 0

        return {
            'decision': final_decision,
            'percentage': percentage,
            'target_price': target_price,
            'stop_loss': gpt4_advice.get('stop_loss'),
            'take_profit': gpt4_advice.get('take_profit'),
            'reasoning': f"Weighted voting decision ({final_decision}) based on individual model predictions. Prediction strength: {prediction_strength:.2f}, Adjusted trade ratio: {adjusted_ratio:.2f}"
        }

    def calculate_all_time_accuracy(self):
        if not self.prediction_history:
            return {}

        correct_predictions = {model: 0 for model in self.prediction_history[0][0].keys()}
        total_predictions = {model: 0 for model in self.prediction_history[0][0].keys()}

        for i in range(len(self.prediction_history) - 1):
            predictions, price = self.prediction_history[i]
            next_price = self.prediction_history[i + 1][1]

            for model, prediction in predictions.items():
                total_predictions[model] += 1
                if (prediction > 0 and next_price > price) or (prediction < 0 and next_price < price) or (
                        prediction == 0 and next_price == price):
                    correct_predictions[model] += 1

        accuracy = {model: correct_predictions[model] / total_predictions[model] if total_predictions[model] > 0 else 0
                    for model in correct_predictions.keys()}
        return accuracy

    def calculate_strategy_performance(self):
        if self.initial_balance is None:
            self.initial_balance = self.upbit_client.get_balance("KRW") + \
                                   self.upbit_client.get_balance("BTC") * self.upbit_client.get_current_price("KRW-BTC")
        current_balance = self.upbit_client.get_balance("KRW") + \
                          self.upbit_client.get_balance("BTC") * self.upbit_client.get_current_price("KRW-BTC")
        return (current_balance - self.initial_balance) / self.initial_balance * 100

    def calculate_hodl_performance(self, historical_data):
        if len(historical_data) < 2:
            return 0
        initial_price = historical_data['close'].iloc[0]
        current_price = historical_data['close'].iloc[-1]
        return (current_price - initial_price) / initial_price * 100

    def calculate_model_weights(self) -> Dict[str, float]:
        model_accuracies = self.performance_monitor.get_all_model_accuracies()
        logger.debug(f"Current model accuracies: {model_accuracies}")

        total_accuracy = sum(model_accuracies.values())

        if total_accuracy == 0:
            logger.warning("Total accuracy is 0. Using equal weights for all models.")
            return {model: 1.0 / len(model_accuracies) for model in model_accuracies}

        weights = {model: acc / total_accuracy for model, acc in model_accuracies.items()}
        logger.info(f"Calculated model weights: {weights}")
        return weights


    def evaluate_trade_success(self, decision, result):
        current_price = self.upbit_client.get_current_price("KRW-BTC")
        if decision['decision'] == 'buy':
            return current_price > result['price']
        elif decision['decision'] == 'sell':
            return current_price < result['price']
        return False



def main():
    upbit_client = UpbitClient(config['upbit_access_key'], config['upbit_secret_key'])
    openai_client = OpenAIClient(config['openai_api_key'])
    data_manager = DataManager(upbit_client)

    data_manager.add_decision_column()
    data_manager.debug_database()

    ml_predictor = MLPredictor()
    rl_agent = RLAgent(state_size=12, action_size=3)
    auto_adjuster = AutoAdjustment(config.get('initial_params', {}))
    anomaly_detector = AnomalyDetector()
    regime_detector = MarketRegimeDetector()
    dynamic_adjuster = DynamicTradingFrequencyAdjuster()
    performance_monitor = PerformanceMonitor(upbit_client)
    position_manager = PositionManager(upbit_client)

    position_monitor = PositionMonitor(upbit_client, position_manager, stop_loss_percentage=5,
                                       take_profit_percentage=10)
    position_monitor.start()

    schedule.every(10).minutes.do(update_data)

    try:
        historical_data = data_manager.ensure_sufficient_data(1440)

        historical_data['date'] = pd.to_datetime(historical_data['date'])
        historical_data.set_index('date', inplace=True)

        if historical_data.empty:
            logger.error("Failed to retrieve historical data.")
            return

        initial_backtest_results = run_backtest(data_manager, historical_data)
        ml_accuracy, _, _ = ml_predictor.train(data_manager)

        initial_strategy = auto_adjuster.generate_initial_strategy(initial_backtest_results, ml_accuracy)

        trading_loop = TradingLoop(
            upbit_client, openai_client, data_manager, ml_predictor, rl_agent,
            auto_adjuster, anomaly_detector, regime_detector, dynamic_adjuster,
            performance_monitor, position_monitor
        )
        trading_loop.run(initial_strategy, initial_backtest_results, historical_data)
    finally:
        position_monitor.stop()

if __name__ == "__main__":
    main()