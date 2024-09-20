import json
import logging
import os
import sys
import time
from collections import deque
from datetime import timedelta, datetime
from logging.handlers import TimedRotatingFileHandler
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from api_client import UpbitClient, OpenAIClient
from auto_adjustment import AutoAdjustment, AnomalyDetector, MarketRegimeDetector, DynamicTradingFrequencyAdjuster
from backtesting_and_ml import MLPredictor, XGBoostPredictor, LSTMPredictor, RLAgent, ARIMAPredictor, ProphetPredictor, \
    TransformerPredictor, BitcoinForecastPredictor
from config import load_config, setup_logging
from data_manager import DataManager
from discord_notifier import send_discord_message
from performance_monitor import PerformanceMonitor
from trading_logic import TradingDecisionMaker
from notion_logger import NotionLogger


# 로깅 설정 함수 호출
setup_logging()

logger = logging.getLogger(__name__)



class TradingLoop:
    def __init__(self, upbit_client, openai_client, data_manager, ml_predictor, xgboost_predictor,
                 lstm_predictor, rl_agent, auto_adjuster, anomaly_detector, regime_detector,
                 dynamic_adjuster, performance_monitor, config):
        self.upbit_client = upbit_client
        self.openai_client = openai_client
        self.data_manager = data_manager
        self.ml_predictor = ml_predictor
        self.xgboost_predictor = xgboost_predictor
        self.lstm_predictor = lstm_predictor
        self.rl_agent = rl_agent
        self.auto_adjuster = auto_adjuster
        self.anomaly_detector = anomaly_detector
        self.regime_detector = regime_detector
        self.dynamic_adjuster = dynamic_adjuster
        self.performance_monitor = performance_monitor
        self.config = config

        # logger.info("Checking and filling missing data in BTC.DB...")
        # self.data_manager.check_and_fill_missing_data()
        # logger.info("DB check and fill completed.")

        # 초기 자산 가치 설정
        initial_balance = self.upbit_client.get_balance("KRW")
        initial_btc_balance = self.upbit_client.get_balance("BTC")
        initial_btc_price = self.upbit_client.get_current_price("KRW-BTC")
        initial_total_value = initial_balance + (initial_btc_balance * initial_btc_price)

        self.performance_monitor.initial_total_value = initial_total_value
        self.performance_monitor.current_balance = initial_balance
        self.performance_monitor.current_btc_balance = initial_btc_balance
        self.performance_monitor.initial_btc_price = initial_btc_price
        self.performance_monitor.save_data()

        self.arima_predictor = ARIMAPredictor()
        self.prophet_predictor = ProphetPredictor()
        self.transformer_predictor = TransformerPredictor(input_dim=13, seq_length=10)
        self.decision_maker = TradingDecisionMaker(config)
        self.bitcoin_forecast_predictor = self.initialize_bitcoin_forecast_predictor()

        self.counter = 0
        self.last_trade_time = None
        self.trading_interval = 600  # 10 minutes
        self.evaluation_delay = 600
        self.pending_evaluations = deque()
        self.prediction_history = deque(maxlen=100)  # 최근 100개의 예측을 저장
        self.evaluation_delay = 600  # 10분 (초 단위)
        self.weights_file = 'model_weights.json'
        self.model_weights = self.load_model_weights()
        self.weight_history = deque(maxlen=3)  # 최근 3번의 가중치 변화를 저장

        self.previous_weights = self.model_weights.copy()  # previous_weights 초기화
        self.current_predictions = {}
        # 여기에 history_file 속성 추가
        self.history_file = 'prediction_history.json'
        self.performance_monitor = PerformanceMonitor(upbit_client)
        self.performance_monitor.update_weights(self.model_weights)
        logger.info("TradingLoop initialized. Starting initial data loading and model training.")
        initial_data = self.data_manager.ensure_sufficient_data()
        self.train_models(initial_data)
        self.total_predictions = 0
        self.retraining_threshold = 20  # 20회 예측 후 재학습 검토
        self.notion_logger = NotionLogger()
        self.prediction_timestamps = {}
        self.last_trade_time = None

    def load_model_weights(self):
        default_weights = {
            'gpt': 0.2,
            'ml': 0.1,
            'xgboost': 0.1,
            'rl': 0.1,
            'lstm': 0.2,
            'arima': 0.1,
            'prophet': 0.1,
            'transformer': 0.1,
            'bitcoin_forecast': 0.1  # 새로 추가
        }

        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, 'r') as f:
                    weights = json.load(f)
                logger.info(f"모델 가중치를 {self.weights_file}에서 성공적으로 로드했습니다.")

                # 새로운 모델이 추가되었을 경우를 대비해 기본 가중치와 병합
                return {**default_weights, **weights}
            except Exception as e:
                logger.error(f"모델 가중치 로드 중 오류 발생: {str(e)}")

        logger.warning("모델 가중치 파일이 없습니다. 기본 가중치를 사용합니다.")
        return default_weights

    def save_model_weights(self):
        try:
            with open(self.weights_file, 'w') as f:
                json.dump(self.model_weights, f, indent=4)
            logger.info(f"모델 가중치를 {self.weights_file}에 성공적으로 저장했습니다.")
        except Exception as e:
            logger.error(f"모델 가중치 저장 중 오류 발생: {str(e)}")

    def initialize_bitcoin_forecast_predictor(self):
        predictor = BitcoinForecastPredictor()
        model_path = 'bitcoin_forecast_model'

        if os.path.exists(f"{model_path}_weights.pth"):
            logger.info("기존 BitcoinForecast 모델을 로드합니다.")
            predictor.load_model(model_path)
        else:
            logger.info("새로운 BitcoinForecast 모델을 학습합니다.")
            initial_data = self.data_manager.ensure_sufficient_data()
            predictor.train(initial_data)
            predictor.save_model(model_path)

        return predictor

    def load_prediction_history(self):
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
                # JSON에서 로드한 타임스탬프를 float으로 변환
                for item in history:
                    item['timestamp'] = float(item['timestamp'])
                return deque(history, maxlen=100)
        except FileNotFoundError:
            logger.warning(f"{self.history_file} 파일을 찾을 수 없습니다. 새로운 히스토리를 시작합니다.")
            return deque(maxlen=100)
        except json.JSONDecodeError:
            logger.error(f"{self.history_file} 파일을 디코딩하는 데 실패했습니다. 새로운 히스토리를 시작합니다.")
            return deque(maxlen=100)

    def save_prediction_history(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(list(self.prediction_history), f, indent=4, default=str)
            logger.info(f"예측 히스토리를 {self.history_file}에 저장했습니다.")
        except Exception as e:
            logger.error(f"예측 히스토리 저장 중 오류 발생: {e}")

    def adjust_model_weights(self, predictions: Dict[str, float], current_price: float):
        total_error = 0
        for model, prediction in predictions.items():
            error = abs(prediction - current_price) / current_price
            total_error += error

            # 오차가 10% 이상이면 가중치 감소, 5% 미만이면 가중치 증가
            if error > 0.1:
                self.model_weights[model] += 0.005
            elif error < 0.05:
                self.model_weights[model] += 0

        logger.info(f"조정된 모델 가중치: {self.model_weights}")

    def get_predictions(self, data: pd.DataFrame, current_price: float, market_analysis: Dict[str, Any]) -> Dict[
        str, float]:
        predictions = {}
        logger.info("Starting predictions for all models")

        models = [
            ('gpt', self.get_gpt_prediction),
            ('ml', self.ml_predictor),
            ('xgboost', self.xgboost_predictor),
            ('lstm', self.lstm_predictor),
            ('rl', self.rl_agent),
            ('arima', self.arima_predictor),
            ('prophet', self.prophet_predictor),
            ('transformer', self.transformer_predictor),
            ('bitcoin_forecast', self.bitcoin_forecast_predictor)
        ]

        for model, predictor in models:
            try:
                logger.info(f"Attempting prediction for {model}")

                if model == 'gpt':
                    pred = predictor(data, current_price, market_analysis)
                elif model == 'rl':
                    state = self.prepare_state(data)
                    action = predictor.act(state)
                    pred = current_price * (1 + (action - 1) * 0.01)
                elif model in ['ml', 'xgboost']:
                    if not predictor.is_fitted:
                        logger.info(f"{model} 모델이 학습되지 않았습니다. 학습을 시작합니다.")
                        X, y = self.data_manager.prepare_data_for_ml(data)
                        predictor.train(X, y)
                    pred = predictor.predict(data.iloc[-1:])
                elif model == 'lstm':
                    if not predictor.is_fitted:
                        logger.info("LSTM 모델이 학습되지 않았습니다. 학습을 시작합니다.")
                        predictor.train(data)
                    pred = predictor.predict(data.iloc[-predictor.seq_length:])
                elif model == 'arima':
                    if not predictor.is_trained:
                        logger.info("ARIMA 모델이 학습되지 않았습니다. 학습을 시작합니다.")
                        predictor.train(data['close'])
                    last_data_point = data.iloc[-1].to_dict()
                    pred = predictor.predict(last_data_point)
                elif model == 'prophet':
                    if not predictor.is_trained:
                        logger.info("Prophet 모델이 학습되지 않았습니다. 학습을 시작합니다.")
                        predictor.train(data)
                    last_data_point = data.iloc[-1].to_dict()
                    pred = predictor.predict(last_data_point)
                elif model == 'transformer':
                    if not predictor.is_fitted:
                        logger.info("Transformer 모델이 학습되지 않았습니다. 학습을 시작합니다.")
                        predictor.train(data)
                    pred = predictor.predict(data.iloc[-predictor.seq_length:])
                elif model == 'bitcoin_forecast':
                    if not predictor.is_fitted:
                        logger.info("BitcoinForecast 모델이 학습되지 않았습니다. 학습을 시작합니다.")
                        predictor.train(data)
                    pred = predictor.predict(data)

                if pred is None:
                    logger.warning(f"{model} 모델이 유효한 예측을 반환하지 않았습니다. 현재 가격을 사용합니다.")
                    predictions[model] = current_price
                elif isinstance(pred, (pd.Series, np.ndarray)):
                    predictions[model] = float(pred[-1])
                else:
                    predictions[model] = float(pred)

                logger.info(f"{model.upper()} prediction: {predictions[model]}")

            except Exception as e:
                logger.error(f"Error in {model.upper()} prediction: {e}")
                logger.exception(f"상세 오류 ({model}):")
                predictions[model] = current_price

        # predictions = self.detect_and_remove_outliers(predictions, current_price)   # 이상치 제거
        return predictions

    def initialize_model_performance(self):
        self.model_performance = {
            model: {
                'predictions': [],
                'actual_values': [],
                'accuracy': 0.0
            } for model in ['gpt', 'ml', 'xgboost', 'rl', 'lstm', 'arima', 'prophet', 'transformer']
        }
        logger.info("Model performance initialized.")

    def evaluate_xgboost_performance(self):
        total, correct, accuracy = self.performance_monitor.get_prediction_stats('xgboost')
        logger.info(f"XGBoost 모델 성능: 정확도 {accuracy:.2f}%, 적중/전체: {correct}/{total}")

        if accuracy < 0.5 and total >= 20:  # 정확도가 50% 미만이고 최소 20회 이상의 예측이 있을 때
            logger.warning("XGBoost 모델의 성능이 좋지 않습니다. 재학습을 시작합니다.")
            recent_data = self.data_manager.get_recent_data(1000)  # 최근 1000개의 데이터 사용
            self.xgboost_predictor.retrain(recent_data)
            logger.info("XGBoost 모델 재학습 완료")

            # 성능 통계 초기화
            self.performance_monitor.reset_prediction_stats('xgboost')

    def train_models(self, data):
        logger.info(f"모델 훈련 시작. 입력 데이터 shape: {data.shape}")
        logger.info(f"입력 데이터 컬럼: {data.columns.tolist()}")

        sampled_data = data.tail(3000).copy()
        logger.info(f"샘플링된 데이터 shape: {sampled_data.shape}")

        # NaN 값 처리
        sampled_data = sampled_data.dropna()
        logger.info(f"NaN 제거 후 데이터 shape: {sampled_data.shape}")

        if len(sampled_data) < 2:
            logger.error("유효한 데이터가 충분하지 않습니다.")
            return

        X, y = self.data_manager.prepare_data_for_ml(sampled_data)
        logger.info(f"ML 데이터 준비 완료. X shape: {X.shape}, y shape: {y.shape}")

        logger.info("Training ML predictor")
        self.ml_predictor.train(X, y)
        logger.info("Training XGBoost predictor")
        self.xgboost_predictor.train(X, y)
        logger.info("Training LSTM predictor")
        self.lstm_predictor.train(sampled_data)
        logger.info("Training RL agent")
        self.rl_agent.train(sampled_data)

        logger.info("ARIMA predictor 훈련 시작")
        arima_data = sampled_data['close'].astype(float)
        logger.info(f"ARIMA 입력 데이터 shape: {arima_data.shape}")
        logger.info(f"ARIMA 입력 데이터 샘플:\n{arima_data.head()}")
        self.arima_predictor.train(arima_data)

        logger.info("Prophet predictor 훈련 시작")
        try:
            # Prophet 모델 하이퍼파라미터 최적화
            logger.info("Prophet 모델 하이퍼파라미터 최적화 시작")
            self.prophet_predictor.optimize_hyperparameters(sampled_data)
            logger.info("Prophet 모델 하이퍼파라미터 최적화 완료")

            # 최적화된 하이퍼파라미터로 Prophet 모델 훈련
            logger.info("최적화된 하이퍼파라미터로 Prophet 모델 훈련 시작")
            self.prophet_predictor.train(sampled_data)
            logger.info("Prophet 모델 훈련 완료")

            # Prophet 모델 성능 평가
            accuracy = self.prophet_predictor.get_accuracy()
            logger.info(f"Prophet 모델 정확도: {accuracy:.2f}")
        except Exception as e:
            logger.error(f"Prophet 모델 훈련 중 오류 발생: {e}")
            logger.exception("상세 오류:")

        logger.info("Training Transformer predictor")
        try:
            self.transformer_predictor.train(sampled_data)
            logger.info("Transformer 모델 훈련 완료")
        except Exception as e:
            logger.error(f"Transformer 모델 훈련 중 오류 발생: {e}")
            logger.exception("상세 오류:")
        logger.info("forcast predictor 훈련 시작")
        self.bitcoin_forecast_predictor.train(data)
        logger.info("모든 모델 훈련 완료")

    def run(self):
        logger.info("Trading loop 시작")
        logger.info(f"현재 저장된 예측 기록 수: {len(self.prediction_history)}")
        while True:
            try:
                current_time = datetime.now()
                # 10분 전 예측 결과 업데이트 (매 반복마다 체크)
                self.update_past_predictions(current_time)

                if self.last_trade_time is None or (current_time - self.last_trade_time).total_seconds() >= self.trading_interval:
                    logger.info("Starting trading cycle.")

                    logger.info("#" * 50)
                    logger.info(f"{self.total_predictions} 회 예측 후 모델 재학습 검토")
                    logger.info("#" * 50)

                    # 1. 신규 데이터 수집 및 DB 업데이트
                    logger.info("1. 신규 데이터 수집 및 DB 업데이트")
                    try:
                        self.data_manager.update_db_with_new_data()
                    except Exception as e:
                        logger.error(f"데이터 업데이트 중 오류 발생: {e}")
                        logger.exception("상세 오류:")
                        continue  # 다음 주기로 넘어감

                    data = self.data_manager.ensure_sufficient_data()
                    data_with_indicators = self.data_manager.add_technical_indicators(data)

                    if data_with_indicators.empty or len(data_with_indicators) < 33:
                        logger.warning("충분한 데이터가 없습니다. 다음 주기를 기다립니다.")
                        time.sleep(60)
                        continue

                    # 예측 후 카운터 증가 및 재학습 검토
                    self.total_predictions += 1
                    if self.total_predictions >= self.retraining_threshold:
                        self.check_and_retrain_models()

                    # 2. 기존 예측 평가 및 가중치 반영
                    logger.info("2. 기존 예측 평가 및 가중치 반영")
                    self.evaluate_predictions()
                    # XGBoost 성능 평가 및 필요시 재학습
                    self.evaluate_xgboost_performance()

                    # 3. 최신 데이터로 각 모델 예측값 받기
                    logger.info("3. 최신 데이터로 각 모델 예측값 받기")
                    current_price = self.upbit_client.get_current_price("KRW-BTC")
                    logger.info(f"Current BTC price: {current_price}")
                    market_analysis = self.analyze_market(data_with_indicators)
                    self.current_predictions = self.get_predictions(data_with_indicators, current_price, market_analysis)

                    # BitcoinForecast 모델 예측 추가
                    bitcoin_forecast_prediction = self.bitcoin_forecast_predictor.predict(data_with_indicators)
                    self.current_predictions['bitcoin_forecast'] = bitcoin_forecast_prediction
                    logger.info(f"BitcoinForecast prediction: {bitcoin_forecast_prediction}")

                    # Notion에 예측 기록
                    for model, prediction in self.current_predictions.items():
                        self.notion_logger.log_prediction(
                            timestamp=current_time,
                            model_name=model,
                            current_btc_price=current_price,
                            predicted_price=prediction
                        )
                        self.prediction_timestamps[model] = current_time

                    # 4. 가중치와 모델별 예측 정도에 따른 계산 및 최종 결정
                    logger.info("4. 가중치와 모델별 예측 정도에 따른 계산 및 최종 결정")
                    weighted_prediction = self.calculate_weighted_prediction(self.current_predictions, current_price)
                    decision = self.make_decision(self.current_predictions, current_price)
                    logger.info(f"Decision: {decision}")
                    logger.info("모델별 가중치:")
                    for model, weight in decision['weights'].items():
                        logger.info(f"  - {model.upper()}: {weight:.4f}")

                    # 5. 모델별 예측값과 최종 예측 등 기록 저장
                    logger.info("5. 모델별 예측값과 최종 예측 등 기록 저장")
                    self.performance_monitor.record_prediction(self.current_predictions, weighted_prediction, current_price)

                    # 6. 거래 실행
                    logger.info("6. 거래 실행")
                    self.cancel_existing_orders()
                    result = self.execute_trade(decision, current_price)
                    if result['success']:
                        logger.info("Trade executed successfully.")
                        self.performance_monitor.record_trade(result)
                    else:
                        logger.warning(f"Trade execution failed: {result['message']}")

                    # 7. summary 리포트 보내기
                    logger.info("7. summary 리포트 보내기")
                    performance_summary = self.performance_monitor.get_performance_summary()
                    send_discord_message(performance_summary)

                    self.last_trade_time = current_time
                    logger.info("Trading cycle completed.")

                # 주기적인 업데이트 (1시간마다)
                if self.last_trade_time and (current_time - self.last_trade_time).total_seconds() % 3600 < 10:
                    logger.info("Performing periodic update.")
                    self.periodic_update()

                # 다음 거래 시간까지 대기
                if self.last_trade_time:
                    time_to_next_trade = max(0, (self.last_trade_time + timedelta(seconds=self.trading_interval) - datetime.now()).total_seconds())
                    logger.debug(f"Sleeping for {time_to_next_trade:.2f} seconds until next trading cycle.")
                    time.sleep(min(time_to_next_trade, 10))  # 최대 10초 간격으로 체크

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                logger.error("Exception details:", exc_info=True)
                logger.error(f"Current state - Last trade time: {self.last_trade_time}, Counter: {self.counter}")
                logger.error(f"Current predictions: {self.current_predictions}")
                time.sleep(60)

    def update_past_predictions(self, current_time):
        ten_minutes_ago = current_time - timedelta(minutes=10)
        for model, timestamp in list(self.prediction_timestamps.items()):
            if timestamp <= ten_minutes_ago:
                try:
                    actual_price = self.upbit_client.get_current_price("KRW-BTC")
                    self.notion_logger.update_actual_price(timestamp, model, actual_price)
                    del self.prediction_timestamps[model]
                    logger.info(f"Updated actual price for {model} prediction made at {timestamp}")
                except Exception as e:
                    logger.error(f"Error updating actual price for {model}: {e}")

    def check_and_retrain_models(self):
        logger.info(f"총 {self.total_predictions}회 예측 완료. 모델 재학습 검토 시작.")

        # 모델 성능 정보 수집
        model_performances = {}
        for model in self.model_weights.keys():
            if model != 'gpt':  # GPT 모델 제외
                total, correct, accuracy = self.performance_monitor.get_prediction_stats(model)
                model_performances[model] = accuracy

        # 성능이 가장 낮은 3개 모델 선별
        models_to_retrain = sorted(model_performances, key=model_performances.get)[:3]

        logger.info(f"재학습 대상 모델: {models_to_retrain}")

        # 선별된 모델 재학습
        for model in models_to_retrain:
            self.retrain_model(model)

        # 모델 가중치 및 성능 데이터 저장
        self.save_model_weights()
        self.performance_monitor.save_data()

        # 총 예측 횟수 초기화
        self.total_predictions = 0

    def retrain_model(self, model_name):
        logger.info(f"{model_name} 모델 재학습 시작")

        # 최신 데이터 가져오기
        recent_data = self.data_manager.get_recent_data(5000)  # 최근 5000개 데이터 사용

        if model_name == 'ml':
            X, y = self.data_manager.prepare_data_for_ml(recent_data)
            self.ml_predictor.train(X, y)
        elif model_name == 'xgboost':
            X, y = self.data_manager.prepare_data_for_ml(recent_data)
            self.xgboost_predictor.train(X, y)
        elif model_name == 'lstm':
            # LSTM 모델 재학습 시 하이퍼파라미터 랜덤 조정
            self.lstm_predictor.randomize_hyperparameters()
            self.lstm_predictor.train(recent_data)
        elif model_name == 'rl':
            # RL 모델 재학습 시 탐험 비율(epsilon) 조정
            self.rl_agent.adjust_exploration_rate()
            self.rl_agent.train(recent_data)
        elif model_name == 'arima':
            self.arima_predictor.train(recent_data)
        elif model_name == 'prophet':
            self.prophet_predictor.train(recent_data)
        elif model_name == 'transformer':
            # Transformer 모델 재학습 시 학습률 조정
            self.transformer_predictor.adjust_learning_rate()
            self.transformer_predictor.train(recent_data)
        elif model_name == 'bitcoin_forecast':
            self.bitcoin_forecast_predictor.train(recent_data)
        # GPT 모델은 재학습이 필요 없으므로 제외

        # 성능 데이터 초기화
        self.performance_monitor.reset_prediction_stats(model_name)

        # 모델 가중치 초기화
        self.model_weights[model_name] = 0.125

        logger.info(f"{model_name} 모델 재학습 완료")

    def prepare_gpt4_prompt(self, data: pd.DataFrame, market_analysis: Dict[str, Any]) -> str:
        recent_data = data.tail(10).to_dict(orient='records')
        prompt = f"""
        최근 비트코인 데이터:
        {json.dumps(recent_data, indent=2)}

        시장 분석:
        {json.dumps(market_analysis, indent=2)}

        위 정보를 바탕으로 다음 10분 동안의 비트코인 가격 움직임을 예측하고, 10분뒤의 가격을 예측해주세요.
        응답은 다음 JSON 형식으로 제공해주세요:
        {{
            "predicted_price_after_10_min": float,
            "confidence": float (0-1),
            "reasoning": string
        }}
        """
        return prompt

    def execute_trade(self, decision: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        logger.info(f"거래 실행 시작: 결정={decision}, 현재 가격={current_price}")
        try:
            krw_balance = self.upbit_client.get_balance("KRW")
            btc_balance = self.upbit_client.get_balance("BTC")
            total_balance = krw_balance + (btc_balance * current_price)

            logger.info(f"현재 잔고: KRW={krw_balance}, BTC={btc_balance}, 총 자산가치={total_balance}")

            expected_price = decision['target_price']
            price_diff_ratio = abs(expected_price - current_price) / current_price
            trade_ratio = self.calculate_trade_ratio(price_diff_ratio)

            min_trade_amount = self.config['min_trade_amount']

            if expected_price > current_price:
                action = 'buy'
                max_buy_amount = min(krw_balance, total_balance * trade_ratio)

                if max_buy_amount < min_trade_amount:
                    # 남은 KRW 잔고 전부를 사용하여 매수
                    amount_to_buy = krw_balance / current_price
                else:
                    amount_to_buy = max_buy_amount / current_price

                if amount_to_buy * current_price < 1:  # 1원 미만의 거래는 무시
                    logger.info("매수 금액이 1원 미만입니다. 거래를 건너뜁니다.")
                    return {"success": True, "message": "No trade executed (amount too small)", "uuid": None}

                logger.info(f"매수 시도: {amount_to_buy} BTC (약 {amount_to_buy * current_price} KRW)")
                result = self.upbit_client.buy_limit_order("KRW-BTC", current_price, amount_to_buy)
            elif expected_price < current_price:
                action = 'sell'
                max_sell_amount = min(btc_balance, (btc_balance * current_price * trade_ratio) / current_price)

                if max_sell_amount * current_price < min_trade_amount:
                    # 남은 BTC 잔고 전부를 매도
                    amount_to_sell = btc_balance
                else:
                    amount_to_sell = max_sell_amount

                if amount_to_sell * current_price < 1:  # 1원 미만의 거래는 무시
                    logger.info("매도 금액이 1원 미만입니다. 거래를 건너뜁니다.")
                    return {"success": True, "message": "No trade executed (amount too small)", "uuid": None}

                logger.info(f"매도 시도: {amount_to_sell} BTC (약 {amount_to_sell * current_price} KRW)")
                result = self.upbit_client.sell_limit_order("KRW-BTC", current_price, amount_to_sell)
            else:
                logger.info("예상 가격이 현재 가격과 동일합니다. 거래를 건너뜁니다.")
                return {"success": True, "message": "No trade executed", "uuid": None}

            logger.info(f"주문 결과: {result}")
            if result and 'uuid' in result:
                logger.info(f"{action} 주문 성공: UUID={result['uuid']}")
                return {"success": True, "message": f"{action} order placed", "uuid": result['uuid']}
            else:
                logger.error("주문 실패")
                return {"success": False, "message": "Failed to place order", "uuid": None}

        except Exception as e:
            logger.error(f"거래 실행 중 오류 발생: {e}")
            logger.exception("상세 오류:")
            return {"success": False, "message": f"Error: {str(e)}", "uuid": None}

    def get_gpt_prediction(self, data: pd.DataFrame, current_price: float, market_analysis: Dict[str, Any]) -> float:
        prompt = self.prepare_gpt4_prompt(data, market_analysis)

        try:
            response = self.openai_client.chat_completion(
                model="o1-mini",
                messages=[
                    {"role": "system", "content": "You are an AI assistant specialized in Bitcoin price prediction."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                response_format={"type": "json_object"}
            )

            gpt_prediction = json.loads(response.choices[0].message.content)
            logger.info(f"GPT prediction: {gpt_prediction}")

            # GPT의 예측값을 직접 반환
            predicted_price = gpt_prediction.get('predicted_price_after_10_min', current_price)

            # 예측값이 현재 가격의 5% 이상 차이나면 현재 가격으로 조정
            if abs(predicted_price - current_price) / current_price > 0.05:
                logger.warning(f"GPT 예측값 ({predicted_price}) 비정상. 현재 가격 ({current_price})으로 조정")
                predicted_price = current_price

            return predicted_price

        except Exception as e:
            logger.error(f"Error in GPT prediction: {e}")
            return current_price

    def analyze_market(self, data: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Analyzing market conditions.")
        current_price = data['close'].iloc[-1]
        sma_20 = data['sma'].iloc[-1]
        rsi_14 = data['rsi'].iloc[-1]

        analysis = {
            'current_price': current_price,
            'sma_20': sma_20,
            'rsi_14': rsi_14,
            'trend': 'Bullish' if current_price > sma_20 else 'Bearish',
            'overbought_oversold': 'Overbought' if rsi_14 > 70 else 'Oversold' if rsi_14 < 30 else 'Neutral'
        }
        logger.info(f"Market analysis: {analysis}")
        return analysis

    def prepare_state(self, data: pd.DataFrame) -> np.ndarray:
        logger.info("Preparing state for RL agent.")
        state = np.array([
            data['close'].iloc[-1],
            data['volume'].iloc[-1],
            data['close'].pct_change().iloc[-1],
            data['sma'].iloc[-1],
            data['rsi'].iloc[-1]
        ]).reshape(1, -1)
        logger.info(f"Prepared state: {state}")
        return state

    def calculate_weighted_prediction(self, predictions: Dict[str, float], current_price: float) -> float:
        self.adjust_model_weights(predictions, current_price)
        weighted_sum = sum(self.model_weights[model] * (price - current_price) for model, price in predictions.items())
        return current_price + weighted_sum

    def detect_and_remove_outliers(self, predictions: Dict[str, float], current_price: float) -> Dict[str, float]:
        valid_predictions = {}
        prediction_values = list(predictions.values())
        Q1 = np.percentile(prediction_values, 25)
        Q3 = np.percentile(prediction_values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        for model, prediction in predictions.items():
            if lower_bound <= prediction <= upper_bound:
                valid_predictions[model] = prediction
            else:
                logger.warning(f"{model} 모델의 예측값 {prediction}이 이상치로 탐지되어 제거되었습니다.")
                valid_predictions[model] = current_price  # 이상치는 현재 가격으로 대체

        return valid_predictions

    def calculate_trade_ratio(self, price_diff_ratio: float) -> float:
        if price_diff_ratio < 0.005:
            return 0.1
        elif price_diff_ratio < 0.01:
            return 0.2
        elif price_diff_ratio < 0.02:
            return 0.3
        elif price_diff_ratio < 0.03:
            return 0.4
        else:
            return 0.5


    def calculate_stop_loss_take_profit(self, current_price: float, decision: str, volatility: float) -> Tuple[
        float, float]:
        logger.info("익절/손절 가격 계산")
        base_percentage = 0.02  # 기본 2%
        volatility_factor = volatility / 0.02  # 2% 변동성을 기준으로 조정
        logger.info(f"기본 퍼센트: {base_percentage:.2f}, 변동성 팩터: {volatility_factor:.2f}")

        stop_loss_percentage = base_percentage * volatility_factor
        take_profit_percentage = base_percentage * volatility_factor

        if decision == 'buy':
            stop_loss = current_price * (1 - stop_loss_percentage)
            take_profit = current_price * (1 + take_profit_percentage)
        else:  # sell
            stop_loss = current_price * (1 + stop_loss_percentage)
            take_profit = current_price * (1 - take_profit_percentage)

        logger.info(f"손절가: {stop_loss:.2f}, 익절가: {take_profit:.2f}")
        return stop_loss, take_profit

    def evaluate_predictions(self):
        current_price = self.upbit_client.get_current_price("KRW-BTC")
        recent_predictions = self.performance_monitor.get_recent_predictions()

        new_weights = self.model_weights.copy()

        if recent_predictions:
            latest_prediction = recent_predictions[-1]
            for model, predicted_price in latest_prediction['predictions'].items():
                previous_weight = self.previous_weights.get(model, 0)

                # 예측 가격이 현재 가격과 0.01% 이상 차이나는 경우에만 평가
                if abs(predicted_price - latest_prediction['current_price']) / latest_prediction[
                    'current_price'] >= 0.0001:
                    is_correct = (predicted_price > latest_prediction['current_price'] and current_price >
                                  latest_prediction['current_price']) or \
                                 (predicted_price < latest_prediction['current_price'] and current_price <
                                  latest_prediction['current_price'])

                    if is_correct:
                        new_weights[model] = min(previous_weight + 0.02, 1.0)
                    else:
                        new_weights[model] = max(previous_weight - 0.02, 0.0)

                    self.performance_monitor.update_prediction_stats(model, is_correct)

                    total, correct, accuracy = self.performance_monitor.get_prediction_stats(model)

                    logger.info(f"{model.upper()} | 이전예측가격: {predicted_price:.0f} | "
                                f"당시BTC가격: {latest_prediction['current_price']:.0f} | "
                                f"예측: {'상승' if predicted_price > latest_prediction['current_price'] else '하락'} | "
                                f"지금 BTC가격: {current_price:.0f} | "
                                f"실결과: {'상승' if current_price > latest_prediction['current_price'] else '하락'} | "
                                f"예측 결과: {'성공' if is_correct else '실패'} | "
                                f"전 가중치: {previous_weight:.4f} | "
                                f"반영가중치값: {0.02 if is_correct else -0.02:.4f} | "
                                f"새 가중치: {new_weights[model]:.4f} | "
                                f"예측 총 횟수: {total} | 예측 성공 수: {correct} | 예측 성공률: {accuracy:.2f}%")
                else:
                    logger.info(f"{model.upper()} | 이전예측가격: {predicted_price:.0f} | "
                                f"당시BTC가격: {latest_prediction['current_price']:.0f} | "
                                f"예측: 변화 없음 (평가에서 제외)")

        # 가중치 정규화
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            for model in new_weights:
                new_weights[model] /= total_weight
        else:
            # 모든 가중치가 0인 경우, 균등하게 분배
            for model in new_weights:
                new_weights[model] = 1.0 / len(new_weights)

        # 새 가중치 저장 및 적용
        self.previous_weights = self.model_weights.copy()
        self.model_weights = new_weights
        self.performance_monitor.update_weights(self.model_weights)
        self.save_model_weights()

        logger.info("예측 평가 및 가중치 업데이트 완료")
        logger.info(f"새로운 가중치: {self.model_weights}")

    def update_model_weights(self, model: str, is_correct: bool):
        old_weight = self.model_weights[model]
        if is_correct:
            new_weight = min(old_weight + 0.02, 1.0)
        else:
            new_weight = max(old_weight - 0.02, 0.0)

        self.model_weights[model] = new_weight

        # 가중치 정규화
        total_weight = sum(self.model_weights.values())
        for m in self.model_weights:
            self.model_weights[m] /= total_weight

        self.performance_monitor.update_weights(self.model_weights.copy())  # 복사본 전달

    def adjust_prediction(self, prediction, current_price):
        if isinstance(prediction, (int, float, np.number)):
            return self.adjust_scalar_prediction(prediction, current_price)
        elif isinstance(prediction, np.ndarray):
            if prediction.size == 1:
                return self.adjust_scalar_prediction(prediction.item(), current_price)
            else:
                logger.warning(f"Received NumPy array of size {prediction.size}. Using mean value.")
                return self.adjust_scalar_prediction(np.mean(prediction), current_price)
        else:
            logger.warning(f"Unexpected prediction type: {type(prediction)}. Using current price.")
            return current_price

    def adjust_scalar_prediction(self, prediction, current_price):
        if 0 <= prediction <= 1:  # 확률 형태의 출력
            return current_price * (1 + (prediction - 0.5) * 0.02)  # 최대 1% 상승 또는 하락
        elif prediction < current_price * 0.5 or prediction > current_price * 2:
            # 예측값이 현재 가격의 절반보다 작거나 2배보다 큰 경우, 이를 비율로 해석
            return current_price * (1 + (prediction - current_price) / current_price * 0.01)  # 최대 1% 상승 또는 하락
        else:
            return prediction  # 직접적인 가격 예측

    def add_pending_evaluation(self, decision: Dict[str, Any], result: Dict[str, Any]):
        evaluation_time = time.time() + self.evaluation_delay
        self.pending_evaluations.append({
            'decision': decision,
            'result': result,
            'evaluation_time': evaluation_time
        })

    def make_decision(self, predictions: Dict[str, float], current_price: float) -> Dict[str, Any]:
        weighted_prediction = self.calculate_weighted_prediction(predictions, current_price)

        if weighted_prediction > current_price:
            decision = 'buy'
        elif weighted_prediction < current_price:
            decision = 'sell'
        else:
            decision = 'hold'

        price_diff_ratio = abs(weighted_prediction - current_price) / current_price
        trade_ratio = self.calculate_trade_ratio(price_diff_ratio)

        return {
            'decision': decision,
            'percentage': trade_ratio * 100,
            'target_price': weighted_prediction,
            'current_price': current_price,
            'predictions': predictions,
            'weights': self.model_weights
        }

    def evaluate_pending_trades(self):
        current_time = time.time()
        evaluated_count = 0
        while self.pending_evaluations and self.pending_evaluations[0]['evaluation_time'] <= current_time:
            trade = self.pending_evaluations.popleft()
            actual_price = self.upbit_client.get_current_price("KRW-BTC")

            # PerformanceMonitor에 예측 결과 기록
            self.performance_monitor.record_prediction(trade['decision'], actual_price)

            for model, prediction in trade['decision'].get('predictions', {}).items():
                current_price = trade['decision'].get('current_price', actual_price)
                is_correct = (prediction > current_price and actual_price > current_price) or \
                             (prediction < current_price and actual_price < current_price)
                self.performance_monitor.update_model_performance(model, is_correct)
                self.update_model_weights(model, is_correct)

            # 'weighted_prediction' 또는 'target_price'가 없는 경우 대체값 사용
            weighted_prediction = trade['decision'].get('weighted_prediction') or \
                                  trade['decision'].get('target_price') or \
                                  actual_price

            logger.info(f"거래 평가: 결정={trade['decision'].get('decision', 'unknown')}, "
                        f"예측 가격={weighted_prediction:.0f}, 실제 가격={actual_price:.0f}")
            evaluated_count += 1

    def cancel_existing_orders(self):
        logger.info("Cancelling existing orders.")
        try:
            orders = self.upbit_client.get_order("KRW-BTC")
            for order in orders:
                self.upbit_client.cancel_order(order['uuid'])
            logger.info(f"Cancelled {len(orders)} orders.")
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")

    def periodic_update(self):
        self.auto_adjuster.adjust_params()
        self.dynamic_adjuster.adjust_threshold(self.calculate_market_volatility(self.data_manager.get_recent_data()))
        performance_summary = self.performance_monitor.get_performance_summary()
        # # logger.info(f"성능 요약:\n{performance_summary}")
        # send_discord_message(performance_summary)

    def calculate_market_volatility(self, data: pd.DataFrame) -> float:
        returns = data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 연간 변동성
        logger.info(f"계산된 시장 변동성: {volatility}")
        return volatility

# Main function
def main():
    logger.info("Starting main function.")
    config = load_config()
    upbit_client = UpbitClient(config['upbit_access_key'], config['upbit_secret_key'])
    openai_client = OpenAIClient(config['openai_api_key'])
    data_manager = DataManager(upbit_client)

    logger.info("Initializing predictors and agents.")
    ml_predictor = MLPredictor()
    xgboost_predictor = XGBoostPredictor()
    lstm_predictor = LSTMPredictor()
    rl_agent = RLAgent(state_size=5, action_size=3)
    auto_adjuster = AutoAdjustment(config.get('initial_params', {}))
    anomaly_detector = AnomalyDetector()
    regime_detector = MarketRegimeDetector()
    dynamic_adjuster = DynamicTradingFrequencyAdjuster()
    performance_monitor = PerformanceMonitor(upbit_client)

    logger.info("Creating TradingLoop instance.")
    trading_loop = TradingLoop(
        upbit_client=upbit_client,
        openai_client=openai_client,
        data_manager=data_manager,
        ml_predictor=ml_predictor,
        xgboost_predictor=xgboost_predictor,
        lstm_predictor=lstm_predictor,
        rl_agent=rl_agent,
        auto_adjuster=auto_adjuster,
        anomaly_detector=anomaly_detector,
        regime_detector=regime_detector,
        dynamic_adjuster=dynamic_adjuster,
        performance_monitor=performance_monitor,
        config=config
    )

    logger.info("Loading initial data and training models.")
    initial_data = data_manager.ensure_sufficient_data()
    trading_loop.train_models(initial_data)

    logger.info("Starting trading loop.")
    trading_loop.run()


if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Logging setup completed.")

    # 여기에 메인 로직 추가
    main()