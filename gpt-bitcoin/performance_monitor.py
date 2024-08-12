import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from api_client import UpbitClient

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    def __init__(self, upbit_client, file_path='performance_data.csv'):
        self.upbit_client = upbit_client
        self.file_path = file_path
        self.data = self.load_data()
        self.total_trades = len(self.data)
        self.initial_btc_price = self.data['current_price'].iloc[0] if not self.data.empty else None
        self.initial_balance = self.data['balance'].iloc[0] if not self.data.empty else None
        self.start_time = pd.to_datetime(self.data['timestamp'].iloc[0],
                                         format='mixed') if not self.data.empty else datetime.now()

        self.lstm_accuracy_history = []
        self.lstm_loss_history = []
        self.lstm_mse_history = []
        self.lstm_mae_history = []

        self.decision_history = []
        self.improvement_suggestions = []

        # 전체 성과 추적을 위한 변수들
        self.total_buy_trades = self.data['decision'].value_counts().get('buy', 0)
        self.total_sell_trades = self.data['decision'].value_counts().get('sell', 0)
        self.total_hold_trades = self.data['decision'].value_counts().get('hold', 0)
        self.total_successful_trades = self.data['success'].sum() if 'success' in self.data.columns else 0
        self.total_successful_buys = self.data[(self.data['decision'] == 'buy') & (self.data['success'] == True)].shape[
            0] if 'success' in self.data.columns else 0
        self.total_successful_sells = \
        self.data[(self.data['decision'] == 'sell') & (self.data['success'] == True)].shape[
            0] if 'success' in self.data.columns else 0
        self.total_successful_holds = \
        self.data[(self.data['decision'] == 'hold') & (self.data['success'] == True)].shape[
            0] if 'success' in self.data.columns else 0
        self.hodl_performance = 0
        self.strategy_performance = 0
        self.model_accuracies = {
            'gpt': 0, 'ml': 0, 'xgboost': 0, 'rl': 0, 'lstm': 0
        }
        self.model_weights = {
            'gpt': 0.2, 'ml': 0.2, 'xgboost': 0.2, 'rl': 0.2, 'lstm': 0.2
        }
        self.initial_balance = 0
        self.current_balance = 0
        self.last_decision = {'decision': 'hold', 'price': 0, 'percentage': 0}
        self.last_result = {'success': False, 'profit': 0}
        self.buy_hold_success_rate = 0
        self.sell_success_rate = 0
        self.total_trades = 0
        self.total_profit = 0
        self.trades = []


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

    def get_performance_summary(self):
        def safe_format(value, format_spec):
            return format(value, format_spec) if value is not None else "N/A"

        current_krw, current_btc, current_total = self.get_current_balances()

        summary = f"""
        트레이딩 성과 요약
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        1. 수익률 비교:
           • 전략 진행 수익률: {safe_format(self.strategy_performance, '.2f')}%
           • HODL 수익률: {safe_format(self.hodl_performance, '.2f')}%

        2. 모델별 예측 성공률:
           • GPT: {safe_format(self.model_accuracies.get('gpt') * 100, '.2f')}%
           • ML: {safe_format(self.model_accuracies.get('ml') * 100, '.2f')}%
           • XGBoost: {safe_format(self.model_accuracies.get('xgboost') * 100, '.2f')}%
           • RL: {safe_format(self.model_accuracies.get('rl') * 100, '.2f')}%
           • LSTM: {safe_format(self.model_accuracies.get('lstm') * 100, '.2f')}%

        3. 모델별 의사결정 비중:
           • GPT: {safe_format(self.model_weights.get('gpt'), '.2f')}
           • ML: {safe_format(self.model_weights.get('ml'), '.2f')}
           • XGBoost: {safe_format(self.model_weights.get('xgboost'), '.2f')}
           • RL: {safe_format(self.model_weights.get('rl'), '.2f')}
           • LSTM: {safe_format(self.model_weights.get('lstm'), '.2f')}

        4. 자산 현황:
           • 시작 금액: {safe_format(self.initial_balance, ',.0f')} KRW
           • 현재 KRW: {safe_format(current_krw, ',.0f')} KRW
           • 현재 BTC: {safe_format(current_btc, '.8f')} BTC
           • 현재 총 자산 가치: {safe_format(current_total, ',.0f')} KRW

        5. 최근 거래 정보:
           • 결정: {self.last_decision.get('decision', 'N/A')}
           • 거래가: {safe_format(self.last_decision.get('price'), ',.0f')} KRW
           • 거래 비율: {safe_format(self.last_decision.get('percentage'), '.2f')}%

        6. 거래 성공률:
           • 매수(+홀드) 성공률: {safe_format(self.buy_hold_success_rate, '.2f')}%
           • 매도 성공률: {safe_format(self.sell_success_rate, '.2f')}%

        7. 직전 거래 결과:
           • 성공 여부: {'성공' if self.last_result.get('success') else '실패'}
           • 수익: {safe_format(self.last_result.get('profit'), ',.0f')} KRW

        마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        return summary

    def save_to_file(self):
        new_record = {
            'timestamp': pd.Timestamp.now(),
            'strategy_performance': self.strategy_performance,
            'hodl_performance': self.hodl_performance,
            'current_balance': self.current_balance,
            'total_trades': self.total_trades,
            'total_profit': self.total_profit,
            'buy_hold_success_rate': self.buy_hold_success_rate,
            'sell_success_rate': self.sell_success_rate,
            'last_decision': self.last_decision['decision'],
            'last_trade_price': self.last_decision['price'],
            'last_trade_percentage': self.last_decision['percentage'],
            'last_trade_success': self.last_result['success'],
            'last_trade_profit': self.last_result['profit']
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

    def load_data(self) -> pd.DataFrame:
        if os.path.exists(self.file_path):
            df = pd.read_csv(self.file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
            return df
        return pd.DataFrame()

    def update_lstm_metrics(self, accuracy, loss=None, predictions=None, actual_values=None):
        """
        LSTM 모델의 성능 지표를 업데이트합니다.

        :param accuracy: LSTM 모델의 정확도
        :param loss: LSTM 모델의 손실 값 (옵션)
        :param predictions: LSTM 모델의 예측값 배열 (옵션)
        :param actual_values: 실제 값 배열 (옵션)
        """
        global mse, mae
        self.lstm_accuracy_history.append(accuracy)

        if loss is not None:
            self.lstm_loss_history.append(loss)

        if predictions is not None and actual_values is not None:
            mse = np.mean((np.array(predictions) - np.array(actual_values)) ** 2)
            mae = np.mean(np.abs(np.array(predictions) - np.array(actual_values)))
            self.lstm_mse_history.append(mse)
            self.lstm_mae_history.append(mae)

        # 최근 N개의 지표만 유지 (예: 최근 100개)
        max_history_length = 100
        self.lstm_accuracy_history = self.lstm_accuracy_history[-max_history_length:]
        self.lstm_loss_history = self.lstm_loss_history[-max_history_length:]
        self.lstm_mse_history = self.lstm_mse_history[-max_history_length:]
        self.lstm_mae_history = self.lstm_mae_history[-max_history_length:]

        # 로깅
        logger.info(f"LSTM Metrics Updated - Accuracy: {accuracy:.4f}")
        if loss is not None:
            logger.info(f"LSTM Loss: {loss:.4f}")
        if predictions is not None and actual_values is not None:
            logger.info(f"LSTM MSE: {mse:.4f}, MAE: {mae:.4f}")

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
        self.last_decision = {
            'decision': decision,
            'price': trade_price,
            'percentage': trade_percentage
        }
        self.save_to_file()

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
        self.save_to_file()

    def record_trade(self, trade_info):
        self.trades.append(trade_info)
        logger.info(f"Trade recorded: {trade_info}")