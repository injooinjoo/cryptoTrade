import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    def __init__(self, file_path='performance_data.csv'):
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

    def get_performance_summary(self):
        if self.data.empty:
            return "데이터가 충분하지 않습니다."

        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], format='mixed')
        recent_data = self.data[self.data['timestamp'] > datetime.now() - timedelta(hours=10)]

        total_return = (self.data['balance'].iloc[
                            -1] - self.initial_balance) / self.initial_balance * 100 if self.initial_balance else 0
        recent_return = (recent_data['balance'].iloc[-1] - recent_data['balance'].iloc[0]) / \
                        recent_data['balance'].iloc[0] * 100 if not recent_data.empty else 0

        total_price_change = (self.data['current_price'].iloc[
                                  -1] - self.initial_btc_price) / self.initial_btc_price * 100 if self.initial_btc_price else 0
        recent_price_change = (recent_data['current_price'].iloc[-1] - recent_data['current_price'].iloc[0]) / \
                              recent_data['current_price'].iloc[0] * 100 if not recent_data.empty else 0

        total_success_rate = (self.total_successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        recent_success_rate = (recent_data['success'].sum() / len(
            recent_data) * 100) if not recent_data.empty and 'success' in recent_data.columns else 0

        recent_improvements = "; ".join(self.improvement_suggestions[-3:])

        last_decision = self.data['decision'].iloc[-1] if not self.data.empty else "N/A"
        last_success = "성공" if ('success' in self.data.columns and not self.data.empty and self.data['success'].iloc[-1]) else "N/A"

        return f"""
        전체 트레이딩 성과 (프로젝트 시작 이후)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        📈 트레이딩 수익률: {total_return:.2f}%
        💹 누적 트레이딩 성과:
        🔢 총 거래 횟수: {self.total_trades}회
          • 🛒 매수: {self.total_buy_trades}회 (성공: {self.total_successful_buys}회)
          • 💰 매도: {self.total_sell_trades}회 (성공: {self.total_successful_sells}회)
          • 💼 홀딩: {self.total_hold_trades}회 (성공: {self.total_successful_holds}회)
        📉 BTC 가격 변동: {total_price_change:.2f}%
           (시작 가격: {self.initial_btc_price:,.0f} KRW, 현재 가격: {self.data['current_price'].iloc[-1]:,.0f} KRW)
        💸 총 자산 변동: {total_return:.2f}%
           (시작 자산: {self.initial_balance:,.0f} KRW, 현재 자산: {self.data['balance'].iloc[-1]:,.0f} KRW)
        ✅ 전체 거래 성공률: {total_success_rate:.2f}%

        트레이딩 성과 비교 (최근 10시간)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        📈 트레이딩 수익률: {recent_return:.2f}%
        💹 최근 트레이딩 성과:
        🔢 총 거래 횟수: {len(recent_data)}회
          • 🛒 매수: {recent_data['decision'].value_counts().get('buy', 0)}회 (성공: {recent_data[recent_data['decision'] == 'buy']['success'].sum() if 'success' in recent_data.columns else 0}회)
          • 💰 매도: {recent_data['decision'].value_counts().get('sell', 0)}회 (성공: {recent_data[recent_data['decision'] == 'sell']['success'].sum() if 'success' in recent_data.columns else 0}회)
          • 💼 홀딩: {recent_data['decision'].value_counts().get('hold', 0)}회 (성공: {recent_data[recent_data['decision'] == 'hold']['success'].sum() if 'success' in recent_data.columns else 0}회)
        📉 BTC 가격 변동: {recent_price_change:.2f}%
           (시작 가격: {recent_data['current_price'].iloc[0]:,.0f} KRW, 현재 가격: {recent_data['current_price'].iloc[-1]:,.0f} KRW)
        💸 총 자산 변동: {recent_return:.2f}%
           (시작 자산: {recent_data['balance'].iloc[0]:,.0f} KRW, 현재 자산: {recent_data['balance'].iloc[-1]:,.0f} KRW)
        ✅ 최근 거래 성공률: {recent_success_rate:.2f}%

        최근 판단 리뷰:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        마지막 결정: {last_decision}
        결과: {last_success}
        개선점: {recent_improvements}

        📊 현재 시장 상태:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        💰 현재 BTC 가격: {self.data['current_price'].iloc[-1]:,.0f} KRW
        💼 현재 총 자산: {self.data['balance'].iloc[-1]:,.0f} KRW
        🏦 보유 BTC: {self.data['btc_amount'].iloc[-1]:.8f} BTC

        🕒 프로젝트 시작 시간: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
        🕒 마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

    def save_to_file(self):
        self.data.to_csv(self.file_path, index=False)

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

    def add_decision(self, decision, reason, actual_result):
        self.decision_history.append({
            'decision': decision,
            'reason': reason,
            'actual_result': actual_result,
            'timestamp': pd.Timestamp.now()
        })
        if len(self.decision_history) > 100:  # 최근 100개의 결정만 유지
            self.decision_history.pop(0)

    def get_improvement_suggestion(self, openai_client):
        # GPT에 개선 제안을 요청
        prompt = f"다음은 최근 트레이딩 결정과 그 결과입니다:\n\n"
        for decision in self.decision_history[-5:]:  # 최근 5개의 결정만 사용
            prompt += f"결정: {decision['decision']}\n"
            prompt += f"이유: {decision['reason']}\n"
            prompt += f"실제 결과: {decision['actual_result']}\n\n"
        prompt += "이 정보를 바탕으로, 트레이딩 전략을 개선하기 위한 제안을 해주세요."

        try:
            response = openai_client.chat_completion(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "Please provide an analysis of a past decision made by GPT for a BTC trade, including the reasons behind the decision and the outcome. Afterward, please outline the factors that GPT should take into consideration to avoid repeating the same mistake in future trades. Please answer in a structured way to enhance readability."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                response_format={"type": "text"}
            )

            suggestion = response.choices[0].message.content
            self.improvement_suggestions.append(suggestion)
            if len(self.improvement_suggestions) > 10:  # 최근 10개의 제안만 유지
                self.improvement_suggestions.pop(0)

            return suggestion
        except Exception as e:
            logger.error(f"GPT-4로부터 개선 제안을 받는 중 오류 발생: {e}")
            return "개선 제안을 받는 중 오류가 발생했습니다."

