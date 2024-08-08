from typing import Dict, Any, Tuple

import pandas as pd
import json
from datetime import datetime
import os


class PerformanceMonitor:
    def __init__(self, file_path='performance_data.csv'):
        self.file_path = file_path
        self.data = self.load_data()
        self.total_trades = len(self.data)
        self.initial_btc_price = self.data['current_price'].iloc[0] if not self.data.empty else None
        self.initial_balance = self.data['balance'].iloc[0] if not self.data.empty else None

    def record(self, decision: dict, current_price: float, balance: float, btc_amount: float, params: dict, regime: str, anomalies: bool):
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
            'anomalies': anomalies
        }])
        self.data = pd.concat([self.data, new_record], ignore_index=True)
        self.total_trades += 1
        self.save_to_file()

    def save_to_file(self):
        self.data.to_csv(self.file_path, index=False)

    def get_performance_comparison(self) -> Dict[str, float]:
        if self.data.empty or self.initial_btc_price is None or self.initial_balance is None:
            return {"error": "Not enough data for comparison"}

        current_price = self.data['current_price'].iloc[-1]
        current_balance = self.data['balance'].iloc[-1] + self.data['btc_amount'].iloc[-1] * current_price

        trading_return = (current_balance - self.initial_balance) / self.initial_balance * 100
        hodl_return = (current_price - self.initial_btc_price) / self.initial_btc_price * 100

        return {
            "trading_return": trading_return,
            "hodl_return": hodl_return,
            "outperformance": trading_return - hodl_return
        }

    def get_cumulative_summary(self) -> Dict[str, Any]:
        if self.data.empty:
            return {"error": "No data available for cumulative summary."}

        summary = self._calculate_summary(self.data)
        summary['total_trades'] = self.total_trades
        summary['successful_trades'] = len(self.data[self.data['decision'] != 'hold'])
        summary['success_rate'] = (summary['successful_trades'] / summary['total_trades'] * 100) if summary[
                                                                                                        'total_trades'] > 0 else 0

        return summary

    def _calculate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            'buy_trades': len(df[df['decision'] == 'buy']),
            'sell_trades': len(df[df['decision'] == 'sell']),
            'hold_decisions': len(df[df['decision'] == 'hold']),
            'avg_trade_size': df['percentage'].mean(),
            'price_change': (df['current_price'].iloc[-1] - df['current_price'].iloc[0]) /
                            df['current_price'].iloc[0] * 100 if len(df) > 1 else 0,
            'balance_change': (df['balance'].iloc[-1] - df['balance'].iloc[0]) /
                              df['balance'].iloc[0] * 100 if len(df) > 1 else 0,
        }

    def load_data(self) -> pd.DataFrame:
        if os.path.exists(self.file_path):
            return pd.read_csv(self.file_path)
        return pd.DataFrame()

    def get_performance_summary(self) -> str:
        df = self.load_data()
        if df.empty:
            return "No data available for summary."

        recent_df = df.tail(10)  # 최근 10개의 거래 데이터만 사용 (약 100분)

        summary = self._calculate_summary(recent_df)
        return "\n".join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in summary.items()])

    def get_prediction_accuracy(self) -> float:
        df = self.load_data()
        if df.empty:
            return 0.0

        recent_df = df.tail(100)  # 최근 100개의 거래 데이터만 사용
        correct_predictions = ((recent_df['decision'] == 'buy') & (
                    recent_df['current_price'] < recent_df['current_price'].shift(-1))) | \
                              ((recent_df['decision'] == 'sell') & (
                                          recent_df['current_price'] > recent_df['current_price'].shift(-1)))

        accuracy = correct_predictions.mean() * 100
        return accuracy

    def get_detailed_analysis(self) -> Dict[str, Any]:
        df = self.load_data()
        if df.empty:
            return {"error": "No data available for analysis."}

        df['price_change'] = df['current_price'].pct_change()
        df['predicted_direction'] = self._get_predicted_direction(df)
        df['actual_direction'] = (df['price_change'] > 0).astype(int)
        df['prediction_success'] = (df['predicted_direction'] == df['actual_direction']).astype(int)

        up_predictions = df[df['predicted_direction'] == 1]
        down_predictions = df[df['predicted_direction'] == 0]

        analysis = {'up_predictions': {
            'total': len(up_predictions),
            'successful': up_predictions['prediction_success'].sum(),
            'accuracy': up_predictions['prediction_success'].mean() * 100
        }, 'down_predictions': {
            'total': len(down_predictions),
            'successful': down_predictions['prediction_success'].sum(),
            'accuracy': down_predictions['prediction_success'].mean() * 100
        }, 'overall_accuracy': df['prediction_success'].mean() * 100,
            'failure_reasons': (self._analyze_failures(df))[0],
            'improvement_suggestions': (self._analyze_failures(df))[1]}

        return analysis

    def _get_predicted_direction(self, df: pd.DataFrame) -> pd.Series:
        predicted_direction = pd.Series(index=df.index)
        predicted_direction[df['decision'] == 'buy'] = 1
        predicted_direction[df['decision'] == 'sell'] = 0

        # HOLD 결정에 대한 처리
        hold_mask = df['decision'] == 'hold'
        btc_held = df['btc_amount'] > 0
        predicted_direction[hold_mask & btc_held] = 1  # BTC 보유 중 HOLD는 상승 예측
        predicted_direction[hold_mask & ~btc_held] = 0  # BTC 미보유 중 HOLD는 하락 예측

        return predicted_direction

    def _analyze_failures(self, df: pd.DataFrame) -> Tuple[str, str]:
        failed_predictions = df[df['prediction_success'] == 0]

        if failed_predictions.empty:
            return "No failures observed.", "Continue with the current strategy."

        failure_reasons = []
        improvement_suggestions = []

        # 과도한 변동성 체크
        if (failed_predictions['price_change'].abs() > failed_predictions['price_change'].abs().mean() * 2).any():
            failure_reasons.append("High market volatility")
            improvement_suggestions.append("Implement volatility filters")

        # 연속적인 실패 체크
        if (failed_predictions['prediction_success'].rolling(window=3).sum() == 0).any():
            failure_reasons.append("Consecutive prediction failures")
            improvement_suggestions.append("Review and adjust the prediction model")

        # 특정 결정에 대한 낮은 정확도 체크
        for decision in ['buy', 'sell', 'hold']:
            decision_accuracy = df[df['decision'] == decision]['prediction_success'].mean()
            if decision_accuracy < 0.4:  # 40% 미만의 정확도를 낮다고 가정
                failure_reasons.append(f"Low accuracy for {decision} decisions")
                improvement_suggestions.append(f"Refine criteria for {decision} decisions")

        return ", ".join(failure_reasons), ", ".join(improvement_suggestions)