import os
from datetime import datetime
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self, file_path='performance_data.csv'):
        self.file_path = file_path
        self.data = self.load_data()
        self.total_trades = len(self.data)
        self.initial_btc_price = self.data['current_price'].iloc[0] if not self.data.empty else None
        self.initial_balance = self.data['balance'].iloc[0] if not self.data.empty else None

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
            'ml_loss': ml_loss
        }])

        new_record_cleaned = new_record.dropna(axis=1, how='all')
        self.data = pd.concat([self.data, new_record_cleaned], ignore_index=True)
        self.total_trades += 1
        self.save_to_file()

    def save_to_file(self):
        self.data.to_csv(self.file_path, index=False)

    def load_data(self) -> pd.DataFrame:
        if os.path.exists(self.file_path):
            return pd.read_csv(self.file_path)
        return pd.DataFrame()

    def get_performance_comparison(self) -> dict:
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

    def get_performance_summary(self) -> str:
        df = self.load_data()
        if df.empty:
            return "데이터가 없습니다."

        recent_df = df.tail(60)  # 최근 10시간의 데이터 (10분 * 60 = 10시간)

        # 수익률 계산
        initial_balance = recent_df['balance'].iloc[0] + recent_df['btc_amount'].iloc[0] * \
                          recent_df['current_price'].iloc[0]
        final_balance = recent_df['balance'].iloc[-1] + recent_df['btc_amount'].iloc[-1] * \
                        recent_df['current_price'].iloc[-1]
        trading_return = ((final_balance - initial_balance) / initial_balance) * 100

        # HODL 수익률 계산
        hodl_return = ((recent_df['current_price'].iloc[-1] - recent_df['current_price'].iloc[0]) /
                       recent_df['current_price'].iloc[0]) * 100

        # 거래 횟수 및 성공률 계산
        buy_trades = recent_df[recent_df['decision'] == 'buy']
        sell_trades = recent_df[recent_df['decision'] == 'sell']
        hold_with_btc = recent_df[(recent_df['decision'] == 'hold') & (recent_df['btc_amount'] > 0)]
        hold_without_btc = recent_df[(recent_df['decision'] == 'hold') & (recent_df['btc_amount'] == 0)]

        successful_buys = buy_trades[buy_trades['current_price'] < buy_trades['current_price'].shift(-1)]
        successful_sells = sell_trades[sell_trades['current_price'] > sell_trades['current_price'].shift(-1)]
        successful_holds_with_btc = hold_with_btc[
            hold_with_btc['current_price'] < hold_with_btc['current_price'].shift(-1)]
        successful_holds_without_btc = hold_without_btc[
            hold_without_btc['current_price'] > hold_without_btc['current_price'].shift(-1)]

        total_trades = len(buy_trades) + len(sell_trades) + len(hold_with_btc) + len(hold_without_btc)
        successful_trades = len(successful_buys) + len(successful_sells) + len(successful_holds_with_btc) + len(
            successful_holds_without_btc)
        success_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0

        # 최근 판단 리뷰
        last_decision = recent_df['decision'].iloc[-1]
        last_price = recent_df['current_price'].iloc[-2]
        current_price = recent_df['current_price'].iloc[-1]
        last_decision_correct = (
                (last_decision == 'buy' and current_price > last_price) or
                (last_decision == 'sell' and current_price < last_price) or
                (last_decision == 'hold' and recent_df['btc_amount'].iloc[-1] > 0 and current_price > last_price) or
                (last_decision == 'hold' and recent_df['btc_amount'].iloc[-1] == 0 and current_price < last_price)
        )

        avg_trade_size = recent_df['percentage'].mean()
        price_change = ((recent_df['current_price'].iloc[-1] / recent_df['current_price'].iloc[0]) - 1) * 100
        balance_change = trading_return

        return f"""
        📊 트레이딩 성과 비교 (최근 10시간)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        📈 트레이딩 수익률: {trading_return:.2f}% | 📉 HODL 수익률: {hodl_return:.2f}% | 🔄 초과 성과: {trading_return - hodl_return:.2f}%

        💹 누적 트레이딩 성과:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        🔢 총 거래 횟수: {total_trades}회
          • 🛒 매수: {len(buy_trades)}회 (성공: {len(successful_buys)}회)
          • 💰 매도: {len(sell_trades)}회 (성공: {len(successful_sells)}회)
          • 💼 BTC 보유 홀딩: {len(hold_with_btc)}회 (성공: {len(successful_holds_with_btc)}회)
          • 🕰️ BTC 미보유 홀딩: {len(hold_without_btc)}회 (성공: {len(successful_holds_without_btc)}회)
        📊 평균 거래 규모: 총 자산의 {avg_trade_size:.2f}% 사용
        📉 BTC 가격 변동: {price_change:.2f}% 
           (시작 가격: {recent_df['current_price'].iloc[0]:,.0f} KRW, 현재 가격: {recent_df['current_price'].iloc[-1]:,.0f} KRW)
        💸 총 자산 변동: {balance_change:.2f}% 
           (시작 자산: {initial_balance:,.0f} KRW, 현재 자산: {final_balance:,.0f} KRW)
        ✅ 전체 거래 성공률: {success_rate:.2f}% (총 {total_trades}회 중 {successful_trades}회 성공)

        🔍 최근 판단 리뷰:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        🤔 마지막 결정: {last_decision}
        📊 결과: {"성공" if last_decision_correct else "실패"}
        💡 개선점: {"없음" if last_decision_correct else "판단 기준 재검토 필요"}

        📊 현재 시장 상태:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        💰 현재 BTC 가격: {current_price:,.0f} KRW
        💼 현재 총 자산: {final_balance:,.0f} KRW
        🏦 보유 BTC: {recent_df['btc_amount'].iloc[-1]:.8f} BTC

        🕒 마지막 업데이트: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

    def _calculate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            'total_trades': len(df),
            'buy_trades': len(df[df['decision'] == 'buy']),
            'sell_trades': len(df[df['decision'] == 'sell']),
            'hold_decisions': len(df[df['decision'] == 'hold']),
            'average_trade_size': df['percentage'].mean(),
            'price_change': (df['current_price'].iloc[-1] - df['current_price'].iloc[0]) /
                            df['current_price'].iloc[0] * 100 if len(df) > 1 else 0,
            'balance_change': (df['balance'].iloc[-1] - df['balance'].iloc[0]) /
                              df['balance'].iloc[0] * 100 if len(df) > 1 else 0,
            'win_rate': (df['balance'].diff() > 0).mean() * 100
        }

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

        analysis = {
            'up_predictions': {
                'total': len(up_predictions),
                'successful': up_predictions['prediction_success'].sum(),
                'accuracy': up_predictions['prediction_success'].mean() * 100
            },
            'down_predictions': {
                'total': len(down_predictions),
                'successful': down_predictions['prediction_success'].sum(),
                'accuracy': down_predictions['prediction_success'].mean() * 100
            },
            'overall_accuracy': df['prediction_success'].mean() * 100
        }

        failure_reasons, improvement_suggestions = self._analyze_failures(df)
        analysis['failure_reasons'] = failure_reasons
        analysis['improvement_suggestions'] = improvement_suggestions

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

    def get_detailed_performance_metrics(self) -> Dict[str, Any]:
        if self.data.empty:
            return {"error": "데이터가 없습니다."}

        metrics = {
            "total_trades": len(self.data),
            "successful_trades": sum(self.data['decision'] != 'hold'),
            "win_rate": (sum(self.data['decision'] != 'hold') / len(self.data)) * 100,
            "average_return": self.data['balance'].pct_change().mean() * 100,
            "sharpe_ratio": self.calculate_sharpe_ratio(),
            "max_drawdown": self.calculate_max_drawdown(),
            "profit_loss_ratio": self.calculate_profit_loss_ratio(),
            "ml_accuracy": self.data['ml_accuracy'].iloc[-1] if 'ml_accuracy' in self.data.columns else 0,
            "ml_loss": self.data['ml_loss'].iloc[-1] if 'ml_loss' in self.data.columns else 0,
        }
        return metrics

    def calculate_sharpe_ratio(self):
        if self.data.empty:
            return 0
        returns = self.data['balance'].pct_change().dropna()
        if returns.empty:
            return 0
        return (returns.mean() / returns.std()) * np.sqrt(252)

    def calculate_max_drawdown(self):
        if self.data.empty:
            return 0
        cumulative_returns = (1 + self.data['balance'].pct_change().fillna(0)).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        return drawdown.min()

    def calculate_win_rate(self) -> float:
        profitable_trades = sum(self.data['balance'].diff() > 0)
        return (profitable_trades / len(self.data)) * 100

    def calculate_profit_loss_ratio(self) -> float:
        profits = self.data['balance'].diff()[self.data['balance'].diff() > 0].mean()
        losses = abs(self.data['balance'].diff()[self.data['balance'].diff() < 0].mean())
        return profits / losses if losses != 0 else 0

    def count_trades(self):
        return len(self.data[self.data['decision'] != 'hold'])

    def calculate_success_rate(self):
        trades = self.data[self.data['decision'] != 'hold']
        successful_trades = trades[trades['balance'].diff() > 0]
        return len(successful_trades) / len(trades) if len(trades) > 0 else 0

    def calculate_cumulative_return(self):
        initial_balance = self.data['balance'].iloc[0]
        final_balance = self.data['balance'].iloc[-1]
        return (final_balance - initial_balance) / initial_balance * 100

    def calculate_gpt4_agreement_rate(self):
        gpt4_decisions = self.data['gpt4_decision']
        actual_movements = (self.data['close'].shift(-1) > self.data['close']).astype(int)
        agreement = (gpt4_decisions == actual_movements)
        return agreement.mean()

    def plot_performance(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.data.index, self.data['balance'])
        plt.title('Balance Over Time')
        plt.xlabel('Date')
        plt.ylabel('Balance')
        plt.savefig('performance_chart.png')

    def update_ml_metrics(self, accuracy, loss):
        self.ml_accuracy = accuracy
        self.ml_loss = loss

    def update_rl_metrics(self, epsilon, avg_reward):
        self.rl_epsilon = epsilon
        self.rl_avg_reward = avg_reward

    def update_gpt4_agreement_rate(self, rate):
        self.gpt4_agreement_rate = rate

    def update_xgboost_metrics(self, accuracy, loss=None):
        self.xgboost_accuracy = accuracy
        if loss is not None:
            self.xgboost_loss = loss

    def generate_detailed_report(self):
        if self.data.empty:
            return "No data available for report generation."

        try:
            initial_balance = self.data['balance'].iloc[0]
            final_balance = self.data['balance'].iloc[-1]
            cumulative_return = (final_balance - initial_balance) / initial_balance * 100

            trades = self.data[self.data['decision'] != 'hold']
            total_trades = len(trades)
            successful_trades = len(trades[trades['balance'].diff() > 0])
            success_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0

            sharpe_ratio = self.calculate_sharpe_ratio()
            max_drawdown = self.calculate_max_drawdown() * 100

            start_price = self.data['close'].iloc[0]
            end_price = self.data['close'].iloc[-1]
            hodl_return = (end_price - start_price) / start_price * 100

            report = f"""
    트레이딩 성과 비교
    트레이딩 수익률: {cumulative_return:.2f}% | HODL 수익률: {hodl_return:.2f}% | 초과 성과: {cumulative_return - hodl_return:.2f}%

    상세 학습 진행 보고서
    머신러닝 모델: 정확도: {self.ml_accuracy:.2f}% | 손실: {self.ml_loss:.2f}%
    강화학습 에이전트: 엡실론: {self.rl_epsilon:.4f} | 평균 보상: {self.rl_avg_reward:.2f}
    GPT-4 일치율: {self.gpt4_agreement_rate:.2f}%

    누적 트레이딩 성과:
    - 총 거래 횟수: {total_trades}회
      • 매수 거래: {len(trades[trades['decision'] == 'buy'])}회
      • 매도 거래: {len(trades[trades['decision'] == 'sell'])}회
      • 홀딩 결정: {len(self.data) - total_trades}회
    - 평균 거래 규모: {trades['percentage'].mean() if 'percentage' in trades.columns else 0:.2f}%
    - 가격 변동: {((end_price / start_price) - 1) * 100:.2f}%
    - 잔고 변동: {cumulative_return:.2f}%
    - 성공률: {success_rate:.2f}%

    리스크 및 수익성 지표:
    - 샤프 비율: {sharpe_ratio:.2f}
    - 최대 낙폭 (MDD): {abs(max_drawdown):.2f}%

    현재 시장 상태:
    - 현재 가격: {end_price:,.0f} KRW
    - 현재 잔고: {final_balance:,.0f} KRW
    - 보유 BTC: {self.data['btc_amount'].iloc[-1] if 'btc_amount' in self.data.columns else 0:.8f} BTC

    마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """

            return report

        except Exception as e:
            return f"Error generating report: {str(e)}"

