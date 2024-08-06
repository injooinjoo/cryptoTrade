import pandas as pd
import json
from datetime import datetime
import os


class PerformanceMonitor:
    def __init__(self, file_path='performance_data.csv'):
        """Initialize the PerformanceMonitor with a file path for storing data."""
        self.file_path = file_path
        self.data = []

    def record(self, decision: dict, current_price: float, balance: float, btc_amount: float, params: dict, regime: str, anomalies: bool):
        self.data.append({
            'timestamp': datetime.now().isoformat(),
            'decision': decision.get('decision', 'unknown'),
            'percentage': decision.get('percentage', 0),
            'target_price': decision.get('target_price', None),
            'ml_prediction': decision.get('ml_prediction', None),
            'rl_action': decision.get('rl_action', None),
            'current_price': current_price,
            'balance': balance,
            'btc_amount': btc_amount,
            'params': json.dumps(params),
            'regime': regime,
            'anomalies': anomalies
        })

    def save_to_file(self):
        """Save recorded data to a CSV file."""
        df = pd.DataFrame(self.data)
        mode = 'a' if os.path.exists(self.file_path) else 'w'
        df.to_csv(self.file_path, mode=mode, header=(mode == 'w'), index=False)
        self.data = []  # Clear the data after saving

    def load_data(self) -> pd.DataFrame:
        """Load data from the CSV file into a DataFrame."""
        if os.path.exists(self.file_path):
            return pd.read_csv(self.file_path)
        return pd.DataFrame()

    def analyze_performance(self) -> str:
        """Analyze performance based on recorded data."""
        df = self.load_data()
        if df.empty:
            return "No data available for analysis."

        analysis = {
            'total_trades': len(df),
            'buy_trades': len(df[df['decision'] == 'buy']),
            'sell_trades': len(df[df['decision'] == 'sell']),
            'hold_decisions': len(df[df['decision'] == 'hold']),
            'avg_trade_size': df['percentage'].mean(),
            'price_change': (df['current_price'].iloc[-1] - df['current_price'].iloc[0]) / df['current_price'].iloc[0] * 100,
            'balance_change': (df['balance'].iloc[-1] - df['balance'].iloc[0]) / df['balance'].iloc[0] * 100,
            'most_common_regime': df['regime'].mode().iloc[0],
            'anomaly_frequency': df['anomalies'].mean() * 100
        }

        return json.dumps(analysis, indent=2)

    def get_recent_data(self, n: int = 100) -> list:
        """Retrieve the most recent n records of performance data."""
        df = self.load_data()
        return df.tail(n).to_dict(orient='records')

    def get_performance_summary(self) -> str:
        """Generate a summary of recent performance."""
        df = self.load_data()
        if df.empty:
            return "No data available for summary."

        recent_df = df.tail(100)  # 최근 100개의 거래 데이터만 사용

        summary = {
            'total_trades': len(recent_df),
            'buy_trades': len(recent_df[recent_df['decision'] == 'buy']),
            'sell_trades': len(recent_df[recent_df['decision'] == 'sell']),
            'hold_decisions': len(recent_df[recent_df['decision'] == 'hold']),
            'avg_trade_size': recent_df['percentage'].mean(),
            'price_change': (recent_df['current_price'].iloc[-1] - recent_df['current_price'].iloc[0]) / recent_df['current_price'].iloc[0] * 100,
            'balance_change': (recent_df['balance'].iloc[-1] - recent_df['balance'].iloc[0]) / recent_df['balance'].iloc[0] * 100,
            'most_common_regime': recent_df['regime'].mode().iloc[0],
            'anomaly_frequency': recent_df['anomalies'].mean() * 100,
            'last_trade_time': recent_df['timestamp'].iloc[-1]
        }

        return json.dumps(summary, indent=2)