import logging
import time
from typing import Dict, Any

from discord_notifier import send_discord_message

logger = logging.getLogger(__name__)


class MonitoringSystem:
    def __init__(self, upbit_client, check_interval: int = 60):
        self.upbit_client = upbit_client
        self.check_interval = check_interval
        self.metrics = {
            'total_balance': [],
            'btc_price': [],
            'trading_volume': [],
            'profit_loss': []
        }

    def start_monitoring(self):
        """Start monitoring the system metrics and checking alerts."""
        while True:
            try:
                self.update_metrics()
                self.check_alerts()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring: {e}")

    def update_metrics(self):
        """Update the system metrics including total balance and BTC price."""
        btc_balance = self.upbit_client.get_balance("BTC")
        krw_balance = self.upbit_client.get_balance("KRW")
        btc_price = self.upbit_client.get_current_price("KRW-BTC")

        total_balance = krw_balance + (btc_balance * btc_price)

        self.metrics['total_balance'].append(total_balance)
        self.metrics['btc_price'].append(btc_price)

        # 거래량과 손익은 별도의 로직이 필요합니다. 여기서는 임시로 0으로 설정합니다.
        self.metrics['trading_volume'].append(0)
        self.metrics['profit_loss'].append(0)

        logger.info(f"Updated metrics: Total balance: {total_balance}, BTC price: {btc_price}")

    def check_alerts(self):
        """Check for any significant changes in metrics and send alerts if necessary."""
        if len(self.metrics['total_balance']) > 1:
            prev_balance = self.metrics['total_balance'][-2]
            current_balance = self.metrics['total_balance'][-1]
            change_percent = (current_balance - prev_balance) / prev_balance * 100

            if abs(change_percent) > 5:  # 5% 이상 변동 시 알림
                message = f"Alert: Total balance changed by {change_percent:.2f}% in the last {self.check_interval} seconds"
                logger.warning(message)
                send_discord_message(message)

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a performance report based on the current metrics."""
        return {
            'total_balance': self.metrics['total_balance'][-1] if self.metrics['total_balance'] else None,
            'btc_price': self.metrics['btc_price'][-1] if self.metrics['btc_price'] else None,
            'trading_volume_24h': sum(self.metrics['trading_volume'][-24:]),  # 최근 24시간의 거래량
            'profit_loss_24h': sum(self.metrics['profit_loss'][-24:])  # 최근 24시간의 손익
        }


