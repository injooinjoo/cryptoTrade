import time
from threading import Thread

import pyupbit
from openai import OpenAI
import logging
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class UpbitClient:
    def __init__(self, access_key: str, secret_key: str):
        self.upbit = pyupbit.Upbit(access_key, secret_key)

    def get_balance(self, ticker: str) -> float:
        return self.upbit.get_balance(ticker)

    def get_avg_buy_price(self, ticker: str) -> float:
        return self.upbit.get_avg_buy_price(ticker)

    def get_current_price(self, ticker: str) -> float:
        return pyupbit.get_current_price(ticker)

    def get_ohlcv(self, ticker: str, interval: str, count: int) -> pd.DataFrame:
        try:
            if interval == 'week':
                logger.info(f"Fetching weekly data for {ticker}")
                df = self.get_daily_ohlcv(ticker, count * 7)
                return self.resample_to_weekly(df)
            else:
                logger.info(f"Fetching {interval} data for {ticker}")
                df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
                if df is None or df.empty:
                    logger.warning(f"Retrieved empty dataframe for {ticker} with interval {interval}")
                return df
        except Exception as e:
            logger.error(f"Error in get_ohlcv for {ticker} with interval {interval}: {e}")
            return pd.DataFrame()

    def get_daily_ohlcv(self, ticker: str, count: int) -> pd.DataFrame:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=count)
            logger.info(f"Fetching daily data for {ticker} from {start_date} to {end_date}")
            df = pyupbit.get_ohlcv(ticker, interval="day", to=end_date, count=count)
            if df is None or df.empty:
                logger.warning(f"Retrieved empty dataframe for daily data of {ticker}")
            return df
        except Exception as e:
            logger.error(f"Error in get_daily_ohlcv for {ticker}: {e}")
            return pd.DataFrame()

    def resample_to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Resampling daily data to weekly")
            if df.empty:
                logger.warning("Empty dataframe provided for resampling")
                return pd.DataFrame()

            df_weekly = df.resample('W').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'value': 'sum'
            }).dropna()

            if df_weekly.empty:
                logger.warning("Resampled weekly dataframe is empty")
            else:
                logger.info(f"Successfully resampled to weekly data. Shape: {df_weekly.shape}")

            return df_weekly
        except Exception as e:
            logger.error(f"Error in resample_to_weekly: {e}")
            return pd.DataFrame()

    def buy_limit_order(self, ticker: str, price: float, volume: float):
        return self.upbit.buy_limit_order(ticker, price, volume)

    def sell_limit_order(self, ticker: str, price: float, volume: float):
        return self.upbit.sell_limit_order(ticker, price, volume)

    def cancel_order(self, uuid: str):
        return self.upbit.cancel_order(uuid)

    def get_order(self, ticker: str):
        return self.upbit.get_order(ticker)

    def cancel_existing_orders(self):
        try:
            orders = self.get_order("KRW-BTC")
            for order in orders:
                self.cancel_order(order['uuid'])
            logger.info("All existing orders cancelled successfully.")
        except Exception as e:
            logger.error(f"Failed to cancel existing orders: {e}")

    def check_connection(self) -> bool:
        try:
            balance = self.get_balance("KRW")
            logger.info(f"Successfully connected to Upbit. KRW balance: {balance}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Upbit API: {e}")
            return False

    def get_orderbook(self, ticker: str):
        """
        Get the current orderbook for the specified ticker.
        """
        try:
            logger.info(f"Fetching orderbook for {ticker}")
            orderbook = pyupbit.get_orderbook(ticker)

            logger.debug(f"Raw orderbook data: {orderbook}")  # 디버그 로깅 추가

            if orderbook is None:
                logger.warning(f"Received None orderbook for {ticker}")
                return None

            if isinstance(orderbook, list) and len(orderbook) > 0:
                return orderbook[0]
            elif isinstance(orderbook, dict):
                return orderbook
            else:
                logger.warning(f"Unexpected orderbook format for {ticker}: {type(orderbook)}")
                return orderbook  # 원본 데이터 그대로 반환

        except Exception as e:
            logger.error(f"Error fetching orderbook for {ticker}: {e}", exc_info=True)
            return None


class OpenAIClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def chat_completion(self, model: str, messages: list, max_tokens: int, response_format: str):
        try:
            return self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                response_format=response_format
            )
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {e}")
            raise


class OrderManager:
    def __init__(self, upbit_client, position_manager):
        self.upbit_client = upbit_client
        self.position_manager = position_manager
        self.active_orders = {}
        self.monitoring_thread = None

    def add_order(self, order_id, order_type, price, stop_loss, take_profit):
        self.active_orders[order_id] = {
            'type': order_type,
            'price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        self._set_stop_loss_take_profit(stop_loss, take_profit)

    def remove_order(self, order_id):
        if order_id in self.active_orders:
            del self.active_orders[order_id]

    def _set_stop_loss_take_profit(self, stop_loss, take_profit):
        position = self.position_manager.get_position()
        if position['amount'] > 0:
            try:
                # 기존 스탑로스, 익절 주문 취소
                self.upbit_client.cancel_existing_orders()

                # 새로운 스탑로스, 익절 주문 설정
                self.upbit_client.sell_limit_order("KRW-BTC", stop_loss, position['amount'])
                self.upbit_client.sell_limit_order("KRW-BTC", take_profit, position['amount'])
                logger.info(f"Set new stop loss at {stop_loss} and take profit at {take_profit}")
            except Exception as e:
                logger.error(f"Error setting stop loss and take profit: {e}")

    def start_monitoring(self):
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.monitoring_thread = Thread(target=self._monitor_orders)
            self.monitoring_thread.start()

    def _monitor_orders(self):
        while True:
            current_price = self.upbit_client.get_current_price("KRW-BTC")
            position = self.position_manager.get_position()

            if position['amount'] > 0:
                for order_id, order in list(self.active_orders.items()):
                    if current_price <= order['stop_loss'] or current_price >= order['take_profit']:
                        self._execute_sell(order_id, current_price, position['amount'])

            time.sleep(10)  # Check every 10 seconds

    def _execute_sell(self, order_id, current_price, amount):
        try:
            logger.info(f"Executing sell for order {order_id} at price {current_price}")
            self.upbit_client.sell_market_order("KRW-BTC", amount)
            self.remove_order(order_id)
            self.position_manager.update_position()
        except Exception as e:
            logger.error(f"Error executing sell for order {order_id}: {e}")

    def update_stop_loss_take_profit(self, stop_loss, take_profit):
        self._set_stop_loss_take_profit(stop_loss, take_profit)


class PositionManager:
    def __init__(self, upbit_client):
        self.upbit_client = upbit_client
        self.current_position = self._get_current_position()

    def _get_current_position(self):
        btc_balance = self.upbit_client.get_balance("BTC")
        avg_buy_price = self.upbit_client.get_avg_buy_price("KRW-BTC")
        return {
            'amount': btc_balance,
            'avg_price': avg_buy_price
        }

    def update_position(self):
        self.current_position = self._get_current_position()

    def get_position(self):
        return self.current_position