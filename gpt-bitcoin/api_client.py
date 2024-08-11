import logging
from datetime import datetime, timedelta

import pandas as pd
import pyupbit
from openai import OpenAI

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

    def get_ohlcv(self, ticker: str, interval: str, count: int, to=None):
        try:
            if to:
                df = pyupbit.get_ohlcv(ticker, interval=interval, to=to, count=count)
            else:
                df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
            return df
        except Exception as e:
            logger.error(f"Error in get_ohlcv for {ticker} with interval {interval}: {e}")
            return pd.DataFrame()

    # def get_daily_ohlcv(self, ticker: str, count: int) -> pd.DataFrame:
    #     try:
    #         end_date = datetime.now()
    #         start_date = end_date - timedelta(days=count)
    #         logger.info(f"Fetching daily data for {ticker} from {start_date} to {end_date}")
    #         df = pyupbit.get_ohlcv(ticker, interval="day", to=end_date, count=count)
    #         if df is None or df.empty:
    #             logger.warning(f"Retrieved empty dataframe for daily data of {ticker}")
    #         return df
    #     except Exception as e:
    #         logger.error(f"Error in get_daily_ohlcv for {ticker}: {e}")
    #         return pd.DataFrame()
    #
    # def resample_to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
    #     try:
    #         logger.info("Resampling daily data to weekly")
    #         if df.empty:
    #             logger.warning("Empty dataframe provided for resampling")
    #             return pd.DataFrame()
    #
    #         df_weekly = df.resample('W').agg({
    #             'open': 'first',
    #             'high': 'max',
    #             'low': 'min',
    #             'close': 'last',
    #             'volume': 'sum',
    #             'value': 'sum'
    #         }).dropna()
    #
    #         if df_weekly.empty:
    #             logger.warning("Resampled weekly dataframe is empty")
    #         else:
    #             logger.info(f"Successfully resampled to weekly data. Shape: {df_weekly.shape}")
    #
    #         return df_weekly
    #     except Exception as e:
    #         logger.error(f"Error in resample_to_weekly: {e}")
    #         return pd.DataFrame()

    def buy_limit_order(self, ticker: str, price: float, volume: float):
        return self.upbit.buy_limit_order(ticker, price, volume)

    def sell_limit_order(self, ticker: str, price: float, volume: float):
        return self.upbit.sell_limit_order(ticker, price, volume)

    def cancel_order(self, uuid: str):
        return self.upbit.cancel_order(uuid)

    def get_order(self, ticker: str):
        return self.upbit.get_order(ticker)

    # def cancel_existing_orders(self):
    #     try:
    #         orders = self.get_order("KRW-BTC")
    #         for order in orders:
    #             self.cancel_order(order['uuid'])
    #         logger.info("All existing orders cancelled successfully.")
    #     except Exception as e:
    #         logger.error(f"Failed to cancel existing orders: {e}")
    #
    # def check_connection(self) -> bool:
    #     try:
    #         balance = self.get_balance("KRW")
    #         logger.info(f"Successfully connected to Upbit. KRW balance: {balance}")
    #         return True
    #     except Exception as e:
    #         logger.error(f"Failed to connect to Upbit API: {e}")
    #         return False
    #
    # def get_orderbook(self, ticker: str):
    #     """
    #     Get the current orderbook for the specified ticker.
    #     """
    #     try:
    #         logger.info(f"Fetching orderbook for {ticker}")
    #         orderbook = pyupbit.get_orderbook(ticker)
    #
    #         logger.debug(f"Raw orderbook data: {orderbook}")  # 디버그 로깅 추가
    #
    #         if orderbook is None:
    #             logger.warning(f"Received None orderbook for {ticker}")
    #             return None
    #
    #         if isinstance(orderbook, list) and len(orderbook) > 0:
    #             return orderbook[0]
    #         elif isinstance(orderbook, dict):
    #             return orderbook
    #         else:
    #             logger.warning(f"Unexpected orderbook format for {ticker}: {type(orderbook)}")
    #             return orderbook  # 원본 데이터 그대로 반환
    #
    #     except Exception as e:
    #         logger.error(f"Error fetching orderbook for {ticker}: {e}", exc_info=True)
    #         return None
    #
    # def calculate_hodl_return(self, start_time, end_time):
    #     start_price = self.get_historical_price("KRW-BTC", start_time)
    #     end_price = self.get_historical_price("KRW-BTC", end_time)
    #     return (end_price - start_price) / start_price * 100

    # def get_historical_price(self, ticker, timestamp):
    #     """
    #     특정 시점의 가격을 가져오는 함수
    #
    #     :param ticker: 코인 티커 (예: "KRW-BTC")
    #     :param timestamp: Unix timestamp (초 단위)
    #     :return: 해당 시점의 종가
    #     """
    #     try:
    #         # timestamp를 datetime 객체로 변환
    #         date_time = datetime.fromtimestamp(timestamp)
    #
    #         # 해당 날짜의 일봉 데이터 조회
    #         df = pyupbit.get_ohlcv(ticker, interval="day", to=date_time + timedelta(days=1), count=1)
    #
    #         if df is not None and not df.empty:
    #             return df['close'].iloc[0]
    #         else:
    #             logger.warning(f"No data found for {ticker} at {date_time}")
    #             return None
    #     except Exception as e:
    #         logger.error(f"Error getting historical price for {ticker} at {timestamp}: {e}")
    #         return None


class OpenAIClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def chat_completion(self, model: str, messages: list, max_tokens: int, response_format):
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


class PositionManager:
    def __init__(self, upbit_client):
        self.upbit_client = upbit_client
        self.current_position = self._get_current_position()

    def _get_current_position(self):
        try:
            btc_balance = self.upbit_client.get_balance("BTC")
            krw_balance = self.upbit_client.get_balance("KRW")
            btc_current_price = self.upbit_client.get_current_price("KRW-BTC")
            return {
                'btc_balance': btc_balance,
                'krw_balance': krw_balance,
                'btc_current_price': btc_current_price,
                'total_balance_krw': (btc_balance * btc_current_price) + krw_balance
            }
        except Exception as e:
            logger.error(f"Error getting current position: {e}")
            return {'btc_balance': 0, 'krw_balance': 0, 'btc_current_price': 0, 'total_balance_krw': 0}

    def update_position(self):
        self.current_position = self._get_current_position()

    def get_position(self):
        self.current_position = self._get_current_position()
        return self.current_position

    def calculate_profit_loss(self):
        position = self.get_position()
        if position['btc_balance'] > 0 and position['btc_current_price'] > 0:
            avg_buy_price = self.upbit_client.get_avg_buy_price("BTC")
            if avg_buy_price > 0:
                profit_loss = ((position['btc_current_price'] - avg_buy_price) / avg_buy_price) * 100
                return profit_loss
        return 0