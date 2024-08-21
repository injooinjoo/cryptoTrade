import logging
import random
import time

import pandas as pd
import pyupbit
from openai import OpenAI

logger = logging.getLogger(__name__)


class UpbitClient:
    def __init__(self, access_key: str, secret_key: str):
        self.upbit = pyupbit.Upbit(access_key, secret_key)

    def get_balance(self, ticker: str) -> float:
        return self.upbit.get_balance(ticker)

    def set_initial_balance(self, balance: float):
        self.initial_balance = balance

    def get_avg_buy_price(self, ticker: str) -> float:
        return self.upbit.get_avg_buy_price(ticker)

    def get_current_price(self, ticker: str, max_retries=3, delay=1) -> float:
        for attempt in range(max_retries):
            try:
                # logger.info(f"Attempting to get current price for {ticker}. Attempt {attempt + 1}/{max_retries}")
                price = pyupbit.get_current_price(ticker)
                if price is not None:
                    if isinstance(price, dict):
                        # 만약 price가 딕셔너리라면, 'trade_price' 키를 사용
                        return float(price.get('trade_price', 0))
                    elif isinstance(price, (int, float)):
                        # price가 이미 숫자라면 그대로 반환
                        return float(price)
                    else:
                        logger.warning(f"Unexpected price format for {ticker}: {price}")
                        return 0.0
                # logger.warning(f"Received None price for {ticker}. Attempt {attempt + 1}/{max_retries}")
            except Exception as e:
                logger.error(f"Error getting current price for {ticker}: {e}")

            if attempt < max_retries - 1:
                sleep_time = delay * (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"Retrying in {sleep_time:.2f} seconds")
                time.sleep(sleep_time)

        logger.error(f"Failed to get current price for {ticker} after {max_retries} attempts")
        return 0.0  # 모든 시도가 실패하면 0을 반환

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

    def buy_limit_order(self, ticker: str, price: float, volume: float):
        return self.upbit.buy_limit_order(ticker, price, volume)

    def sell_market_order(self, ticker: str, volume: float):
        try:
            return self.upbit.sell_market_order(ticker, volume)
        except Exception as e:
            logger.error(f"Error in sell_market_order for {ticker}: {e}")
            return None

    def sell_limit_order(self, ticker: str, price: float, volume: float):
        return self.upbit.sell_limit_order(ticker, price, volume)

    def cancel_order(self, uuid: str):
        return self.upbit.cancel_order(uuid)

    def get_order(self, ticker: str):
        return self.upbit.get_order(ticker)


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