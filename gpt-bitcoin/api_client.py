import pyupbit
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


class UptbitClient:
    def __init__(self, access_key, secret_key):
        self.upbit = pyupbit.Upbit(access_key, secret_key)

    def get_balance(self, ticker):
        return self.upbit.get_balance(ticker)

    def get_avg_buy_price(self, ticker):
        return self.upbit.get_avg_buy_price(ticker)

    def get_current_price(self, ticker):
        return pyupbit.get_current_price(ticker)

    def get_ohlcv(self, ticker, interval, count):
        return pyupbit.get_ohlcv(ticker, interval=interval, count=count)

    def buy_limit_order(self, ticker, price, volume):
        return self.upbit.buy_limit_order(ticker, price, volume)

    def sell_limit_order(self, ticker, price, volume):
        return self.upbit.sell_limit_order(ticker, price, volume)

    def cancel_order(self, uuid):
        return self.upbit.cancel_order(uuid)

    def get_order(self, ticker):
        return self.upbit.get_order(ticker)

    def cancel_existing_orders(self):
        try:
            orders = self.get_order("KRW-BTC")
            for order in orders:
                self.cancel_order(order['uuid'])
            logger.info("All existing orders cancelled successfully.")
        except Exception as e:
            logger.error(f"Failed to cancel existing orders: {e}")

    def check_connection(self):
        try:
            balance = self.get_balance("KRW")
            logger.info(f"Successfully connected to Upbit. KRW balance: {balance}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Upbit API: {e}")
            return False


class OpenAIClient:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def chat_completion(self, model, messages, max_tokens, response_format):
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