import pandas as pd
import pandas_ta as ta
import logging

logger = logging.getLogger(__name__)


def calculate_macd(close_prices, fast=12, slow=26, signal=9):
    exp1 = close_prices.ewm(span=fast, adjust=False).mean()
    exp2 = close_prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({'MACD': macd, 'Signal': signal_line})


def fetch_and_prepare_data(upbit_client) -> pd.DataFrame:
    """Fetch and prepare data for analysis."""
    try:
        df = upbit_client.get_ohlcv("KRW-BTC", interval="minute60", count=24)
        if df is None or df.empty:
            logger.error("Failed to fetch OHLCV data. Returned DataFrame is None or empty.")
            return pd.DataFrame()

        logger.info(f"Successfully fetched OHLCV data. Shape: {df.shape}")

        # Calculate technical indicators
        df['SMA_10'] = ta.sma(df['close'], length=10)
        df['EMA_10'] = ta.ema(df['close'], length=10)
        df['RSI_14'] = ta.rsi(df['close'], length=14)

        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
        df['Stochastic_%K'] = stoch['STOCHk_14_3_3']
        df['Stochastic_%D'] = stoch['STOCHd_14_3_3']

        macd_data = calculate_macd(df['close'])
        df['MACD'] = macd_data['MACD']
        df['MACD_Signal'] = macd_data['Signal']

        bollinger = ta.bbands(df['close'], length=20, std=2)
        df['Upper_Band'] = bollinger['BBU_20_2.0']
        df['Middle_Band'] = bollinger['BBM_20_2.0']
        df['Lower_Band'] = bollinger['BBL_20_2.0']

        df['Market_Sentiment'] = (df['close'] - df['close'].rolling(window=40).mean()) / df['close'].rolling(window=40).std()
        df['Price_Divergence'] = (df['close'] - df['SMA_10']) / df['SMA_10']

        logger.info("All technical indicators calculated successfully")
        logger.info(f"Final DataFrame shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error in fetch_and_prepare_data: {e}")
        logger.exception("Traceback:")
        return pd.DataFrame()