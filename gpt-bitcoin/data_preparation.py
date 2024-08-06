import pandas as pd
import pandas_ta as ta
import logging
import numpy as np
from typing import Tuple, Dict
from config import load_config

logger = logging.getLogger(__name__)
config = load_config()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add various technical indicators to the DataFrame."""
    try:
        if df is None or df.empty:
            logger.warning("Empty or None dataframe provided to add_indicators")
            return pd.DataFrame()

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            missing_columns = [col for col in required_columns if col not in df.columns]
            logger.warning(f"Missing columns in dataframe: {missing_columns}")
            return df

        df = df.dropna()

        # 최소 필요 데이터 포인트 수 정의
        min_periods = {
            'SMA': 10,
            'EMA': 10,
            'RSI': 14,
            'BB': 20,
            'ATR': 14
        }

        if len(df) >= min_periods['SMA']:
            df['SMA_10'] = ta.sma(df['close'], length=10)
        else:
            logger.warning(f"Not enough data points for SMA calculation. Required: {min_periods['SMA']}, Available: {len(df)}")

        if len(df) >= min_periods['EMA']:
            df['EMA_10'] = ta.ema(df['close'], length=10)
        else:
            logger.warning(f"Not enough data points for EMA calculation. Required: {min_periods['EMA']}, Available: {len(df)}")

        if len(df) >= min_periods['RSI']:
            df['RSI_14'] = ta.rsi(df['close'], length=14)
        else:
            logger.warning(f"Not enough data points for RSI calculation. Required: {min_periods['RSI']}, Available: {len(df)}")

        if len(df) >= min_periods['BB']:
            bollinger = ta.bbands(df['close'], length=20, std=2)
            if 'BBU_20_2.0' in bollinger.columns:
                df['BB_Upper'] = bollinger['BBU_20_2.0']
                df['BB_Middle'] = bollinger['BBM_20_2.0']
                df['BB_Lower'] = bollinger['BBL_20_2.0']
            else:
                logger.warning("Bollinger Bands columns not found in the result")
        else:
            logger.warning(f"Not enough data points for Bollinger Bands calculation. Required: {min_periods['BB']}, Available: {len(df)}")

        if len(df) >= min_periods['ATR']:
            df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        else:
            logger.warning(f"Not enough data points for ATR calculation. Required: {min_periods['ATR']}, Available: {len(df)}")

        df = df.dropna()

        logger.info(f"Indicators added successfully. Dataframe shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error in add_indicators: {e}")
        return df


def safe_fetch_multi_timeframe_data(upbit_client, ticker="KRW-BTC") -> Dict[str, pd.DataFrame]:
    """Fetch and prepare OHLCV data with selected indicators across multiple timeframes."""
    timeframes = {
        'short': {'interval': 'minute60', 'count': 100},  # 이전: 24
        'medium': {'interval': 'day', 'count': 60},      # 이전: 30
        'long': {'interval': 'week', 'count': 24}        # 이전: 12
    }

    data = {}
    for tf, params in timeframes.items():
        try:
            logger.info(f"Fetching {tf} timeframe data")
            df = upbit_client.get_ohlcv(ticker, interval=params['interval'], count=params['count'])
            if df is not None and not df.empty:
                df = add_indicators(df)
                data[tf] = df
                logger.info(f"Successfully fetched and processed data for {tf} timeframe. Shape: {df.shape}")
            else:
                logger.warning(f"Empty dataframe for {tf} timeframe")
                data[tf] = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching data for {tf} timeframe: {e}")
            data[tf] = pd.DataFrame()

    return data


def prepare_data_for_ml(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare data for machine learning by extracting features and labels."""
    features = df[
        ['open', 'high', 'low', 'close', 'volume', 'SMA_10', 'EMA_10', 'RSI_14', 'BB_Upper', 'BB_Middle', 'BB_Lower',
         'ATR_14']].values
    labels = np.where(df['close'].shift(-1) > df['close'], 1, 0)[:-1]
    return features[:-1], labels


def normalize_data(data: np.ndarray) -> np.ndarray:
    """Normalize data to have zero mean and unit variance."""
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
