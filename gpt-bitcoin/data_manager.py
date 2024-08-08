import logging
import sqlite3
import time

import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import tweepy
from textblob import TextBlob
from typing import Dict, Any, Tuple
from datetime import datetime, timedelta
from config import load_config

logger = logging.getLogger(__name__)
config = load_config()

DB_PATH = 'trading_decisions.sqlite'


class DataManager:
    def __init__(self, upbit_client=None, twitter_api_key=None, twitter_api_secret=None,
                 twitter_access_token=None, twitter_access_token_secret=None):
        self.upbit_client = upbit_client
        self.twitter_api_key = twitter_api_key
        self.twitter_api_secret = twitter_api_secret
        self.twitter_access_token = twitter_access_token
        self.twitter_access_token_secret = twitter_access_token_secret
        self.twitter_api = None  # Twitter API 객체를 나중에 초기화

    def initialize_twitter_api(self):
        if all([self.twitter_api_key, self.twitter_api_secret,
                self.twitter_access_token, self.twitter_access_token_secret]):
            auth = tweepy.OAuthHandler(self.twitter_api_key, self.twitter_api_secret)
            auth.set_access_token(self.twitter_access_token, self.twitter_access_token_secret)
            self.twitter_api = tweepy.API(auth)

    @classmethod
    def initialize_db(cls):
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    decision TEXT,
                    percentage REAL,
                    target_price REAL,
                    btc_balance REAL,
                    krw_balance REAL,
                    btc_avg_buy_price REAL,
                    btc_krw_price REAL,
                    accuracy REAL
                );
            ''')
            conn.commit()
        logger.info("Database initialized successfully.")

    def log_decision(self, decision: dict) -> None:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO decisions (timestamp, decision, percentage, target_price, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, accuracy)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now(),
                    decision['decision'],
                    decision['percentage'],
                    decision.get('target_price'),
                    self.upbit_client.get_balance("BTC"),
                    self.upbit_client.get_balance("KRW"),
                    self.upbit_client.get_avg_buy_price("BTC"),
                    self.upbit_client.get_current_price("KRW-BTC"),
                    None  # Accuracy will be updated later
                ))
            except sqlite3.OperationalError as e:
                logger.error(f"Error logging decision: {e}")
                raise
            conn.commit()
        logger.info("Decision logged to database.")

    def get_previous_decision(self) -> dict:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM decisions
                ORDER BY timestamp DESC
                LIMIT 1
            ''')
            row = cursor.fetchone()
            if row:
                result = dict(row)
                logger.debug(f"Previous decision: {result}")
                return result
            logger.debug("No previous decision found")
            return None

    def update_decision_accuracy(self, decision_id: int, accuracy: float) -> None:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE decisions
                SET accuracy = ?
                WHERE id = ?
            ''', (accuracy, decision_id))
            conn.commit()
        logger.info(f"Updated accuracy for decision {decision_id}: {accuracy}")

    def get_average_accuracy(self, days: int = 7) -> float:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT AVG(accuracy)
                FROM decisions
                WHERE timestamp >= ? AND accuracy IS NOT NULL
            ''', (datetime.now() - timedelta(days=days),))
            return cursor.fetchone()[0] or 0.0

    def get_recent_decisions(self, days: int = 7) -> list:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, timestamp, decision, percentage, 
                       COALESCE(target_price, btc_krw_price) as target_price, 
                       btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, accuracy
                FROM decisions
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            ''', (datetime.now() - timedelta(days=days),))
            rows = cursor.fetchall()
            decisions = [dict(row) for row in rows]
            logger.debug(f"Recent decisions: {decisions}")
            return decisions

    def get_accuracy_over_time(self) -> dict:
        periods = [1, 7, 30]  # days
        accuracies = {}
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            for period in periods:
                cursor.execute('''
                    SELECT AVG(accuracy)
                    FROM decisions
                    WHERE timestamp >= ? AND accuracy IS NOT NULL
                ''', (datetime.now() - timedelta(days=period),))
                accuracies[f'{period}_day'] = cursor.fetchone()[0] or 0.0
        return accuracies

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
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

            data_points = len(df)
            sma_period = min(60, max(5, data_points // 3))
            rsi_period = min(14, max(2, data_points // 10))
            bb_period = min(20, max(5, data_points // 5))
            atr_period = min(14, max(2, data_points // 10))

            df['SMA'] = ta.sma(df['close'], length=sma_period)
            df['EMA'] = ta.ema(df['close'], length=sma_period)
            df['RSI'] = ta.rsi(df['close'], length=rsi_period)

            bollinger = ta.bbands(df['close'], length=bb_period)
            if bollinger is not None and isinstance(bollinger, pd.DataFrame):
                df['BB_Upper'] = bollinger[f'BBU_{bb_period}_2.0']
                df['BB_Middle'] = bollinger[f'BBM_{bb_period}_2.0']
                df['BB_Lower'] = bollinger[f'BBL_{bb_period}_2.0']
            else:
                logger.warning("Bollinger Bands calculation failed")
                df['BB_Upper'] = df['close'] * 1.02
                df['BB_Middle'] = df['close']
                df['BB_Lower'] = df['close'] * 0.98

            df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)

            df = df.dropna()

            logger.info(f"Indicators added successfully. Dataframe shape: {df.shape}")
            logger.info(f"Dataframe columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            logger.error(f"Error in add_indicators: {e}")
            logger.exception("Traceback:")
            return df

    def safe_fetch_multi_timeframe_data(self, ticker="KRW-BTC") -> Dict[str, pd.DataFrame]:
        timeframes = {
            'short': {'interval': 'minute10', 'count': 1440},  # 10분 간격으로 10일치 (144 * 10)
            'medium': {'interval': 'day', 'count': 90},
            'long': {'interval': 'week', 'count': 52}
        }

        data = {}
        for tf, params in timeframes.items():
            try:
                logger.info(f"Fetching {tf} timeframe data")
                df = self.upbit_client.get_ohlcv(ticker, interval=params['interval'], count=params['count'])
                if df is not None and not df.empty:
                    df = self.add_indicators(df)
                    data[tf] = df
                    logger.info(f"Successfully fetched and processed data for {tf} timeframe. Shape: {df.shape}")
                else:
                    logger.warning(f"Empty dataframe for {tf} timeframe")
                    data[tf] = self.create_dummy_data(params['interval'], params['count'])
            except Exception as e:
                logger.error(f"Error fetching data for {tf} timeframe: {e}")
                data[tf] = self.create_dummy_data(params['interval'], params['count'])
        return data

    def create_dummy_data(self, interval: str, count: int) -> pd.DataFrame:
        current_time = pd.Timestamp.now()
        if interval == 'minute60':
            start_time = current_time - pd.Timedelta(hours=count)
            index = pd.date_range(start=start_time, end=current_time, freq='H')
        elif interval == 'day':
            start_time = current_time - pd.Timedelta(days=count)
            index = pd.date_range(start=start_time, end=current_time, freq='D')
        elif interval == 'week':
            start_time = current_time - pd.Timedelta(weeks=count)
            index = pd.date_range(start=start_time, end=current_time, freq='W')

        dummy_data = pd.DataFrame(index=index, columns=['open', 'high', 'low', 'close', 'volume', 'value'])
        dummy_data['close'] = np.random.randint(70000000, 80000000, size=len(dummy_data))
        dummy_data['open'] = dummy_data['close'] + np.random.randint(-1000000, 1000000, size=len(dummy_data))
        dummy_data['high'] = np.maximum(dummy_data['open'], dummy_data['close']) + np.random.randint(0, 500000,
                                                                                                     size=len(
                                                                                                         dummy_data))
        dummy_data['low'] = np.minimum(dummy_data['open'], dummy_data['close']) - np.random.randint(0, 500000, size=len(
            dummy_data))
        dummy_data['volume'] = np.random.randint(100, 1000, size=len(dummy_data))
        dummy_data['value'] = dummy_data['close'] * dummy_data['volume']

        return dummy_data

    def prepare_data_for_ml(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        features = df[
            ['open', 'high', 'low', 'close', 'volume', 'SMA', 'EMA', 'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower',
             'ATR']].values
        labels = np.where(df['close'].shift(-1) > df['close'], 1, 0)[:-1]
        return features[:-1], labels

    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    def calculate_market_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(window=window).std().iloc[-1]
        return volatility * np.sqrt(252)

    def get_fear_greed_index(self) -> Dict[str, Any]:
        try:
            response = requests.get(self.fear_greed_url, timeout=10)
            response.raise_for_status()
            data = response.json()['data'][0]
            return {
                'value': int(data['value']),
                'classification': data['value_classification']
            }
        except Exception as e:
            logger.error(f"Error in fetching Fear & Greed Index: {e}")
            return {'value': None, 'classification': None}

    def get_blockchain_data(self) -> Dict[str, Any]:
        try:
            response = requests.get(self.blockchain_info_url)
            response.raise_for_status()
            data = response.json()
            return {
                'transactions_per_second': data['transactions_per_second'],
                'mempool_size': data['mempool_size'],
                'difficulty': data['difficulty'],
                'hash_rate': data['hash_rate']
            }
        except Exception as e:
            logger.error(f"Error fetching blockchain data: {e}")
            return {
                'transactions_per_second': None,
                'mempool_size': None,
                'difficulty': None,
                'hash_rate': None
            }

    def get_twitter_sentiment(self) -> Dict[str, float]:
        try:
            tweets = self.twitter_api.search_tweets(q="bitcoin", count=100, lang="en")
            sentiments = [TextBlob(tweet.text).sentiment.polarity for tweet in tweets]
            positive_tweets = sum(1 for s in sentiments if s > 0) / len(sentiments)
            negative_tweets = sum(1 for s in sentiments if s < 0) / len(sentiments)
            return {
                'average_sentiment': sum(sentiments) / len(sentiments),
                'positive_tweets': positive_tweets,
                'negative_tweets': negative_tweets
            }
        except Exception as e:
            logger.error(f"Error fetching Twitter sentiment: {e}")
            return {
                'average_sentiment': 0,
                'positive_tweets': 0,
                'negative_tweets': 0
            }

    def fetch_all_data(self) -> Dict[str, Any]:
        return {
            'fear_greed_index': self.get_fear_greed_index(),
            'blockchain_data': self.get_blockchain_data(),
            'twitter_sentiment': self.get_twitter_sentiment()
        }

    def get_recent_data(self, count: int) -> pd.DataFrame:
        try:
            return self.upbit_client.get_ohlcv("KRW-BTC", interval="minute10", count=count)
        except Exception as e:
            logger.error(f"최근 데이터 가져오기 실패: {e}")
            return pd.DataFrame()

    def ensure_sufficient_data(self, required_length: int = 1420) -> pd.DataFrame:
        data = self.get_recent_data(required_length)
        attempts = 0
        while len(data) < required_length and attempts < 5:
            logger.warning(f"데이터 부족: {len(data)} 행. {required_length} 행 필요. 추가 데이터 요청 중...")
            time.sleep(60)  # 1분 대기
            new_data = self.get_recent_data(required_length - len(data))
            data = pd.concat([new_data, data]).drop_duplicates().sort_index()
            attempts += 1

        if len(data) < required_length:
            logger.error(f"충분한 데이터를 가져오지 못했습니다: {len(data)} 행")

        return data.tail(required_length)