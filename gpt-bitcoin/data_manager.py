import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas_ta as ta
import pandas as pd
from numpy import dtype

logger = logging.getLogger(__name__)


class DataManager:
    def __init__(self, upbit_client, db_path='crypto_data.db'):
        self.upbit_client = upbit_client
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    date TEXT PRIMARY KEY,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    value REAL
                )
            ''')

    def add_new_data(self, new_data: pd.DataFrame):
        logger.info(f"Adding new data. Shape: {new_data.shape}")

        # 'level_0' 열이 있다면 제거
        if 'level_0' in new_data.columns:
            new_data = new_data.drop('level_0', axis=1)

        # 인덱스를 리셋하고 'index' 열 이름 변경
        new_data = new_data.reset_index(drop=True)
        new_data = new_data.rename(columns={'index': 'date'})

        # 'date' 열을 문자열로 변환 (이미 문자열인 경우 처리)
        if 'date' in new_data.columns:
            new_data['date'] = new_data['date'].apply(
                lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if hasattr(x, 'strftime') else str(x))
        else:
            logger.warning("'date' column not found in new_data")

        # 기존 데이터베이스에 있는 date를 조회
        existing_dates = pd.read_sql_query("SELECT date FROM ohlcv_data", self.conn)['date'].tolist()

        # 중복 제거
        new_data = new_data[~new_data['date'].isin(existing_dates)]

        if not new_data.empty:
            try:
                new_data.to_sql('ohlcv_data', self.conn, if_exists='append', index=False)
                logger.info(f"Successfully added {len(new_data)} new data points")
            except sqlite3.IntegrityError as e:
                logger.warning(f"Some data points already exist: {e}")
                # 개별적으로 데이터 삽입 시도
                for _, row in new_data.iterrows():
                    try:
                        row.to_frame().T.to_sql('ohlcv_data', self.conn, if_exists='append', index=False)
                    except sqlite3.IntegrityError:
                        logger.debug(f"Skipping duplicate data point at date {row['date']}")
        else:
            logger.info("No new data to add")

    def add_technical_indicators(self, data: pd.DataFrame):
        indicators = pd.DataFrame(index=data.index)
        indicators['sma'] = self.calculate_sma(data)
        indicators['ema'] = self.calculate_ema(data)
        indicators['rsi'] = self.calculate_rsi(data)
        indicators['macd'], indicators['signal_line'] = self.calculate_macd(data)
        indicators['BB_Upper'], indicators['BB_Lower'] = self.calculate_bollinger_bands(data)

        # NaN 값을 앞뒤 값으로 채우기
        indicators = indicators.ffill().bfill()

        # 기존 데이터와 새로 계산한 지표를 병합
        result = pd.concat([data, indicators], axis=1)
        return result

    def get_data_for_training(self, start_time: int, end_time: int) -> pd.DataFrame:
        query = f'''
            SELECT * FROM ohlcv_data
            WHERE date BETWEEN datetime({start_time}, 'unixepoch') AND datetime({end_time}, 'unixepoch')
            ORDER BY date
        '''
        df = pd.read_sql_query(query, self.conn)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def update_data(self):
        latest_data = self.upbit_client.get_ohlcv("KRW-BTC", interval="minute10", count=200)
        if latest_data is not None and not latest_data.empty:
            latest_data.reset_index(inplace=True)
            latest_data.rename(columns={'index': 'date'}, inplace=True)
            self.add_new_data(latest_data)

    def calculate_sma(self, data, period=20):
        return data['close'].rolling(window=period).mean()

    def calculate_ema(self, data, period=20):
        return data['close'].ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, data, period=14):
        close_data = data['close']
        delta = close_data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        close_data = data['close']
        fast_ema = close_data.ewm(span=fast_period, adjust=False).mean()
        slow_ema = close_data.ewm(span=slow_period, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal_line

    def calculate_bollinger_bands(self, data, period=20, num_std=2):
        close_data = data['close']
        sma = close_data.rolling(window=period).mean()
        std = close_data.rolling(window=period).std()
        upper_bb = sma + (std * num_std)
        lower_bb = sma - (std * num_std)
        return upper_bb, lower_bb

    def ensure_sufficient_data(self, required_length: int = 1440) -> pd.DataFrame:
        end_time = pd.Timestamp.now()
        start_time = end_time - pd.Timedelta(minutes=required_length * 10)
        data = self.get_data_for_training(int(start_time.timestamp()), int(end_time.timestamp()))
        print('ensure_sufficient_data')
        print(data)
        print(data.columns)
        if len(data) < required_length:
            self.fetch_and_store_historical_data(required_length - len(data))
            data = self.get_data_for_training(int(start_time.timestamp()), int(end_time.timestamp()))

        return data

    def fetch_and_store_historical_data(self, count: int):
        historical_data = self.upbit_client.get_ohlcv("KRW-BTC", interval="minute10", count=count)
        if historical_data is not None and not historical_data.empty:
            historical_data.reset_index(inplace=True)
            historical_data.rename(columns={'index': 'date'}, inplace=True)
            self.add_new_data(historical_data)

    def get_average_accuracy(self, days: int = 7) -> float:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # datetime 객체를 문자열로 변환
        end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
        start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')

        query = f'''
            SELECT AVG(accuracy)
            FROM (
                SELECT 
                    CASE 
                        WHEN decision = 'buy' AND lead_price > price THEN 1
                        WHEN decision = 'sell' AND lead_price < price THEN 1
                        ELSE 0
                    END as accuracy
                FROM (
                    SELECT 
                        date, 
                        decision, 
                        close as price,
                        LEAD(close) OVER (ORDER BY date) as lead_price
                    FROM ohlcv_data
                    WHERE date BETWEEN '{start_time_str}' AND '{end_time_str}'
                )
            )
        '''
        try:
            with self.conn:
                cursor = self.conn.cursor()
                cursor.execute(query)
                result = cursor.fetchone()[0]
            return result if result is not None else 0.0
        except Exception as e:
            logger.error(f"Error in get_average_accuracy: {e}")
            logger.error(f"Query: {query}")
            return 0.0

    def get_recent_decisions(self, n: int = 5) -> List[Dict[str, Any]]:
        query = f"""
        SELECT * FROM decisions
        ORDER BY timestamp DESC
        LIMIT {n}
        """
        with self.conn:
            df = pd.read_sql_query(query, self.conn)
        return df.to_dict('records')

    def fetch_extended_historical_data(self, days: int = 365):
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=days)
        historical_data = self.upbit_client.get_ohlcv("KRW-BTC", interval="day", to=end_date, count=days)
        if historical_data is not None and not historical_data.empty:
            # 인덱스를 'date' 열로 변환
            historical_data = historical_data.reset_index()
            historical_data = historical_data.rename(columns={'index': 'date'})
            historical_data['date'] = historical_data['date'].astype(str)
            self.add_new_data(historical_data)
            return True
        return False

    def add_decision_column(self):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("PRAGMA table_info(ohlcv_data)")
            columns = [info[1] for info in cursor.fetchall()]

            if 'decision' not in columns:
                cursor.execute("ALTER TABLE ohlcv_data ADD COLUMN decision TEXT")
                logger.info("Added 'decision' column to ohlcv_data table")

    def prepare_data_for_ml(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"prepare_data_for_ml 메서드 시작. 데이터 shape: {data.shape}")

        try:
            # 기술적 지표 계산
            data['sma'] = ta.sma(data['close'], length=20)
            data['ema'] = ta.ema(data['close'], length=20)
            data['rsi'] = ta.rsi(data['close'], length=14)
            macd = ta.macd(data['close'])
            data['macd'] = macd['MACD_12_26_9']
            data['signal_line'] = macd['MACDs_12_26_9']
            bb = ta.bbands(data['close'], length=20)
            data['BB_Upper'] = bb['BBU_20_2.0']
            data['BB_Lower'] = bb['BBL_20_2.0']

            # NaN 값을 앞뒤 값으로 채우기
            data = data.ffill().bfill()

            logger.info(f"기술적 지표 추가 후 데이터 shape: {data.shape}")
            logger.info(f"데이터 컬럼: {data.columns}")

            features = ['open', 'high', 'low', 'close', 'volume', 'sma', 'ema', 'rsi', 'macd', 'signal_line',
                        'BB_Upper', 'BB_Lower']
            X = data[features].values
            y = (data['close'].shift(-1) > data['close']).astype(int).values[:-1]
            X = X[:-1]  # 마지막 행 제거 (타겟 변수와 길이 맞추기)

            logger.info(f"최종 X shape: {X.shape}, y shape: {y.shape}")

            return X, y

        except Exception as e:
            logger.error(f"데이터 준비 중 오류 발생: {e}")
            logger.exception("상세 오류:")
            raise

    def check_table_structure(self):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("PRAGMA table_info(ohlcv_data)")
            ohlcv_columns = [info[1] for info in cursor.fetchall()]
            cursor.execute("PRAGMA table_info(technical_indicators)")
            indicator_columns = [info[1] for info in cursor.fetchall()]

        logger.info(f"ohlcv_data columns: {ohlcv_columns}")
        logger.info(f"technical_indicators columns: {indicator_columns}")

    def debug_database(self):
        with self.conn:
            cursor = self.conn.cursor()

            # 테이블 목록 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            logger.info(f"Tables in database: {tables}")

            # ohlcv_data 테이블 구조 확인
            cursor.execute("PRAGMA table_info(ohlcv_data)")
            columns = cursor.fetchall()
            logger.info("ohlcv_data table structure:")
            for column in columns:
                logger.info(f"  {column}")

            # 샘플 데이터 확인
            cursor.execute("SELECT * FROM ohlcv_data LIMIT 5")
            samples = cursor.fetchall()
            logger.info("Sample data from ohlcv_data:")
            for sample in samples:
                logger.info(f"  {sample}")

    def get_recent_trades(self, days=30):
        """최근 거래 데이터를 가져옵니다."""
        end_time = int(time.time())
        start_time = end_time - (days * 24 * 60 * 60)  # days일 전의 timestamp

        query = f"""
        SELECT * FROM ohlcv_data
        WHERE date BETWEEN datetime({start_time}, 'unixepoch') AND datetime({end_time}, 'unixepoch')
        ORDER BY date
        """

        df = pd.read_sql_query(query, self.conn)
        df['date'] = pd.to_datetime(df['date'])
        return df
