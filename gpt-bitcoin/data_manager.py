import logging
import sqlite3
from datetime import timedelta, datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

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
                    value REAL,
                    decision TEXT
                )
            ''')

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data['sma'] = ta.sma(data['close'], length=20)
        data['ema'] = ta.ema(data['close'], length=20)
        data['rsi'] = ta.rsi(data['close'], length=14)
        macd = ta.macd(data['close'])
        data['macd'] = macd['MACD_12_26_9']
        data['signal_line'] = macd['MACDs_12_26_9']
        bb = ta.bbands(data['close'], length=20)
        data['BB_Upper'] = bb['BBU_20_2.0']
        data['BB_Lower'] = bb['BBL_20_2.0']
        return data.ffill().bfill()

    def add_new_data(self, new_data: pd.DataFrame):
        new_data = new_data.reset_index(drop=True)
        new_data = new_data.rename(columns={'index': 'date'})

        if 'date' in new_data.columns:
            new_data['date'] = new_data['date'].apply(
                lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if hasattr(x, 'strftime') else str(x))

        existing_dates = pd.read_sql_query("SELECT date FROM ohlcv_data", self.conn)['date'].tolist()
        new_data = new_data[~new_data['date'].isin(existing_dates)]

        if not new_data.empty:
            try:
                new_data.to_sql('ohlcv_data', self.conn, if_exists='append', index=False)
                logger.info(f"Successfully added {len(new_data)} new data points")
            except sqlite3.IntegrityError as e:
                logger.warning(f"Some data points already exist: {e}")
                for _, row in new_data.iterrows():
                    try:
                        row.to_frame().T.to_sql('ohlcv_data', self.conn, if_exists='append', index=False)
                    except sqlite3.IntegrityError:
                        pass
        else:
            logger.info("No new data to add")

    def get_data_for_training(self, start_time: int, end_time: int) -> pd.DataFrame:
        query = f'''
            SELECT * FROM ohlcv_data
            WHERE date BETWEEN datetime({start_time}, 'unixepoch') AND datetime({end_time}, 'unixepoch')
            ORDER BY date
        '''
        df = pd.read_sql_query(query, self.conn)

        # 'date' 열이 없는 경우 인덱스를 'date' 열로 변환
        if 'date' not in df.columns:
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'date'}, inplace=True)

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df

    def update_data(self):
        latest_data = self.upbit_client.get_ohlcv("KRW-BTC", interval="minute10", count=200)
        if latest_data is not None and not latest_data.empty:
            latest_data.reset_index(inplace=True)
            latest_data.rename(columns={'index': 'date'}, inplace=True)
            self.add_new_data(latest_data)

    def ensure_sufficient_data(self, required_length: int = 10000) -> pd.DataFrame:
        end_time = pd.Timestamp.now()
        start_time = end_time - pd.Timedelta(days=365)  # 1년치 데이터
        data = self.get_data_for_training(int(start_time.timestamp()), int(end_time.timestamp()))

        if len(data) < required_length:
            self.fetch_and_store_historical_data(required_length - len(data))
            data = self.get_data_for_training(int(start_time.timestamp()), int(end_time.timestamp()))
        return data

    def fetch_and_store_historical_data(self, count: int):
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(minutes=count * 10)  # 10분 간격 데이터 가정

        historical_data = self.upbit_client.get_ohlcv("KRW-BTC", interval="minute10", to=end_date, count=count)

        if historical_data is not None and not historical_data.empty:
            historical_data = historical_data.reset_index()
            historical_data = historical_data.rename(columns={'index': 'date'})
            historical_data['date'] = historical_data['date'].astype(str)

            self.add_new_data(historical_data)
            logger.info(f"Added {len(historical_data)} historical data points.")
        else:
            logger.warning("Failed to fetch historical data or no data available.")

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.sort_values('date')
        data = self.calculate_technical_indicators(data)
        return data

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        data['sma'] = ta.sma(data['close'], length=20)
        data['ema'] = ta.ema(data['close'], length=20)
        data['rsi'] = ta.rsi(data['close'], length=14)
        macd = ta.macd(data['close'])
        data['macd'] = macd['MACD_12_26_9']
        data['signal_line'] = macd['MACDs_12_26_9']
        bb = ta.bbands(data['close'], length=20)
        data['BB_Upper'] = bb['BBU_20_2.0']
        data['BB_Lower'] = bb['BBL_20_2.0']
        return data.ffill().bfill()

    def prepare_data_for_ml(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"prepare_data_for_ml 메서드 시작. 데이터 shape: {data.shape}")

        try:
            data = self.add_technical_indicators(data)

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

    def get_recent_data(self, count: int = 100) -> pd.DataFrame:
        query = f'''
            SELECT * FROM ohlcv_data
            ORDER BY date DESC
            LIMIT {count}
        '''
        df = pd.read_sql_query(query, self.conn)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return self.prepare_data(df)

    def get_previous_price(self) -> float:
        query = '''
            SELECT close FROM ohlcv_data
            ORDER BY date DESC
            LIMIT 1
        '''
        result = self.conn.execute(query).fetchone()
        return result[0] if result else None

    def add_decision_column(self):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("PRAGMA table_info(ohlcv_data)")
            columns = [info[1] for info in cursor.fetchall()]

            if 'decision' not in columns:
                cursor.execute("ALTER TABLE ohlcv_data ADD COLUMN decision TEXT")
                logger.info("Added 'decision' column to ohlcv_data table")

    def update_decision(self, date: str, decision: str):
        query = '''
            UPDATE ohlcv_data
            SET decision = ?
            WHERE date = ?
        '''
        with self.conn:
            self.conn.execute(query, (decision, date))

    def get_recent_decisions(self, count: int = 5) -> List[Dict[str, Any]]:
        query = f'''
            SELECT date, decision, close as btc_krw_price
            FROM ohlcv_data
            WHERE decision IS NOT NULL
            ORDER BY date DESC
            LIMIT {count}
        '''
        with self.conn:
            df = pd.read_sql_query(query, self.conn)
        return df.to_dict('records')

    def close(self):
        self.conn.close()

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
                    AND decision IS NOT NULL
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

    def save_decision(self, date: str, decision: str):
        query = '''
            UPDATE ohlcv_data
            SET decision = ?
            WHERE date = ?
        '''
        with self.conn:
            self.conn.execute(query, (decision, date))