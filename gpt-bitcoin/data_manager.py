import logging
import sqlite3
import time
from datetime import timedelta, datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
import pyupbit
from pytz import timezone

logger = logging.getLogger(__name__)


class DataManager:
    def __init__(self, upbit_client, db_path='crypto_data.db'):
        self.upbit_client = upbit_client
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
        self.korea_tz = timezone('Asia/Seoul')

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

    def check_and_fill_missing_data(self):
        logger.info("Checking for missing data in BTC.DB...")

        # Upbit 거래소 시작일 설정
        upbit_start_date = datetime(2017, 9, 26)

        # 1. DB에서 현재 저장된 데이터의 시작일과 종료일 확인
        query = "SELECT MIN(date), MAX(date) FROM ohlcv_data"
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()

        if result[0] is None or result[1] is None:
            logger.info("Database is empty. Fetching all data from Upbit.")
            self.fetch_all_data()
            return

        db_start_date = max(datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S'), upbit_start_date)
        db_end_date = datetime.strptime(result[1], '%Y-%m-%d %H:%M:%S')

        # 2. Upbit 거래소 시작일부터 현재까지의 모든 날짜 생성
        all_dates = pd.date_range(start=upbit_start_date, end=datetime.now(), freq='10min')

        # 3. DB에 저장된 날짜들 가져오기
        query = "SELECT date FROM ohlcv_data"
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(query)
            db_dates = set([datetime.strptime(date[0], '%Y-%m-%d %H:%M:%S') for date in cursor.fetchall()])

        # 4. 누락된 날짜 찾기 (Upbit 시작일 이후만 고려)
        missing_dates = [date for date in all_dates if date not in db_dates and date >= upbit_start_date]

        if not missing_dates:
            logger.info("No missing data found in BTC.DB")
            return

        logger.info(f"Found {len(missing_dates)} missing data points. Fetching missing data...")

        # 5. 누락된 데이터 가져오기 및 저장
        for start_date in missing_dates:
            end_date = start_date + timedelta(minutes=10)
            df = pyupbit.get_ohlcv("KRW-BTC", interval="minute10", to=end_date, count=1)

            if df is not None and not df.empty:
                self.add_new_data(df)
                logger.info(f"Fetched and saved data for {start_date}")
            else:
                logger.warning(f"Failed to fetch data for {start_date}")

            time.sleep(0.1)  # API 호출 제한 고려

    def add_new_data(self, new_data: pd.DataFrame):
        logger.info(f"Adding new data. Shape: {new_data.shape}")
        logger.info(f"Columns: {new_data.columns.tolist()}")
        logger.info(f"Index: {new_data.index}")

        # date 컬럼의 데이터 타입 확인
        logger.info(f"Date column data type: {new_data['date'].dtype}")
        logger.info(f"Sample date values: {new_data['date'].head().tolist()}")

        # 중복된 날짜 처리: 가장 최근의 데이터만 유지
        new_data = new_data.sort_values('date').drop_duplicates(subset='date', keep='last')

        # date 컬럼이 이미 datetime 형식이므로 바로 문자열로 변환
        new_data['date'] = new_data['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

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

        logger.info(f"Final new data shape after processing: {new_data.shape}")
        logger.info(f"Final date range: {new_data['date'].min()} to {new_data['date'].max()}")

    def fetch_all_data(self):
        end_date = datetime.now()
        start_date = datetime(2017, 9, 26)  # Upbit 거래소 시작일

        while start_date < end_date:
            df = pyupbit.get_ohlcv("KRW-BTC", interval="minute10", to=end_date, count=200)
            if df is not None and not df.empty:
                self.add_new_data(df)
                logger.info(f"Fetched and saved data from {df.index[0]} to {df.index[-1]}")
                end_date = df.index[0] - timedelta(minutes=1)
            else:
                logger.warning(f"Failed to fetch data for period ending at {end_date}")
            time.sleep(0.1)


    def get_data_for_training(self, start_time: int, end_time: int) -> pd.DataFrame:
        query = f'''
            SELECT * FROM ohlcv_data
            WHERE date BETWEEN datetime({start_time}, 'unixepoch') AND datetime({end_time}, 'unixepoch')
            ORDER BY date
        '''
        df = pd.read_sql_query(query, self.conn)
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
            self.update_db_with_new_data()
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

    def update_db_with_new_data(self):
        try:
            logger.info("데이터베이스 업데이트 시작")

            # 현재 데이터베이스의 최신 날짜 확인
            query = "SELECT MAX(date) FROM ohlcv_data"
            with self.conn:
                cursor = self.conn.cursor()
                cursor.execute(query)
                last_date_in_db = cursor.fetchone()[0]

            if last_date_in_db:
                last_date_in_db = pd.to_datetime(last_date_in_db)
                logger.info(f"현재 데이터베이스의 최신 날짜: {last_date_in_db}")
            else:
                logger.info("데이터베이스가 비어있습니다.")
                last_date_in_db = pd.Timestamp.now() - pd.Timedelta(days=30)

            # 최신 데이터 가져오기
            latest_data = self.upbit_client.get_ohlcv("KRW-BTC", interval="minute10", count=200)

            if latest_data is not None and not latest_data.empty:
                latest_data.reset_index(inplace=True)
                latest_data.rename(columns={'index': 'date'}, inplace=True)

                # 새로운 데이터의 날짜 범위 확인
                new_data_start = latest_data['date'].min()
                new_data_end = latest_data['date'].max()
                logger.info(f"새로운 데이터 범위: {new_data_start} ~ {new_data_end}")

                # 중복 제거 및 새 데이터만 필터링
                new_data = latest_data[latest_data['date'] > last_date_in_db]

                if not new_data.empty:
                    self.add_new_data(new_data)
                    logger.info(f"새로운 데이터 {len(new_data)}개 추가됨")
                    logger.info(f"추가된 최신 데이터의 날짜: {new_data['date'].max()}")
                else:
                    logger.info("추가할 새로운 데이터가 없습니다.")
            else:
                logger.warning("최신 데이터를 가져오는 데 실패했거나 데이터가 비어있습니다.")

            logger.info("데이터베이스 업데이트 완료")

        except Exception as e:
            logger.error(f"데이터베이스 업데이트 중 오류 발생: {e}")
            logger.exception("상세 오류:")

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