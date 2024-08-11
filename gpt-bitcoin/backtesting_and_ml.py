import logging
import pandas_ta as ta
from sklearn.metrics import accuracy_score
from tensorflow import keras
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from typing import Tuple


class MLPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_selector = VarianceThreshold(threshold=0.0)
        self.last_accuracy = 0
        self.last_loss = 0
        self.features = ['open', 'high', 'low', 'close', 'volume', 'sma', 'ema', 'rsi', 'macd', 'signal_line',
                         'BB_Upper', 'BB_Lower']

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"prepare_data 메서드 시작. 데이터 shape: {data.shape}")

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
            logger.info(f"데이터 샘플:\n{data[self.features].head()}")

            if data.empty:
                logger.error("모든 행이 NaN 값으로 제거되었습니다.")
                return np.array([]), np.array([])

            X = data[self.features].values
            y = (data['close'].shift(-1) > data['close']).astype(int).values[:-1]
            X = X[:-1]  # 마지막 행 제거 (타겟 변수와 길이 맞추기)

            logger.info(f"최종 X shape: {X.shape}, y shape: {y.shape}")

            return X, y

        except Exception as e:
            logger.error(f"데이터 준비 중 오류 발생: {e}")
            logger.exception("상세 오류:")
            raise

    def train(self, data_manager):
        data = data_manager.ensure_sufficient_data(1440)  # 24시간 데이터
        X, y = self.prepare_data(data)

        if not self.validate_data(pd.DataFrame(X, columns=self.features)):
            raise ValueError("유효하지 않은 데이터 형식")

        if X.size == 0 or y.size == 0:
            logger.error("유효한 데이터가 없어 학습을 진행할 수 없습니다.")
            return 0, 1, {}

        # NaN 값 처리
        self.imputer.fit(X)
        X = self.imputer.transform(X)

        # 스케일링
        X_scaled = self.scaler.fit_transform(X)

        # 분산이 0인 특성 제거
        self.feature_selector.fit(X_scaled)
        X_scaled = self.feature_selector.transform(X_scaled)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        accuracy = self.model.score(X_test, y_test)
        self.last_accuracy = accuracy
        self.last_loss = 1 - accuracy

        logger.info(f"모델 학습 완료. 정확도: {accuracy}, 손실: {self.last_loss}")

        return accuracy, self.last_loss, {"loss": self.last_loss}

    def predict(self, data: pd.DataFrame) -> int:
        logger.info(f"predict 메서드 시작. 입력 데이터 shape: {data.shape}")

        try:
            X = data[self.features].astype(float).values[-1].reshape(1, -1)

            logger.info(f"선택된 특성 데이터 shape: {X.shape}")
            logger.info(f"선택된 특성 데이터:\n{X}")

            # NaN 값 처리
            X = self.imputer.transform(X)

            logger.info(f"NaN 처리 후 데이터:\n{X}")

            # 분산이 0인 특성 제거
            X = self.feature_selector.transform(X)

            logger.info(f"특성 선택 후 데이터 shape: {X.shape}")

            # 스케일링
            X_scaled = self.scaler.transform(X)

            logger.info(f"스케일링 후 데이터:\n{X_scaled}")

            prediction = self.model.predict(X_scaled)

            logger.info(f"예측 결과: {prediction[0]}")

            return prediction[0]

        except Exception as e:
            logger.error(f"예측 중 오류 발생: {e}")
            logger.exception("상세 오류:")
            return 0  # 오류 발생 시 기본값 반환

    def get_accuracy(self):
        return self.last_accuracy

    def get_loss(self):
        return self.last_loss

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # 무한대 값을 NaN으로 변환
        data = data.replace([np.inf, -np.inf], np.nan)

        # NaN 값이 있는 행 제거
        data = data.dropna()

        return data

    def validate_data(self, data: pd.DataFrame) -> bool:
        if not all(col in data.columns for col in self.features):
            missing_columns = set(self.features) - set(data.columns)
            logger.error(f"Missing columns in data: {missing_columns}")
            return False
        return True

    def calculate_accuracy(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred)


class XGBoostPredictor(MLPredictor):
    def __init__(self):
        super().__init__()
        self.model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        accuracy = self.model.score(X_test, y_test)
        return accuracy, 1 - accuracy, {}

    def predict(self, data):
        print('predict')
        print(data)
        print(not hasattr(self.model, 'fitted_'))
        print(not self.model.fit)
        if not hasattr(self.model, 'fit') or not self.model.fit:
            raise ValueError("Model is not fitted yet. Call 'train' before using 'predict'.")
        X, _ = self.prepare_data(data)
        return self.model.predict(X)[-1]


class RLAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = (reward + self.gamma *
                      np.amax(self.model.predict(next_state)[0]))
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_average_reward(self):
        if not self.memory:
            return 0
        return np.mean([m[2] for m in self.memory])


class Backtester:
    def __init__(self, data_manager, initial_balance: float, fee_rate: float):
        self.data_manager = data_manager
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate

    def run_backtest(self, strategy, start_time: pd.Timestamp, end_time: pd.Timestamp, historical_data: pd.DataFrame):
        data = historical_data[(historical_data.index >= start_time) & (historical_data.index <= end_time)]
        data['date'] = data.index.strftime('%Y-%m-%d %H:%M:%S')

        if data.empty:
            logger.warning("No data available for the specified time range.")
            return {'total_return': 0, 'num_trades': 0, 'final_balance': self.initial_balance}

        balance = self.initial_balance
        btc_amount = 0
        trades = []

        logger.info(f"Starting backtest from {start_time} to {end_time} with initial balance {self.initial_balance}.")

        for i in range(1, len(data)):
            signal = strategy(data.iloc[:i])
            current_price = data['close'].iloc[i]

            if signal == 1 and balance > 0:  # Buy signal
                btc_to_buy = balance / current_price * (1 - self.fee_rate)
                btc_amount += btc_to_buy
                balance = 0
                trades.append(('buy', current_price, btc_to_buy))
                # logger.info(f"Buy executed at {current_price}, BTC bought: {btc_to_buy}, New balance: {balance}")
            elif signal == -1 and btc_amount > 0:  # Sell signal
                balance += btc_amount * current_price * (1 - self.fee_rate)
                btc_amount = 0
                trades.append(('sell', current_price, balance))
                # logger.info(f"Sell executed at {current_price}, New balance: {balance}, BTC amount: {btc_amount}")

        final_balance = balance + btc_amount * data['close'].iloc[-1]
        total_return = (final_balance - self.initial_balance) / self.initial_balance

        logger.info(
            f"Backtest complete. Final balance: {final_balance}, Total return: {total_return}, Number of trades: {len(trades)}")

        return {
            'total_return': total_return,
            'num_trades': len(trades),
            'final_balance': final_balance
        }

    def calculate_sharpe_ratio(self, trades, data):
        if len(trades) < 2:
            return 0

        returns = []
        for i in range(1, len(trades)):
            if trades[i][0] == 'sell' and trades[i - 1][0] == 'buy':
                returns.append((trades[i][1] - trades[i - 1][1]) / trades[i - 1][1])

        if not returns:
            return 0

        return np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0

    def calculate_max_drawdown(self, trades, data):
        if len(trades) < 2:
            return 0

        peak = 0
        max_drawdown = 0

        for trade in trades:
            if trade[0] == 'buy':
                peak = max(peak, trade[1])
            elif trade[0] == 'sell':
                drawdown = (peak - trade[1]) / peak
                max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

def simple_moving_average_strategy(data: pd.DataFrame) -> int:
    if len(data) < 20:
        return 0

    sma_short = data['close'].rolling(window=5).mean().iloc[-1]
    sma_long = data['close'].rolling(window=20).mean().iloc[-1]

    if sma_short > sma_long:
        return 1  # Buy signal
    elif sma_short < sma_long:
        return -1  # Sell signal
    else:
        return 0  # Hold


def run_backtest(data_manager, historical_data):
    backtester = Backtester(data_manager, initial_balance=10_000_000, fee_rate=0.0005)
    end_time = pd.Timestamp.now()
    start_time = end_time - pd.Timedelta(days=30)
    historical_data.reset_index(inplace=True)
    historical_data['date'] = pd.to_datetime(historical_data['date'])
    historical_data.set_index('date', inplace=True)

    results = backtester.run_backtest(simple_moving_average_strategy, start_time, end_time, historical_data)

    return results


def tune_hyperparameters(self, X, y):
    param_dist = {
        'n_estimators': [10, 50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7]
    }
    random_search = RandomizedSearchCV(self.model, param_distributions=param_dist, n_iter=10, cv=3, random_state=42)
    random_search.fit(X, y)
    self.model = random_search.best_estimator_
    return random_search.best_params_
