import json
import logging
import math
from typing import Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
import torch
import torch.nn as nn
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow import keras
from torch import optim
from xgboost import XGBClassifier
import numpy as np
import tensorflow as tf
from collections import deque
import random

logger = logging.getLogger(__name__)

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
            # 데이터 복사본 생성
            data = data.copy()

            # 데이터가 충분한지 확인
            if len(data) < 33:  # MACD에 필요한 최소 데이터 포인트
                logger.warning("데이터가 충분하지 않습니다. 최소 33개의 데이터 포인트가 필요합니다.")
                return np.array([]), np.array([])

            # 기술적 지표 계산
            data['sma'] = ta.sma(data['close'], length=20)
            data['ema'] = ta.ema(data['close'], length=20)
            data['rsi'] = ta.rsi(data['close'], length=14)

            macd = ta.macd(data['close'])
            if macd is not None:
                data['macd'] = macd['MACD_12_26_9']
                data['signal_line'] = macd['MACDs_12_26_9']
            else:
                logger.warning("MACD 계산 실패. 기본값 0으로 설정합니다.")
                data['macd'] = 0
                data['signal_line'] = 0

            bb = ta.bbands(data['close'], length=20)
            if bb is not None:
                data['BB_Upper'] = bb['BBU_20_2.0']
                data['BB_Lower'] = bb['BBL_20_2.0']
            else:
                logger.warning("Bollinger Bands 계산 실패. 기본값으로 설정합니다.")
                data['BB_Upper'] = data['close']
                data['BB_Lower'] = data['close']

            # NaN 값을 앞뒤 값으로 채우기
            data = data.ffill().bfill()

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
            return np.array([]), np.array([])

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
            # 필요한 기술적 지표 계산
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
        self.is_fitted = False

    def train(self, X, y):
        logger.info(f"XGBoost 훈련 시작. 입력 데이터 shape: X={X.shape}, y={y.shape}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        accuracy = self.model.score(X_test, y_test)
        logger.info(f"XGBoost 훈련 완료. 정확도: {accuracy}")
        return accuracy, 1 - accuracy, {}

    def predict(self, data):
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'train' before using 'predict'.")
        X, _ = self.prepare_data(data)
        if X.size == 0:
            logger.warning("예측을 위한 유효한 데이터가 없습니다.")
            return None
        return self.model.predict(X)[-1]


class RLAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # 리플레이 메모리로 사용
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(24, activation='relu', input_dim=self.state_size),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            next_state_tensor = tf.convert_to_tensor(next_state, dtype=tf.float32)
            next_q_values = self.model.predict(next_state_tensor)
            target = reward + self.gamma * np.amax(next_q_values[0])

        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        target_f = self.model.predict(state_tensor)
        target_f[0][action] = target
        self.model.fit(state_tensor, target_f, epochs=1, verbose=0)

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
        data = data.copy()
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
            elif signal == -1 and btc_amount > 0:  # Sell signal
                balance += btc_amount * current_price * (1 - self.fee_rate)
                btc_amount = 0
                trades.append(('sell', current_price, balance))

        final_balance = balance + btc_amount * data['close'].iloc[-1]
        total_return = (final_balance - self.initial_balance) / self.initial_balance

        logger.info(
            f"Backtest complete. Final balance: {final_balance}, Total return: {total_return}, Number of trades: {len(trades)}")

        return {
            'total_return': total_return,
            'num_trades': len(trades),
            'final_balance': final_balance
        }


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim  # 추가된 부분
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


class LSTMPredictor:
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, factor=0.5)

    def prepare_data(self, data):
        features = ['open', 'high', 'low', 'close', 'volume']
        X = self.scaler.fit_transform(data[features])
        y = self.scaler.fit_transform(data[['close']])

        seq_length = 10
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i + seq_length])
            y_seq.append(y[i + seq_length])

        return np.array(X_seq), np.array(y_seq)

    def train(self, data, epochs=100, batch_size=32):
        try:
            X, y = self.prepare_data(data)
            if X.shape[0] == 0 or y.shape[0] == 0:
                logger.error("준비된 데이터가 비어있습니다.")
                return 0

            input_dim = X.shape[2]
            self.model = LSTMModel(input_dim, self.hidden_dim, self.num_layers, self.output_dim)
            self.model.to(self.device)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                self.scheduler.step(avg_loss)

                if (epoch + 1) % 10 == 0:
                    logger.info(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

            self.model.eval()
            with torch.no_grad():
                X_test = torch.FloatTensor(X_test).to(self.device)
                y_test = torch.FloatTensor(y_test).to(self.device)
                test_outputs = self.model(X_test)
                test_loss = self.criterion(test_outputs, y_test)
            logger.info(f'Test Loss: {test_loss.item():.4f}')

            accuracy = 1.0 - test_loss.item()
            return accuracy

        except Exception as e:
            logger.error(f"LSTM 모델 학습 중 오류 발생: {str(e)}")
            logger.exception("상세 오류:")
            return 0

    def predict(self, data):
        self.model.eval()
        X, _ = self.prepare_data(data)
        X = torch.FloatTensor(X[-1:]).to(self.device)
        with torch.no_grad():
            prediction = self.model(X)
        return self.scaler.inverse_transform(prediction.cpu().numpy())[0, 0]


class ARIMAPredictor:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.model_fit = None
        self.actual_values = None
        self.predictions = None

    def train(self, data):
        if len(data) < 2:
            raise ValueError("데이터가 충분하지 않습니다. ARIMA 모델 학습을 위해서는 최소 2개 이상의 데이터 포인트가 필요합니다.")
        self.model = ARIMA(data['close'], order=self.order)
        self.model_fit = self.model.fit()

    def predict(self, steps=1):
        if self.model_fit is None:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 train 메소드를 호출하세요.")
        self.predictions = self.model_fit.forecast(steps=steps)
        if isinstance(self.predictions, pd.Series):
            return self.predictions.iloc[0]
        elif isinstance(self.predictions, np.ndarray):
            return self.predictions[0]
        return self.predictions

    def get_accuracy(self):
        if self.actual_values is None or self.predictions is None:
            return 0
        mse = np.mean((self.actual_values - self.predictions) ** 2)
        return 1 - (mse / np.var(self.actual_values))


class ProphetPredictor:
    def __init__(self):
        self.model = None
        self.actual_values = None
        self.forecast = None
        self.is_fitted = False

    def train(self, data):
        if len(data) < 2:
            raise ValueError("데이터가 충분하지 않습니다. Prophet 모델 학습을 위해서는 최소 2개 이상의 데이터 포인트가 필요합니다.")
        df = data.reset_index()
        df = df.rename(columns={'date': 'ds', 'close': 'y'})
        self.model = Prophet()
        self.model.fit(df)
        self.is_fitted = True

    def predict(self, periods=1):
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 train 메소드를 호출하세요.")
        future = self.model.make_future_dataframe(periods=periods, freq='10min')
        self.forecast = self.model.predict(future)
        return self.forecast.iloc[-1]['yhat']

    def get_accuracy(self):
        if self.actual_values is None or self.forecast is None:
            return 0
        mape = np.mean(np.abs((self.actual_values - self.forecast['yhat']) / self.actual_values)) * 100
        return 1 - (mape / 100)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*4, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, src):
        if src.dim() == 2:
            src = src.unsqueeze(1)  # Add sequence length dimension
        src = self.input_linear(src)
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        output = self.output_linear(output.mean(dim=1))  # 평균 풀링 사용
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=52500):  # max_len을 증가시킵니다
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerPredictor:
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads):
        self.model = TransformerModel(input_dim, hidden_dim, output_dim, num_layers, num_heads)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        self.predictions = None
        self.actual_values = None

    def train(self, data):
        self.model.train()
        X, y = self.prepare_data(data)
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)

        n_epochs = 100
        batch_size = 32
        for epoch in range(n_epochs):
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i + batch_size]
                batch_y = y[i:i + batch_size]

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')

        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X)
            test_loss = self.criterion(test_outputs, y)
        print(f'Final Loss: {test_loss.item():.4f}')
        return 1 - test_loss.item()

    def prepare_data(self, data):
        features = ['open', 'high', 'low', 'close', 'volume']
        X = data[features].values
        y = data['close'].shift(-1).dropna().values
        X = X[:-1]
        return X, y

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            X = self.prepare_prediction_data(data)
            X = torch.FloatTensor(X).unsqueeze(0)  # (batch_size, sequence_length, features)
            self.predictions = self.model(X)
        return self.predictions.numpy().flatten()[0]

    def prepare_prediction_data(self, data):
        features = ['open', 'high', 'low', 'close', 'volume']
        return data[features].values

    def get_accuracy(self):
        if self.actual_values is None or self.predictions is None:
            return 0
        mse = nn.MSELoss()(torch.FloatTensor(self.actual_values), self.predictions)
        return 1 - mse.item()


class ModelUpdater:
    def __init__(self, data_manager, xgboost_predictor, lstm_predictor, ml_predictor, rl_agent):
        self.data_manager = data_manager
        self.xgboost_predictor = xgboost_predictor
        self.lstm_predictor = lstm_predictor
        self.ml_predictor = ml_predictor
        self.rl_agent = rl_agent
        self.arima_predictor = ARIMAPredictor()
        self.prophet_predictor = ProphetPredictor()
        self.transformer_predictor = TransformerPredictor(input_dim=5, hidden_dim=64, output_dim=1, num_layers=2, num_heads=8)
        self.model_weights_file = 'model_weights.json'
        self.weights = self.load_model_weights()

    def update_xgboost_model(self):
        logger.info("XGBOOST 모델 업데이트 시작")
        data = self.data_manager.ensure_sufficient_data()
        X, y = self.data_manager.prepare_data_for_ml(data)

        param_dist = {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'max_depth': [3, 4, 5, 6, 7],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4]
        }

        xgb = XGBClassifier(random_state=42, enable_categorical=True)

        # XGBoost의 경고 메시지 숨기기
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

        random_search = RandomizedSearchCV(
            xgb, param_distributions=param_dist, n_iter=50,
            cv=TimeSeriesSplit(n_splits=5), scoring='f1', n_jobs=-1, random_state=42, verbose=1
        )
        random_search.fit(X, y)

        # 최적 모델 저장 및 성능 평가
        self.xgboost_predictor.model = random_search.best_estimator_
        y_pred = self.xgboost_predictor.model.predict(X)
        performance = self.evaluate_performance(y, y_pred)

        # 경고 메시지 다시 활성화
        warnings.resetwarnings()

        return random_search.best_params_, performance

    def update_lstm_model(self):
        try:
            logger.info("LSTM 모델 업데이트 시작")
            data = self.data_manager.ensure_sufficient_data()
            accuracy = self.lstm_predictor.train(data)

            performance = {
                "accuracy": accuracy,
                "loss": 1.0 - accuracy
            }

            return {"accuracy": accuracy}, performance
        except Exception as e:
            logger.error(f"LSTM 모델 업데이트 중 오류 발생: {str(e)}")
            logger.exception("상세 오류:")
            return None, None

    def update_ml_model(self):
        logger.info("ML 모델 업데이트 시작")
        data = self.data_manager.ensure_sufficient_data()
        X, y = self.data_manager.prepare_data_for_ml(data)

        from sklearn.ensemble import RandomForestClassifier

        param_dist = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]  # 'auto'를 제거하고 None을 추가
        }

        rf = RandomForestClassifier(random_state=42)
        random_search = RandomizedSearchCV(
            rf, param_distributions=param_dist, n_iter=50,
            cv=TimeSeriesSplit(n_splits=5), scoring='f1', n_jobs=-1, random_state=42, verbose=1
        )
        random_search.fit(X, y)

        # 최적 모델 저장 및 성능 평가
        self.ml_predictor.model = random_search.best_estimator_
        y_pred = self.ml_predictor.model.predict(X)
        performance = self.evaluate_performance(y, y_pred)

        return random_search.best_params_, performance

    def update_rl_model(self):
        logger.info("RL 모델 업데이트 시작")
        try:
            data = self.data_manager.ensure_sufficient_data()
            numeric_data = data.select_dtypes(include=[np.number])

            logger.info(f"Numeric columns: {numeric_data.columns}")
            logger.info(f"Numeric data shape: {numeric_data.shape}")

            env = TradingEnv(numeric_data)

            state_size = len(numeric_data.columns)
            action_size = 3  # Sell, Hold, Buy

            # RLAgent 재초기화
            self.rl_agent = RLAgent(state_size, action_size)

            num_episodes = 5
            max_steps_per_episode = min(100, len(numeric_data) - 1)  # 데이터 길이를 초과하지 않도록 함
            batch_size = 32
            update_frequency = 4

            for episode in range(num_episodes):
                state = env.reset()
                state = np.reshape(state, [1, state_size])
                total_reward = 0
                for step in range(max_steps_per_episode):
                    action = self.rl_agent.act(state)
                    next_state, reward, done, _ = env.step(action)
                    next_state = np.reshape(next_state, [1, state_size])
                    self.rl_agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward

                    if len(self.rl_agent.memory) > batch_size and step % update_frequency == 0:
                        self.rl_agent.replay(batch_size)

                    if done:
                        break

                logger.info(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

            # 성능 평가
            eval_episodes = 5
            total_eval_reward = 0
            for eval_episode in range(eval_episodes):
                logger.info(f"평가 에피소드 {eval_episode + 1}/{eval_episodes} 시작")
                state = env.reset()
                state = np.reshape(state, [1, state_size])
                done = False
                episode_reward = 0
                step = 0
                while not done and step < max_steps_per_episode:
                    action = np.argmax(self.rl_agent.model.predict(state, verbose=0)[0])
                    next_state, reward, done, _ = env.step(action)
                    state = np.reshape(next_state, [1, state_size])
                    episode_reward += reward
                    step += 1
                    if step % 10 == 0:  # 10단계마다 로그 출력
                        logger.info(f"  평가 단계 {step}, 현재 보상: {episode_reward}")
                logger.info(f"평가 에피소드 {eval_episode + 1} 완료, 총 보상: {episode_reward}")
                total_eval_reward += episode_reward

            average_reward = total_eval_reward / eval_episodes
            logger.info(f"RL 모델 업데이트 완료. 평균 평가 보상: {average_reward}")
            return {"average_reward": average_reward}, {"accuracy": average_reward / len(numeric_data)}

        except Exception as e:
            logger.error(f"RL 모델 업데이트 중 오류 발생: {e}")
            logger.exception("상세 오류:")
            return {"average_reward": 0}, {"accuracy": 0}


    def update_arima_model(self):
        logger.info("ARIMA 모델 업데이트 시작")
        data = self.data_manager.ensure_sufficient_data()

        p = d = q = range(0, 3)
        pdq = [(x, y, z) for x in p for y in d for z in q]

        best_aic = float("inf")
        best_order = None

        for param in pdq:
            try:
                model = ARIMA(data['close'], order=param)
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = param
            except:
                continue

        best_model = ARIMA(data['close'], order=best_order)
        self.arima_predictor.model = best_model.fit()

        predictions = self.arima_predictor.model.forecast(steps=len(data))
        performance = self.evaluate_performance(data['close'], predictions)

        return {"best_order": best_order}, performance

    def update_prophet_model(self):
        logger.info("PROPHET 모델 업데이트 시작")
        data = self.data_manager.ensure_sufficient_data()
        df = data.reset_index()
        df = df.rename(columns={'date': 'ds', 'close': 'y'})

        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        }

        best_params = None
        best_score = float('inf')

        for params in ParameterGrid(param_grid):
            m = Prophet(**params)
            m.fit(df)

            df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='90 days')
            df_p = performance_metrics(df_cv)
            score = df_p['mae'].mean()

            if score < best_score:
                best_score = score
                best_params = params

        best_model = Prophet(**best_params)
        best_model.fit(df)
        self.prophet_predictor.model = best_model

        future = best_model.make_future_dataframe(periods=len(data))
        forecast = best_model.predict(future)
        performance = self.evaluate_performance(data['close'], forecast['yhat'][-len(data):])

        return best_params, performance

    def update_transformer_model(self):
        try:
            logger.info("Transformer 모델 업데이트 시작")
            data = self.data_manager.ensure_sufficient_data()

            # 데이터 준비
            features = ['open', 'high', 'low', 'close', 'volume']
            X = data[features].values
            y = data['close'].shift(-1).dropna().values
            X = X[:-1]  # y와 길이 맞추기

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 모델 파라미터 설정
            input_dim = X.shape[1]
            hidden_dim = 64
            num_layers = 2
            num_heads = 8
            output_dim = 1

            # Transformer 모델 초기화
            model = self.transformer_predictor.model
            if model is None:
                model = TransformerModel(input_dim, hidden_dim, output_dim, num_layers, num_heads)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # 학습 루프
            num_epochs = 100
            batch_size = 32
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0
                for i in range(0, len(X_train), batch_size):
                    batch_X = torch.FloatTensor(X_train[i:i + batch_size])
                    batch_y = torch.FloatTensor(y_train[i:i + batch_size]).unsqueeze(1)

                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / (len(X_train) // batch_size)
                if (epoch + 1) % 10 == 0:
                    logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

            # 성능 평가
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test)
                y_pred = model(X_test_tensor).squeeze().numpy()
                mse = ((y_test - y_pred) ** 2).mean()
                rmse = np.sqrt(mse)

            logger.info(f"Transformer 모델 평가 - RMSE: {rmse}")

            # 모델 업데이트
            self.transformer_predictor.model = model

            performance = {
                "rmse": rmse,
                "mse": mse,
                "num_epochs": num_epochs
            }

            return {"message": "Transformer model updated successfully"}, performance

        except Exception as e:
            logger.error(f"Transformer 모델 업데이트 중 오류 발생: {str(e)}")
            logger.exception("상세 오류:")
            return None, None

    def load_model_weights(self):
        try:
            with open(self.model_weights_file, 'r') as f:
                loaded_weights = json.load(f)
                # 모든 모델이 있는지 확인하고, 없으면 기본값 사용
                for model in ['gpt', 'ml', 'xgboost', 'rl', 'lstm', 'arima', 'prophet', 'transformer']:
                    if model not in loaded_weights:
                        loaded_weights[model] = 0.125  # 기본값
            logger.info(f"Loaded weights: {loaded_weights}")
            return loaded_weights
        except FileNotFoundError:
            logger.warning(f"Weights file not found. Using default weights.")
            return self.get_default_weights()
        except json.JSONDecodeError:
            logger.error(f"Error decoding weights file. Using default weights.")
            return self.get_default_weights()

    def get_default_weights(self):
        return {model: 1.0 / 8 for model in ['gpt', 'ml', 'xgboost', 'rl', 'lstm', 'arima', 'prophet', 'transformer']}

    def save_model_weights(self, weights):
        try:
            with open(self.model_weights_file, 'w') as f:
                json.dump(weights, f, indent=2)
            logger.info("Model weights saved successfully.")
        except Exception as e:
            logger.error(f"Error saving model weights: {e}")

    @staticmethod
    def evaluate_performance(y_true, y_pred):
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred)
        }

    def adjust_weights_based_on_performance(self, model_performance):
        # 모든 모델의 정확도 계산
        accuracies = {model: data['accuracy'] for model, data in model_performance.items()}

        # 정확도의 평균과 표준편차 계산
        mean_accuracy = np.mean(list(accuracies.values()))
        std_accuracy = np.std(list(accuracies.values()))

        new_weights = {}
        for model, accuracy in accuracies.items():
            # Z-score를 사용하여 상대적 성능 계산
            z_score = (accuracy - mean_accuracy) / std_accuracy if std_accuracy != 0 else 0

            # 시그모이드 함수를 사용하여 0.5~1.5 범위의 조정 계수 생성
            adjustment_factor = 1 / (1 + np.exp(-z_score))

            # 현재 가중치에 조정 계수를 적용하여 새 가중치 계산
            new_weight = self.weights[model] * (0.9 + 0.2 * adjustment_factor)

            # 최소 가중치 보장 (예: 0.05)
            new_weights[model] = max(new_weight, 0.05)

        # 정규화
        total_weight = sum(new_weights.values())
        self.weights = {model: weight / total_weight for model, weight in new_weights.items()}

        return self.weights


class TradingEnv:
    def __init__(self, data):
        self.data = data
        self.current_step = 0
        self.max_steps = len(data) - 1

    def reset(self):
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps

        if done:
            next_price = self.data['close'].iloc[-1]
        else:
            next_price = self.data['close'].iloc[self.current_step]

        current_price = self.data['close'].iloc[self.current_step - 1]

        if action == 0:  # Sell
            reward = current_price - next_price
        elif action == 2:  # Buy
            reward = next_price - current_price
        else:  # Hold
            reward = 0

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        return self.data.iloc[self.current_step].values

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

