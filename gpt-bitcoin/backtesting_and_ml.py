import json
import logging
import os
import random
from collections import deque
from typing import Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
import torch
import torch.nn as nn
import torch.optim as optim
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

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
        self.is_fitted = False  # 새로 추가된 속성

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"prepare_data 메서드 시작. 데이터 shape: {data.shape}")

        try:
            data = data.copy()

            if len(data) < 33:
                logger.warning("데이터가 충분하지 않습니다. 최소 33개의 데이터 포인트가 필요합니다.")
                return np.array([]), np.array([])

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

            X = data[self.features].values
            y = (data['close'].shift(-1) > data['close']).astype(int).values[:-1]
            X = X[:-1]  # 마지막 행 제거 (타겟 변수와 길이 맞추기)

            logger.info(f"최종 X shape: {X.shape}, y shape: {y.shape}")

            return X, y

        except Exception as e:
            logger.error(f"데이터 준비 중 오류 발생: {e}")
            logger.exception("상세 오류:")
            return np.array([]), np.array([])

    def train(self, X, y):
        if X.size == 0 or y.size == 0:
            logger.error("유효한 데이터가 없어 학습을 진행할 수 없습니다.")
            return 0, 1, {}

        X = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = self.feature_selector.fit_transform(X_scaled)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        accuracy = self.model.score(X_test, y_test)
        self.last_accuracy = accuracy
        self.last_loss = 1 - accuracy

        logger.info(f"모델 학습 완료. 정확도: {accuracy}, 손실: {self.last_loss}")

        return accuracy, self.last_loss, {"loss": self.last_loss}

    def predict(self, data: pd.DataFrame) -> float:
        logger.info(f"predict 메서드 시작. 입력 데이터 shape: {data.shape}")

        try:
            # 기술적 지표 계산
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

            # NaN 값을 앞뒤 값으로 채우기
            data = data.ffill().bfill()

            X = data[self.features].iloc[-1].values.reshape(1, -1)

            X = self.imputer.transform(X)
            X = self.feature_selector.transform(X)
            X_scaled = self.scaler.transform(X)
            prediction = self.model.predict(X_scaled)

            return prediction[0]

        except Exception as e:
            logger.error(f"예측 중 오류 발생: {e}")
            logger.exception("상세 오류:")
            return 0  # 오류 발생 시 기본값 반환

    def prepare_single_prediction_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data['sma'] = ta.sma(data['close'], length=20)
        data['ema'] = ta.ema(data['close'], length=20)
        data['rsi'] = ta.rsi(data['close'], length=14)
        macd = ta.macd(data['close'])
        data['macd'] = macd['MACD_12_26_9'] if macd is not None else 0
        data['signal_line'] = macd['MACDs_12_26_9'] if macd is not None else 0
        bb = ta.bbands(data['close'], length=20)
        data['BB_Upper'] = bb['BBU_20_2.0'] if bb is not None else data['close']
        data['BB_Lower'] = bb['BBL_20_2.0'] if bb is not None else data['close']
        return data.ffill().bfill()

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


class XGBoostPredictor(MLPredictor):
    def __init__(self):
        super().__init__()
        self.model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        self.is_fitted = False

    def train(self, X, y):
        logger.info(f"XGBoost 훈련 시작. 입력 데이터 shape: X={X.shape}, y={y.shape}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit the imputer
        self.imputer.fit(X_train)
        X_train_imputed = self.imputer.transform(X_train)
        X_test_imputed = self.imputer.transform(X_test)

        # Fit the scaler
        self.scaler.fit(X_train_imputed)
        X_train_scaled = self.scaler.transform(X_train_imputed)
        X_test_scaled = self.scaler.transform(X_test_imputed)

        # Fit the feature selector
        self.feature_selector.fit(X_train_scaled)
        X_train_selected = self.feature_selector.transform(X_train_scaled)
        X_test_selected = self.feature_selector.transform(X_test_scaled)

        self.model.fit(X_train_selected, y_train)
        self.is_fitted = True
        accuracy = self.model.score(X_test_selected, y_test)
        logger.info(f"XGBoost 훈련 완료. 정확도: {accuracy}")
        return accuracy, 1 - accuracy, {}

    def predict(self, data: pd.DataFrame) -> float:
        logger.info(f"XGBoost predict 메서드 시작. 입력 데이터 shape: {data.shape}")

        try:
            X = data[self.features].iloc[-1].values.reshape(1, -1)

            X = self.imputer.transform(X)
            X = self.scaler.transform(X)
            X = self.feature_selector.transform(X)
            prediction_prob = self.model.predict_proba(X)

            # 예측 확률을 반환 (상승 확률)
            return prediction_prob[0][1]

        except Exception as e:
            logger.error(f"XGBoost 예측 중 오류 발생: {e}")
            logger.exception("상세 오류:")
            return 0.5  # 오류 발생 시 중립적인 값 반환


class RLAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state_tensor)
        return np.argmax(act_values.detach().numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model(next_state_tensor).detach().numpy())
            target_f = self.model(state_tensor)
            target_f[0][action] = target
            loss = nn.MSELoss()(self.model(state_tensor), target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def predict_price(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state_tensor)
        action = np.argmax(act_values.detach().numpy())
        if action == 0:  # Sell
            return state[-1] * 0.99
        elif action == 2:  # Buy
            return state[-1] * 1.01
        else:  # Hold
            return state[-1]

    def train(self, data):
        for i in range(1, len(data)):
            state = self.prepare_state(data.iloc[i - 1])
            action = self.act(state)
            next_state = self.prepare_state(data.iloc[i])
            reward = self.calculate_reward(data.iloc[i - 1], data.iloc[i], action)
            done = (i == len(data) - 1)
            self.remember(state, action, reward, next_state, done)
            if len(self.memory) > 32:
                self.replay(32)

        self.update_target_model()

    def prepare_state(self, row):
        return np.array([
            row['open'], row['high'], row['low'], row['close'], row['volume']
        ])

    def calculate_reward(self, prev_row, curr_row, action):
        price_change = (curr_row['close'] - prev_row['close']) / prev_row['close']
        if action == 0:  # Sell
            return -price_change
        elif action == 2:  # Buy
            return price_change
        else:  # Hold
            return 0


class Backtester:
    def __init__(self, data_manager, initial_balance: float = 500000, fee_rate: float = 0.0005):
        self.data_manager = data_manager
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate

    def run_backtest(self, strategy, start_time: pd.Timestamp, end_time: pd.Timestamp, historical_data: pd.DataFrame):
        balance = self.initial_balance
        btc_amount = 0
        trades = []

        for i in range(1, len(historical_data)):
            current_data = historical_data.iloc[:i]
            signal = strategy(current_data)
            current_price = historical_data['close'].iloc[i]

            if signal > 0 and balance > 0:  # Buy signal
                btc_to_buy = (balance / current_price) * (1 - self.fee_rate)
                btc_amount += btc_to_buy
                balance = 0
                trades.append(('buy', current_price, btc_to_buy))
            elif signal < 0 and btc_amount > 0:  # Sell signal
                balance += (btc_amount * current_price) * (1 - self.fee_rate)
                btc_amount = 0
                trades.append(('sell', current_price, balance))

        final_balance = balance + btc_amount * historical_data['close'].iloc[-1]
        total_return = (final_balance - self.initial_balance) / self.initial_balance
        sharpe_ratio = self.calculate_sharpe_ratio(trades, historical_data)

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': len(trades),
            'final_balance': final_balance
        }

    def calculate_sharpe_ratio(self, trades, historical_data):
        returns = []
        for i in range(1, len(trades)):
            if trades[i][0] == 'sell' and trades[i - 1][0] == 'buy':
                returns.append((trades[i][1] - trades[i - 1][1]) / trades[i - 1][1])
        if not returns:
            return 0
        return np.sqrt(252) * np.mean(returns) / np.std(returns)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class LSTMPredictor:
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=1):
        self.model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def prepare_data(self, data):
        features = ['open', 'high', 'low', 'close', 'volume']
        scaled_data = self.scaler.fit_transform(data[features])
        X, y = [], []
        for i in range(len(scaled_data) - 30):
            X.append(scaled_data[i:i + 30])
            y.append(scaled_data[i + 30, 3])  # predict close price
        return np.array(X), np.array(y)

    def train(self, data, epochs=50):
        X, y = self.prepare_data(data)
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).unsqueeze(1)

        for epoch in range(epochs):
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        return self.evaluate(X, y)

    def evaluate(self, X, y):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
        return 1 - loss.item()  # return accuracy-like metric

    def predict(self, data):
        self.model.eval()
        X, _ = self.prepare_data(data.tail(31))  # 31 = seq_length + 1
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X[-1]).unsqueeze(0)
            predicted = self.model(X_tensor).numpy()
        last_original = data['close'].iloc[-1]

        # 스케일러 역변환 수정
        predicted_scaled = np.zeros((1, 5))  # 5는 ['open', 'high', 'low', 'close', 'volume']의 개수
        predicted_scaled[0, 3] = predicted[0, 0]  # 'close' 값만 예측값으로 설정
        predicted_unscaled = self.scaler.inverse_transform(predicted_scaled)
        predicted_change = predicted_unscaled[0, 3]

        return last_original * (1 + predicted_change)


class ARIMAPredictor:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.model_fit = None

    def train(self, data):
        if len(data) < 2:
            raise ValueError("데이터가 충분하지 않습니다. ARIMA 모델 학습을 위해서는 최소 2개 이상의 데이터 포인트가 필요합니다.")

        try:
            # 인덱스를 정수로 변경
            data = data.reset_index(drop=True)
            self.model = ARIMA(data['close'], order=self.order)
            self.model_fit = self.model.fit()
            logger.info("ARIMA model fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            raise

    def predict(self, steps=1):
        if self.model_fit is None:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 train 메소드를 호출하세요.")
        forecast = self.model_fit.forecast(steps=steps)

        # forecast가 Series인 경우와 ndarray인 경우를 모두 처리
        if isinstance(forecast, pd.Series):
            return forecast.iloc[0]
        elif isinstance(forecast, np.ndarray):
            return forecast[0]
        else:
            return forecast  # 다른 형태의 경우 그대로 반환

    def get_accuracy(self):
        if self.model_fit is None:
            return 0
        return 1 - self.model_fit.mse  # Simple accuracy metric based on MSE


class ProphetPredictor:
    def __init__(self):
        self.model = None
        self.last_train_date = None

    def train(self, data):
        df = data.reset_index()
        df = df.rename(columns={'date': 'ds', 'close': 'y'})
        self.model = Prophet(daily_seasonality=True, weekly_seasonality=True)
        self.model.fit(df)
        self.last_train_date = df['ds'].max()

    def predict(self, future_periods=1):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        future_dates = pd.date_range(start=self.last_train_date, periods=future_periods + 1, freq='D')[1:]
        future = pd.DataFrame({'ds': future_dates})
        forecast = self.model.predict(future)
        return forecast['yhat'].values[-1]

    def get_accuracy(self):
        if self.model is None:
            return 0
        # Use cross-validation to get an accuracy metric
        df_cv = self.model.cross_validation(initial='730 days', period='180 days', horizon='30 days')
        df_p = self.model.performance_metrics(df_cv)
        return 1 - df_p['mape'].mean()  # Using 1 - MAPE as an accuracy metric

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.embedding(src)
        output = self.transformer_encoder(src)
        return self.decoder(output[:, -1, :])  # 마지막 시퀀스의 출력만 사용


class TransformerPredictor:
    def __init__(self, input_dim=5, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, seq_length=10):
        self.model = TransformerModel(input_dim, d_model, nhead, num_layers, dim_feedforward)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.seq_length = seq_length  # 시퀀스 길이 추가

    def prepare_data(self, data):
        logger.info(f"Transformer prepare_data 메서드 시작. 입력 데이터 shape: {data.shape}")

        features = ['open', 'high', 'low', 'close', 'volume']

        if len(data) < self.seq_length + 1:
            logger.warning(f"데이터가 충분하지 않습니다. 필요한 최소 길이: {self.seq_length + 1}, 실제 길이: {len(data)}")
            return np.array([]), np.array([])

        scaled_data = self.scaler.fit_transform(data[features])
        X, y = [], []
        for i in range(len(scaled_data) - self.seq_length):
            X.append(scaled_data[i:i + self.seq_length])
            y.append(scaled_data[i + self.seq_length, 3])  # predict close price

        X = np.array(X)
        y = np.array(y)

        logger.info(f"준비된 데이터 shape - X: {X.shape}, y: {y.shape}")
        logger.info(f"X dtype: {X.dtype}, y dtype: {y.dtype}")

        return X, y

    def train(self, data, epochs=10, batch_size=32):
        X, y = self.prepare_data(data)

        if len(X) == 0 or len(y) == 0:
            logger.warning("훈련 데이터가 비어 있습니다. 훈련을 건너뜁니다.")
            return 0  # 또는 적절한 기본값

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 5 == 0:
                logger.info(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

        return self.evaluate(X_tensor, y_tensor)

    def evaluate(self, X, y):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            loss = self.criterion(outputs.squeeze(), y)
        return 1 - loss.item()  # return accuracy-like metric

    def predict(self, data):
        self.model.eval()
        logger.info(f"Transformer predict 메서드 시작. 입력 데이터 shape: {data.shape}")

        try:
            X, _ = self.prepare_data(data)

            if X.shape[0] == 0:
                logger.warning("준비된 입력 데이터가 비어 있습니다.")
                return data['close'].iloc[-1]  # 마지막 종가 반환

            with torch.no_grad():
                X_tensor = torch.FloatTensor(X[-1]).unsqueeze(0)
                predicted = self.model(X_tensor).numpy()

            last_original = data['close'].iloc[-1]

            # 스케일러 역변환
            predicted_scaled = np.zeros((1, X.shape[-1]))
            predicted_scaled[0, -1] = predicted[0, 0]  # 마지막 열을 예측값으로 설정
            predicted_unscaled = self.scaler.inverse_transform(predicted_scaled)
            predicted_change = predicted_unscaled[0, -1]

            return last_original * (1 + predicted_change)

        except Exception as e:
            logger.error(f"Transformer 예측 중 오류 발생: {e}")
            logger.exception("상세 오류:")
            return data['close'].iloc[-1]  # 오류 발생 시 마지막 종가 반환


class ModelUpdater:
    def __init__(self, data_manager, xgboost_predictor, lstm_predictor, ml_predictor, rl_agent):
        self.data_manager = data_manager
        self.xgboost_predictor = xgboost_predictor
        self.lstm_predictor = lstm_predictor
        self.ml_predictor = ml_predictor
        self.rl_agent = rl_agent
        self.arima_predictor = ARIMAPredictor()
        self.prophet_predictor = ProphetPredictor()
        self.transformer_predictor = TransformerPredictor()
        self.weights_file = 'model_weights.json'

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

        random_search = RandomizedSearchCV(
            self.xgboost_predictor.model, param_distributions=param_dist, n_iter=50,
            cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_squared_error', n_jobs=-1, random_state=42, verbose=1
        )
        random_search.fit(X, y)

        self.xgboost_predictor.model = random_search.best_estimator_
        y_pred = self.xgboost_predictor.model.predict(X)
        performance = self.evaluate_performance(y, y_pred)

        return random_search.best_params_, performance

    def update_lstm_model(self):
        logger.info("LSTM 모델 업데이트 시작")
        data = self.data_manager.ensure_sufficient_data()
        accuracy = self.lstm_predictor.train(data)
        return {"accuracy": accuracy}, {"accuracy": accuracy}

    def update_ml_model(self):
        logger.info("ML 모델 업데이트 시작")
        data = self.data_manager.ensure_sufficient_data()
        accuracy, loss, performance = self.ml_predictor.train(self.data_manager)
        return {"accuracy": accuracy, "loss": loss}, performance

    def update_rl_model(self):
        logger.info("RL 모델 업데이트 시작")
        data = self.data_manager.ensure_sufficient_data()
        state_size = len(data.columns)
        action_size = 3  # Buy, Hold, Sell

        for _ in range(100):  # Adjust the number of training episodes as needed
            state = self.prepare_state(data.iloc[0])
            for i in range(1, len(data)):
                action = self.rl_agent.act(state)
                next_state = self.prepare_state(data.iloc[i])
                reward = self.calculate_reward(data.iloc[i-1], data.iloc[i], action)
                done = (i == len(data) - 1)
                self.rl_agent.remember(state, action, reward, next_state, done)
                state = next_state

                if len(self.rl_agent.memory) > 32:
                    self.rl_agent.replay(32)

        # Evaluate the model
        total_reward = 0
        for i in range(1, len(data)):
            state = self.prepare_state(data.iloc[i-1])
            action = self.rl_agent.act(state)
            reward = self.calculate_reward(data.iloc[i-1], data.iloc[i], action)
            total_reward += reward

        average_reward = total_reward / (len(data) - 1)
        return {"average_reward": average_reward}, {"accuracy": average_reward / 100}  # Normalize reward to [0, 1]

    def update_arima_model(self):
        logger.info("ARIMA 모델 업데이트 시작")
        data = self.data_manager.ensure_sufficient_data()
        self.arima_predictor.train(data)
        accuracy = self.arima_predictor.get_accuracy()
        return {"accuracy": accuracy}, {"accuracy": accuracy}

    def update_prophet_model(self):
        logger.info("Prophet 모델 업데이트 시작")
        data = self.data_manager.ensure_sufficient_data()
        self.prophet_predictor.train(data)
        accuracy = self.prophet_predictor.get_accuracy()
        return {"accuracy": accuracy}, {"accuracy": accuracy}

    def update_transformer_model(self):
        logger.info("Transformer 모델 업데이트 시작")
        data = self.data_manager.ensure_sufficient_data()
        accuracy = self.transformer_predictor.train(data)
        return {"accuracy": accuracy}, {"accuracy": accuracy}

    def prepare_state(self, row):
        return np.array([
            row['open'], row['high'], row['low'], row['close'], row['volume'],
            row['sma'], row['ema'], row['rsi'], row['macd'], row['signal_line'],
            row['BB_Upper'], row['BB_Lower']
        ])

    def calculate_reward(self, prev_row, curr_row, action):
        price_change = (curr_row['close'] - prev_row['close']) / prev_row['close']
        if action == 0:  # Sell
            return -price_change
        elif action == 2:  # Buy
            return price_change
        else:  # Hold
            return 0

    @staticmethod
    def evaluate_performance(y_true, y_pred):
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred)
        }

    def load_model_weights(self):
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, 'r') as f:
                    weights = json.load(f)
                logger.info(f"모델 가중치를 {self.weights_file}에서 성공적으로 로드했습니다.")
                return weights
            except json.JSONDecodeError:
                logger.error(f"{self.weights_file} 파일 디코딩 중 오류가 발생했습니다.")
            except Exception as e:
                logger.error(f"모델 가중치 로드 중 오류 발생: {str(e)}")

        logger.warning("모델 가중치 파일이 없습니다. 기본 가중치를 사용합니다.")
        return self.get_default_weights()

    def save_model_weights(self, weights):
        try:
            with open(self.weights_file, 'w') as f:
                json.dump(weights, f, indent=4)
            logger.info(f"모델 가중치를 {self.weights_file}에 성공적으로 저장했습니다.")
        except Exception as e:
            logger.error(f"모델 가중치 저장 중 오류 발생: {str(e)}")

    def get_default_weights(self):
        return {
            'gpt': 0.2,
            'ml': 0.1,
            'xgboost': 0.1,
            'rl': 0.1,
            'lstm': 0.2,
            'arima': 0.1,
            'prophet': 0.1,
            'transformer': 0.1
        }

    def update_all_models(self):
        # 모든 모델 업데이트 로직 구현
        self.update_xgboost_model()
        self.update_lstm_model()
        self.update_ml_model()
        self.update_rl_model()
        self.update_arima_model()
        self.update_prophet_model()
        self.update_transformer_model()



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

def run_backtest(data_manager, historical_data):
    backtester = Backtester(data_manager, initial_balance=10_000_000, fee_rate=0.0005)
    end_time = pd.Timestamp.now()
    start_time = end_time - pd.Timedelta(days=30)
    historical_data.reset_index(inplace=True)
    historical_data['date'] = pd.to_datetime(historical_data['date'])
    historical_data.set_index('date', inplace=True)

    # 여기서 백테스팅 전략을 정의합니다.
    def backtest_strategy(data):
        # 예시: 간단한 이동평균 교차 전략
        if len(data) < 20:
            return 0
        short_ma = data['close'].rolling(window=5).mean().iloc[-1]
        long_ma = data['close'].rolling(window=20).mean().iloc[-1]
        if short_ma > long_ma:
            return 1  # Buy signal
        elif short_ma < long_ma:
            return -1  # Sell signal
        return 0  # Hold

    results = backtester.run_backtest(backtest_strategy, start_time, end_time, historical_data)
    return results

# 추가적인 유틸리티 함수들

def prepare_data_for_ml(data: pd.DataFrame):
    """ML 모델을 위한 데이터 준비"""
    features = ['open', 'high', 'low', 'close', 'volume', 'sma', 'ema', 'rsi', 'macd', 'signal_line', 'BB_Upper', 'BB_Lower']
    X = data[features].values
    y = (data['close'].shift(-1) > data['close']).astype(int).values[:-1]
    X = X[:-1]  # 마지막 행 제거 (타겟 변수와 길이 맞추기)
    return X, y

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표 계산"""
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
