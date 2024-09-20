import json
import logging
import math
import os
import random
import warnings
from collections import deque
from typing import List, Dict, Any, Union

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import Parallel
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from scipy.stats import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pmdarima import auto_arima

logger = logging.getLogger(__name__)


class MLPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_selector = VarianceThreshold(threshold=0.0)
        self.last_mse = 0
        self.features = ['open', 'high', 'low', 'close', 'volume', 'sma', 'ema', 'rsi', 'macd', 'signal_line',
                         'BB_Upper', 'BB_Lower']
        self.is_fitted = False

    def prepare_data(self, data: pd.DataFrame):
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
            y = data['close'].shift(-1).values[:-1]  # 다음 시간의 종가를 예측
            X = X[:-1]  # 마지막 행 제거 (타겟 변수와 길이 맞추기)

            # logger.info(f"최종 X shape: {X.shape}, y shape: {y.shape}")

            return X, y

        except Exception as e:
            logger.error(f"데이터 준비 중 오류 발생: {e}")
            logger.exception("상세 오류:")
            return np.array([]), np.array([])

    def train(self, X, y):
        logger.info("ML 모델 훈련 시작")

        X = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = self.feature_selector.fit_transform(X_scaled)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        self.last_mse = mse

        logger.info(f"ML 모델 학습 완료. MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        self.is_fitted = True
        return 0, mse, {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    def predict(self, data: pd.DataFrame) -> float:
        if not self.is_fitted:
            logger.warning("MLPredictor: 모델이 학습되지 않았습니다. 예측을 진행할 수 없습니다.")
            return data['close'].iloc[-1]  # 모델이 학습되지 않았다면 현재 가격 반환

        try:
            # 기술적 지표 계산
            data = self.prepare_single_prediction_data(data)

            X = data[self.features].iloc[-1].values.reshape(1, -1)

            X = self.imputer.transform(X)
            X = self.feature_selector.transform(X)
            X_scaled = self.scaler.transform(X)
            prediction = self.model.predict(X_scaled)

            # 예측값이 확률인 경우 가격으로 변환
            if 0 <= prediction[0] <= 1:
                current_price = data['close'].iloc[-1]
                return current_price * (1 + (prediction[0] - 0.5) * 0.02)
            else:
                return prediction[0]  # 이미 가격을 예측하고 있다면 그대로 반환

        except Exception as e:
            logger.error(f"예측 중 오류 발생: {e}")
            logger.exception("상세 오류:")
            return data['close'].iloc[-1]  # 오류 발생 시 현재 가격 반환

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

    def get_mse(self):
        return self.last_mse

    def validate_data(self, data: pd.DataFrame) -> bool:
        if not all(col in data.columns for col in self.features):
            missing_columns = set(self.features) - set(data.columns)
            logger.error(f"Missing columns in data: {missing_columns}")
            return False
        return True


class XGBoostPredictor:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            random_state=42,
            early_stopping_rounds=50  # 여기로 이동
        )
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_selector = VarianceThreshold(threshold=0.0)
        self.last_mse = 0
        # self.features = ['open', 'high', 'low', 'close', 'volume', 'sma', 'ema', 'rsi', 'macd', 'signal_line',
        #                  'BB_Upper', 'BB_Lower', 'price_change', 'volume_change', 'bb_width', 'price_to_sma']
        self.features = ['open', 'high', 'low', 'close', 'volume', 'sma', 'ema', 'rsi', 'macd', 'signal_line',
                         'BB_Upper', 'BB_Lower']

        self.is_fitted = False

    def prepare_data(self, data: pd.DataFrame):
        logger.info(f"XGBoost prepare_data 메서드 시작. 데이터 shape: {data.shape}")

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

            # 추가 피처 생성
            data['price_change'] = data['close'].pct_change()
            data['volume_change'] = data['volume'].pct_change()
            data['bb_width'] = (data['BB_Upper'] - data['BB_Lower']) / data['close']
            data['price_to_sma'] = data['close'] / data['sma']

            # NaN 값을 앞뒤 값으로 채우기
            data = data.ffill().bfill()

            features = self.features + ['price_change', 'volume_change', 'bb_width', 'price_to_sma']
            X = data[features].values
            y = data['close'].shift(-1).values[:-1]  # 다음 시간의 종가를 예측
            X = X[:-1]  # 마지막 행 제거 (타겟 변수와 길이 맞추기)

            logger.info(f"XGBoost 최종 X shape: {X.shape}, y shape: {y.shape}")

            return X, y

        except Exception as e:
            logger.error(f"XGBoost 데이터 준비 중 오류 발생: {e}")
            logger.exception("상세 오류:")
            return np.array([]), np.array([])

    def train(self, X, y):
        logger.info("XGBoost 훈련 시작")

        if X.size == 0 or y.size == 0:
            logger.error("유효한 데이터가 없어 XGBoost 학습을 진행할 수 없습니다.")
            return 0, float('inf'), {}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Imputer 적용
        self.imputer.fit(X_train)
        X_train_imputed = self.imputer.transform(X_train)
        X_test_imputed = self.imputer.transform(X_test)

        # Scaler 적용
        self.scaler.fit(X_train_imputed)
        X_train_scaled = self.scaler.transform(X_train_imputed)
        X_test_scaled = self.scaler.transform(X_test_imputed)

        # Feature selection
        self.feature_selector.fit(X_train_scaled)
        X_train_selected = self.feature_selector.transform(X_train_scaled)
        X_test_selected = self.feature_selector.transform(X_test_scaled)

        # 모델 학습
        self.model.fit(X_train_selected, y_train,
                       eval_set=[(X_train_selected, y_train), (X_test_selected, y_test)],
                       verbose=100)

        self.is_fitted = True

        # 성능 평가
        predictions = self.model.predict(X_test_selected)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        self.last_mse = mse

        logger.info(f"XGBoost 훈련 완료. MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        return 0, mse, {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    def predict(self, data: pd.DataFrame) -> float:
        logger.info(f"XGBoost predict 메서드 시작. 입력 데이터 shape: {data.shape}")

        try:
            if not self.is_fitted:
                logger.warning("XGBoost: 모델이 학습되지 않았습니다. 예측을 진행할 수 없습니다.")
                return data['close'].iloc[-1]

            data = self.prepare_single_prediction_data(data)
            X = data[self.features].iloc[-1].values.reshape(1, -1)
            X = self.imputer.transform(X)
            X = self.scaler.transform(X)
            X = self.feature_selector.transform(X)

            # 직접적인 가격 예측 대신 변화율 예측으로 변경
            prediction = self.model.predict(X)[0]

            current_price = data['close'].iloc[-1]

            # 예측된 변화율을 -1%에서 1% 사이로 더욱 엄격하게 제한
            capped_change = max(min(prediction, 0.01), -0.01)

            # 이동 평균을 사용하여 추가적인 제한
            sma = data['sma'].iloc[-1]
            max_deviation = 0.005  # 이동 평균의 최대 0.5% 편차 허용
            predicted_price = current_price * (1 + capped_change)
            predicted_price = max(min(predicted_price, sma * (1 + max_deviation)), sma * (1 - max_deviation))

            logger.info(f"XGBoost 원래 예측 변화율: {prediction:.4f}, 조정된 변화율: {capped_change:.4f}")
            logger.info(f"XGBoost 현재 가격: {current_price:.2f}, 예측 가격: {predicted_price:.2f}")
            return predicted_price

        except Exception as e:
            logger.error(f"XGBoost 예측 중 오류 발생: {e}")
            logger.exception("상세 오류:")
            return data['close'].iloc[-1]

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

        # 추가 피처 생성
        data['price_change'] = data['close'].pct_change()
        data['volume_change'] = data['volume'].pct_change()
        data['bb_width'] = (data['BB_Upper'] - data['BB_Lower']) / data['close']
        data['price_to_sma'] = data['close'] / data['sma']

        return data.ffill().bfill()

    def get_mse(self):
        return self.last_mse

    def validate_data(self, data: pd.DataFrame) -> bool:
        required_columns = self.features + ['price_change', 'volume_change', 'bb_width', 'price_to_sma']
        if not all(col in data.columns for col in required_columns):
            missing_columns = set(required_columns) - set(data.columns)
            logger.error(f"XGBoost: Missing columns in data: {missing_columns}")
            return False
        return True

    def retrain(self, data: pd.DataFrame):
        logger.info("XGBoost 모델 재학습 시작")
        X, y = self.prepare_data(data)
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("XGBoost 모델 재학습 완료")



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

    def adjust_exploration_rate(self):
        self.epsilon = max(0.1, self.epsilon * 0.95)  # 탐험 비율을 5% 감소시키되, 최소 10%는 유지
        logger.info(f"RL 에이전트 탐험 비율 조정: epsilon={self.epsilon}")


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
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2, output_dim=1, seq_length=10):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.model = self._build_model()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.scaler = StandardScaler()
        self.features = ['open', 'high', 'low', 'close', 'volume', 'value', 'sma']
        self.is_fitted = False
        self.last_mse = float('inf')

    def _build_model(self):
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

        return LSTMModel(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)

    def prepare_data(self, data: pd.DataFrame):
        logger.info(f"{self.__class__.__name__} prepare_data 메서드 시작. 입력 데이터 shape: {data.shape}")

        try:
            data = data.copy()

            # 기술적 지표 계산
            data['sma'] = ta.sma(data['close'], length=20) if len(data) >= 20 else data['close']
            data['ema'] = ta.ema(data['close'], length=20) if len(data) >= 20 else data['close']
            data['rsi'] = ta.rsi(data['close'], length=14) if len(data) >= 14 else 50
            macd = ta.macd(data['close']) if len(data) >= 26 else None
            if macd is not None:
                data['macd'] = macd['MACD_12_26_9']
                data['signal_line'] = macd['MACDs_12_26_9']
            else:
                data['macd'] = 0
                data['signal_line'] = 0
            bb = ta.bbands(data['close'], length=20) if len(data) >= 20 else None
            if bb is not None:
                data['BB_Upper'] = bb['BBU_20_2.0']
                data['BB_Lower'] = bb['BBL_20_2.0']
            else:
                data['BB_Upper'] = data['close']
                data['BB_Lower'] = data['close']

            # NaN 값 처리
            data = data.ffill().bfill()

            logger.info(f"기술적 지표 계산 후 데이터 shape: {data.shape}")
            logger.info(f"데이터 컬럼: {data.columns.tolist()}")
            logger.info(f"NaN 값 개수:\n{data.isnull().sum()}")

            # feature 선택 및 스케일링
            scaled_data = self.scaler.fit_transform(data[self.features])

            # 데이터 길이가 seq_length보다 작은 경우 처리
            if len(scaled_data) < self.seq_length:
                logger.warning(f"데이터 길이가 seq_length보다 작습니다. 데이터를 반복하여 확장합니다.")
                repeat_times = self.seq_length // len(scaled_data) + 1
                scaled_data = np.tile(scaled_data, (repeat_times, 1))[:self.seq_length]

            X = scaled_data[-self.seq_length:].reshape(1, self.seq_length, -1)
            y = scaled_data[-1, self.features.index('close')].reshape(1, -1)

            logger.info(f"준비된 데이터 shape - X: {X.shape}, y: {y.shape}")
            logger.info(f"X dtype: {X.dtype}, y dtype: {y.dtype}")
            logger.info(f"X 샘플:\n{X[0]}")
            logger.info(f"y 샘플:\n{y[0]}")

            return X, y

        except Exception as e:
            logger.error(f"데이터 준비 중 오류 발생: {e}")
            logger.exception("상세 오류:")
            # 오류 발생 시 더미 데이터 반환
            dummy_X = np.zeros((1, self.seq_length, len(self.features)))
            dummy_y = np.zeros((1, 1))
            return dummy_X, dummy_y

    def train(self, data: pd.DataFrame, epochs=50, batch_size=32):
        logger.info(f"LSTM 훈련 시작. 입력 데이터 shape: {data.shape}")

        X, y = self.prepare_data(data)
        if X.size == 0 or y.size == 0:
            logger.error("유효한 데이터가 없어 학습을 진행할 수 없습니다.")
            return 0, float('inf'), {}

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(-1)

        logger.debug(f"X_tensor shape: {X_tensor.shape}, y_tensor shape: {y_tensor.shape}")

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

        self.is_fitted = True
        mse = self.evaluate(X_tensor, y_tensor)
        self.last_mse = mse
        logger.info(f"LSTM 훈련 완료. MSE: {mse}")
        return 0, mse, {"mse": mse}

    def evaluate(self, X, y):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            mse = self.criterion(outputs, y)
        return mse.item()

    def predict(self, data: pd.DataFrame) -> float:
        logger.info(f"{self.__class__.__name__} predict 메서드 시작. 입력 데이터 shape: {data.shape}")

        if not self.is_fitted:
            logger.warning(f"{self.__class__.__name__}: 모델이 학습되지 않았습니다. 예측을 진행할 수 없습니다.")
            return data['close'].iloc[-1]

        try:
            X, _ = self.prepare_data(data)

            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                logger.info(f"모델 입력 텐서 shape: {X_tensor.shape}")
                predicted_scaled = self.model(X_tensor)
                logger.info(f"모델 출력 (스케일링된 예측): {predicted_scaled.numpy()}")

            if torch.isnan(predicted_scaled).any():
                logger.warning("모델 출력에 NaN 값이 있습니다. 현재 가격을 반환합니다.")
                return data['close'].iloc[-1]

            # 스케일러 역변환
            predicted_full = np.zeros((1, len(self.features)))
            predicted_full[0, self.features.index('close')] = predicted_scaled.numpy()[0, 0]
            predicted_unscaled = self.scaler.inverse_transform(predicted_full)
            predicted_price = predicted_unscaled[0, self.features.index('close')]

            logger.info(f"스케일러 역변환 후 예측 가격: {predicted_price}")

            if np.isnan(predicted_price) or np.isinf(predicted_price):
                logger.error(f"예측된 가격이 유효하지 않습니다: {predicted_price}")
                return data['close'].iloc[-1]

            logger.info(f"{self.__class__.__name__} 최종 예측 가격: {predicted_price}")
            return predicted_price

        except Exception as e:
            logger.error(f"{self.__class__.__name__} 예측 중 오류 발생: {e}")
            logger.exception("상세 오류:")
            return data['close'].iloc[-1]

    def get_mse(self):
        return self.last_mse

    def randomize_hyperparameters(self):
        self.hidden_dim = np.random.randint(32, 128)
        self.num_layers = np.random.randint(1, 4)
        learning_rate = 10 ** np.random.uniform(-4, -2)
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        logger.info(f"LSTM 하이퍼파라미터 랜덤 조정: hidden_dim={self.hidden_dim}, num_layers={self.num_layers}, learning_rate={learning_rate}")


class ARIMAPredictor:
    def __init__(self):
        self.model = None
        self.volatility_model = None
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3)
        self.is_trained = False
        self.last_train_data = None

    def preprocess_data(self, data):
        if isinstance(data, pd.Series):
            df = data.to_frame(name='close')
        elif isinstance(data, pd.DataFrame) and 'close' in data.columns:
            df = data[['close']].copy()
        else:
            raise ValueError("Input data must be a Series or a DataFrame with 'close' column")

        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['sma'] = df['close'].rolling(window=20).mean()
        df['upper_bb'], df['lower_bb'] = self._calculate_bollinger_bands(df['close'])

        # NaN 값 처리
        df = df.dropna()

        return df

    def train(self, data):
        logger.info(f"ARIMA 훈련 시작. 입력 데이터 타입: {type(data)}")
        logger.info(f"데이터 샘플:\n{data.head()}")

        processed_data = self.preprocess_data(data)
        logger.info(f"전처리 후 데이터 shape: {processed_data.shape}")
        logger.info(f"전처리 후 데이터 컬럼: {processed_data.columns.tolist()}")

        # 데이터 스케일링
        scaled_data = processed_data['close'] * 100

        # 변동성 모델 학습 (경고 무시)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.volatility_model = arch_model(processed_data['returns'].dropna(), vol='GARCH', p=1, q=1, rescale=False)
            self.volatility_results = self.volatility_model.fit(disp='off')

        # 최적의 ARIMAX 모델 찾기
        exog_vars = ['volatility', 'sma', 'upper_bb', 'lower_bb']

        # 경고 무시 및 매개변수 조정
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = auto_arima(scaled_data, exogenous=processed_data[exog_vars],
                                    start_p=0, start_q=0, max_p=2, max_q=2, m=1,
                                    start_P=0, seasonal=False, d=1, D=0, trace=True,
                                    error_action='ignore', suppress_warnings=True, stepwise=True,
                                    maxiter=50, method='lbfgs')

        self.is_trained = True
        logger.info("ARIMA 모델 훈련 완료")

    def predict(self, data: Union[pd.DataFrame, Dict[str, float]]) -> Union[float, None]:
        if not self.is_trained:
            logger.warning("모델이 학습되지 않았습니다. 먼저 train 메서드를 호출하세요.")
            return None

        try:
            logger.info("ARIMA 예측 시작")

            if isinstance(data, dict):
                logger.info(f"입력 데이터: {data}")
                last_data_point = data
            elif isinstance(data, pd.DataFrame):
                logger.info(f"입력 데이터 shape: {data.shape}")
                last_data_point = data.iloc[-1].to_dict()
            else:
                raise ValueError("Unsupported data type for ARIMA prediction")

            future_exog = self._forecast_exog_vars(last_data_point, steps=1)
            logger.info(f"Future exogenous variables: {future_exog}")

            # predict 메서드 사용
            forecast, conf_int = self.model.predict(n_periods=1, return_conf_int=True, exogenous=future_exog)
            logger.info(f"ARIMA 예측 결과: {forecast}")
            logger.info(f"신뢰 구간: {conf_int}")

            # forecast가 Series인 경우 첫 번째 값을 가져옵니다.
            if isinstance(forecast, pd.Series):
                prediction = forecast.iloc[0]
            else:
                prediction = forecast[0]

            # 스케일링 되돌리기 (100으로 나누기)
            prediction = prediction / 100

            logger.info(f"최종 ARIMA 예측값: {prediction}")
            return float(prediction)

        except Exception as e:
            logger.error(f"ARIMA 예측 중 오류 발생: {e}")
            logger.exception("상세 오류:")
            return None

    def _forecast_exog_vars(self, last_data_point: Dict[str, float], steps: int) -> pd.DataFrame:
        logger.info("Exogenous 변수 예측 시작")
        exog_vars = ['volatility', 'sma', 'upper_bb', 'lower_bb']
        future_exog = pd.DataFrame({var: [last_data_point.get(var, 0)] * steps for var in exog_vars})
        logger.info(f"예측된 Exogenous 변수:\n{future_exog}")
        return future_exog

    def _calculate_confidence_intervals(self, forecast: np.ndarray, volatility_forecast: Any) -> List[tuple]:
        return [
            (pred - 1.96 * np.sqrt(vol), pred + 1.96 * np.sqrt(vol))
            for pred, vol in zip(forecast, volatility_forecast.variance.values[:, 0])
        ]

    def _calculate_bollinger_bands(self, price: pd.Series, window: int = 20, num_std: int = 2) -> tuple:
        sma = price.rolling(window=window).mean()
        std = price.rolling(window=window).std()
        upper_bb = sma + (std * num_std)
        lower_bb = sma - (std * num_std)
        return upper_bb, lower_bb


class ProphetPredictor:
    def __init__(self):
        self.model = None
        self.last_train_time = None
        self.train_interval = 1800  # 30분마다 재학습
        self.is_trained = False
        self.scaler = RobustScaler()

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.reset_index()
        df = df.rename(columns={'date': 'ds', 'close': 'y'})

        # 최근 7일 데이터만 사용
        df = df.tail(7 * 24 * 6)  # 7일 * 24시간 * 6 (10분 간격)

        # 퍼센트 변화율 계산
        df['y'] = df['y'].pct_change()
        df = df.dropna()

        # 추가 리그레서
        df['volume'] = data['volume'].values[-len(df):]
        df['sma'] = data['close'].rolling(window=20).mean().values[-len(df):]
        df['rsi'] = ta.rsi(data['close'], length=14).values[-len(df):]

        return df

    def train(self, data: pd.DataFrame) -> None:
        current_time = pd.Timestamp.now()
        if self.last_train_time is None or (current_time - self.last_train_time).total_seconds() > self.train_interval:
            logger.info("Prophet 모델 훈련 시작")
            processed_data = self.preprocess_data(data)

            self.model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                seasonality_mode='additive',
                changepoint_prior_scale=0.001,
                seasonality_prior_scale=10.0,
                holidays_prior_scale=1.0,
                changepoint_range=0.8
            )

            self.model.add_regressor('volume')
            self.model.add_regressor('sma')
            self.model.add_regressor('rsi')

            self.model.fit(processed_data)
            self.last_train_time = current_time
            self.is_trained = True

            logger.info("Prophet 모델 훈련 완료")

    def predict(self, last_data_point: Dict[str, float], periods: int = 2) -> Union[float, None]:
        if not self.is_trained:
            logger.warning("모델이 학습되지 않았습니다. 먼저 train 메서드를 호출하세요.")
            return None

        try:
            future = self.model.make_future_dataframe(periods=periods, freq='10T')
            future['volume'] = last_data_point['volume']
            future['sma'] = last_data_point['sma']
            future['rsi'] = last_data_point['rsi']

            forecast = self.model.predict(future)
            prediction_pct = forecast['yhat'].iloc[-1]  # 10분 후 예측

            # 퍼센트 변화율을 실제 가격으로 변환
            current_price = last_data_point['close']
            prediction = current_price * (1 + prediction_pct)

            # if abs(prediction - current_price) / current_price > 0.01:
            #     logger.warning(
            #         f"Prophet 예측값 ({prediction:.2f})이 현재 가격 ({current_price:.2f})과 1% 이상 차이납니다. 현재 가격을 사용합니다.")
            #     return current_price

            return prediction

        except Exception as e:
            logger.error(f"Prophet 예측 중 오류 발생: {e}")
            return last_data_point.get('close', None)

    def get_accuracy(self) -> float:
        if not self.is_trained or self.model is None:
            logger.warning("모델이 학습되지 않았습니다.")
            return 0

        historical_forecast = self.model.predict()
        mape = np.mean(np.abs((self.model.history['y'] - historical_forecast['yhat']) / self.model.history['y']))
        return max(0, 1 - mape)

    def update_model(self, new_data: pd.DataFrame) -> None:
        self.train(new_data)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, seq_length, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # 입력 임베딩
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_length)

        # 다중 스케일 컨볼루션 레이어
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)

        # Transformer 인코더
        encoder_layers = nn.TransformerEncoderLayer(d_model * 3, nhead, dim_feedforward, dropout=dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 어텐션 메커니즘
        self.attention = nn.MultiheadAttention(d_model * 3, nhead, dropout=dropout, batch_first=True)

        # 출력 레이어
        self.fc1 = nn.Linear(d_model * 3, d_model)
        self.fc2 = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

    def forward(self, src):
        # 입력 임베딩
        src = self.embedding(src)
        src = self.pos_encoder(src)

        # 다중 스케일 특징 추출
        conv1 = self.conv1(src.transpose(1, 2)).transpose(1, 2)
        conv3 = self.conv3(src.transpose(1, 2)).transpose(1, 2)
        conv5 = self.conv5(src.transpose(1, 2)).transpose(1, 2)

        # 특징 연결
        combined = torch.cat((conv1, conv3, conv5), dim=2)

        # Transformer 인코딩
        encoded = self.transformer_encoder(combined)

        # 셀프 어텐션
        attn_output, _ = self.attention(encoded, encoded, encoded)

        # 최종 예측
        output = self.activation(self.fc1(attn_output[:, -1, :]))
        output = self.dropout(output)
        output = self.fc2(output)

        return output

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TransformerPredictor:
    def __init__(self, input_dim=13, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, seq_length=10):
        self.model = TransformerModel(input_dim, d_model, nhead, num_layers, dim_feedforward, seq_length)
        self.criterion = nn.HuberLoss(reduction='mean')
        self.learning_rate = 0.001  # 초기 학습률 설정
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scaler = StandardScaler()
        self.seq_length = seq_length
        self.features = ['open', 'high', 'low', 'close', 'volume', 'value', 'sma', 'ema', 'rsi', 'macd', 'signal_line',
                         'BB_Upper', 'BB_Lower']
        self.is_fitted = False
        self.last_mse = float('inf')
        self.fallback_strategy = 'ewm'  # 새로운 fallback 전략

        logger.info(f"TransformerPredictor initialized with input_dim={input_dim}, seq_length={seq_length}, learning_rate={self.learning_rate}")

    def prepare_data(self, data: pd.DataFrame):
        logger.info(f"Transformer prepare_data 메서드 시작. 입력 데이터 shape: {data.shape}")

        try:
            data = data.copy()

            # 기술적 지표 계산
            data['sma'] = ta.sma(data['close'], length=20) if len(data) >= 20 else data['close']
            data['ema'] = ta.ema(data['close'], length=20) if len(data) >= 20 else data['close']
            data['rsi'] = ta.rsi(data['close'], length=14) if len(data) >= 14 else 50
            macd = ta.macd(data['close']) if len(data) >= 26 else None
            if macd is not None:
                data['macd'] = macd['MACD_12_26_9']
                data['signal_line'] = macd['MACDs_12_26_9']
            else:
                data['macd'] = 0
                data['signal_line'] = 0
            bb = ta.bbands(data['close'], length=20) if len(data) >= 20 else None
            if bb is not None:
                data['BB_Upper'] = bb['BBU_20_2.0']
                data['BB_Lower'] = bb['BBL_20_2.0']
            else:
                data['BB_Upper'] = data['close']
                data['BB_Lower'] = data['close']

            logger.info(f"기술적 지표 계산 후 데이터 shape: {data.shape}")
            logger.info(f"데이터 컬럼: {data.columns.tolist()}")
            logger.info(f"NaN 값 개수:\n{data.isnull().sum()}")

            # NaN 값 처리를 개선
            data = data.fillna(method='ffill').fillna(method='bfill')
            if data.isnull().any().any():
                logger.warning("NaN 값이 여전히 존재합니다. 0으로 대체합니다.")
                data = data.fillna(0)

            # feature 선택 및 스케일링
            scaled_data = self.scaler.fit_transform(data[self.features])

            # 데이터 길이가 seq_length보다 작은 경우 처리
            if len(scaled_data) < self.seq_length:
                logger.warning(f"데이터 길이가 seq_length보다 작습니다. 데이터를 반복하여 확장합니다.")
                repeat_times = self.seq_length // len(scaled_data) + 1
                scaled_data = np.tile(scaled_data, (repeat_times, 1))[:self.seq_length]

            X = scaled_data[-self.seq_length:].reshape(1, self.seq_length, -1)
            y = scaled_data[-1, self.features.index('close')].reshape(1, -1)

            logger.info(f"준비된 데이터 shape - X: {X.shape}, y: {y.shape}")
            logger.info(f"X dtype: {X.dtype}, y dtype: {y.dtype}")
            logger.info(f"X 샘플:\n{X[0]}")
            logger.info(f"y 샘플:\n{y[0]}")

            return X, y

        except Exception as e:
            logger.error(f"데이터 준비 중 오류 발생: {e}")
            logger.exception("상세 오류:")
            # 오류 발생 시 더미 데이터 반환
            dummy_X = np.zeros((1, self.seq_length, len(self.features)))
            dummy_y = np.zeros((1, 1))
            return dummy_X, dummy_y

    def train(self, data: pd.DataFrame, epochs=50, batch_size=32):
        logger.info(f"Transformer 훈련 시작. 입력 데이터 shape: {data.shape}")

        X, y = self.prepare_data(data)
        if X.size == 0 or y.size == 0:
            logger.error("유효한 데이터가 없어 학습을 진행할 수 없습니다.")
            return 0, float('inf'), {}

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(-1)

        logger.debug(f"X_tensor shape: {X_tensor.shape}, y_tensor shape: {y_tensor.shape}")

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 조기 종료를 위한 변수들
        best_loss = float('inf')
        patience = 3
        no_improve = 0

        # 학습 중 NaN 체크 추가
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze(-1)
                if torch.isnan(outputs).any():
                    logger.error(f"Epoch {epoch}: NaN 출력 발생. 이 배치를 건너뜁니다.")
                    continue
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

            # 조기 종료 검사
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"조기 종료: {patience} 에폭 동안 개선 없음")
                    break

        self.is_fitted = True
        mse = self.evaluate(X_tensor, y_tensor)
        self.last_mse = mse
        logger.info(f"Transformer 훈련 완료. MSE: {mse}")
        return 0, mse, {"mse": mse}

    def evaluate(self, X, y):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            mse = self.criterion(outputs, y)
        return mse.item()

    def predict(self, data: pd.DataFrame) -> float:
        logger.info(f"TransformerPredictor predict 메서드 시작. 입력 데이터 shape: {data.shape}")

        if not self.is_fitted:
            logger.warning("TransformerPredictor: 모델이 학습되지 않았습니다. 예측을 진행할 수 없습니다.")
            return self.fallback_prediction(data)

        try:
            X, _ = self.prepare_data(data)

            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                logger.info(f"모델 입력 텐서 shape: {X_tensor.shape}")
                predicted_scaled = self.model(X_tensor).squeeze(-1)
                logger.info(f"모델 출력 (스케일링된 예측): {predicted_scaled.numpy()}")

            if torch.isnan(predicted_scaled).any():
                logger.error("모델 출력에 NaN 값이 있습니다. 대체 예측 방법을 사용합니다.")
                return self.fallback_prediction(data)

            # 스케일러 역변환
            predicted_full = np.zeros((1, len(self.features)))
            predicted_full[0, self.features.index('close')] = predicted_scaled.numpy()[0]
            predicted_unscaled = self.scaler.inverse_transform(predicted_full)
            predicted_price = predicted_unscaled[0, self.features.index('close')]

            logger.info(f"스케일러 역변환 후 예측 가격: {predicted_price}")

            if np.isnan(predicted_price) or np.isinf(predicted_price):
                logger.error(f"예측된 가격이 유효하지 않습니다: {predicted_price}")
                return self.fallback_prediction(data)

            logger.info(f"TransformerPredictor 최종 예측 가격: {predicted_price}")
            return predicted_price

        except Exception as e:
            logger.error(f"TransformerPredictor 예측 중 오류 발생: {e}")
            logger.exception("상세 오류:")
            return self.fallback_prediction(data)

    def fallback_prediction(self, data: pd.DataFrame) -> float:
        logger.warning("Fallback 예측 방법 사용")
        if self.fallback_strategy == 'ewm':
            return data['close'].ewm(span=5).mean().iloc[-1]
        elif self.fallback_strategy == 'last_price':
            return data['close'].iloc[-1]
        else:
            return data['close'].mean()

    def adjust_learning_rate(self):
        self.learning_rate *= 0.9  # 학습률을 10% 감소
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        logger.info(f"Transformer 학습률 조정: learning_rate={self.learning_rate}")


class BitcoinForecastPredictor:
    def __init__(self, seq_length=30, features=['close', 'volume']):
        self.seq_length = seq_length
        self.features = features
        # self.scaler = MinMaxScaler()
        self.scaler = RobustScaler(unit_variance=True)  # unit_variance=True를 추가
        self.model = None
        self.is_fitted = False

    def create_model(self, input_dim):
        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim=16, num_layers=1, output_dim=1):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return out

        self.model = LSTMModel(input_dim)

    def prepare_data(self, data):
        logger.info(f"BitcoinForecastPredictor prepare_data 시작. 입력 데이터 shape: {data.shape}")

        # 최근 3000개의 데이터만 사용
        data = data.tail(3000)

        # NaN과 무한대 값 처리
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        if data.empty:
            logger.warning("유효한 데이터가 없습니다.")
            return np.array([]), np.array([])

        # 스케일링 전 유효성 검사
        valid_data = data[self.features].select_dtypes(include=[np.number]).dropna()
        if valid_data.empty:
            logger.warning("스케일링에 적합한 유효한 데이터가 없습니다.")
            return np.array([]), np.array([])

        # 0이 아닌 값만 사용하여 스케일링
        non_zero_mask = (valid_data != 0).all(axis=1)
        non_zero_data = valid_data[non_zero_mask]

        if non_zero_data.empty:
            logger.warning("0이 아닌 유효한 데이터가 없습니다.")
            return np.array([]), np.array([])

        scaled_data = self.scaler.fit_transform(non_zero_data)

        X, y = [], []
        for i in range(len(scaled_data) - self.seq_length):
            X.append(scaled_data[i:i + self.seq_length])
            y.append(scaled_data[i + self.seq_length, 0])  # 'close'가 첫 번째 특성이라고 가정

        X = np.array(X)
        y = np.array(y)

        logger.info(f"준비된 데이터 shape - X: {X.shape}, y: {y.shape}")

        return X, y

    def train(self, data):
        logger.info("BitcoinForecastPredictor 훈련 시작")
        X, y = self.prepare_data(data)

        if X.size == 0 or y.size == 0:
            logger.warning("훈련에 사용할 유효한 데이터가 없습니다.")
            return

        self.create_model(len(self.features))

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)

        if len(X_tensor) < self.seq_length:
            logger.warning(f"데이터가 충분하지 않습니다. 필요한 최소 길이: {self.seq_length}, 실제 길이: {len(X_tensor)}")
            return

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        epochs = 30
        batch_size = min(64, len(X_tensor))  # 데이터 크기에 따라 배치 크기 조정
        n_batches = len(X_tensor) // batch_size

        if n_batches == 0:
            logger.warning("배치 크기가 데이터 크기보다 큽니다. 훈련을 진행할 수 없습니다.")
            return

        for epoch in range(epochs):
            total_loss = 0
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                batch_X = X_tensor[start:end]
                batch_y = y_tensor[start:end]

                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / n_batches
            logger.info(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

        self.is_fitted = True
        logger.info("BitcoinForecastPredictor 훈련 완료")

    def predict(self, data):
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 train() 메서드를 호출하세요.")

        scaled_data = self.scaler.transform(data[self.features].tail(self.seq_length))
        X = np.array([scaled_data])
        X_tensor = torch.FloatTensor(X)

        with torch.no_grad():
            predicted = self.model(X_tensor)

        predicted_reshaped = np.zeros((1, len(self.features)))
        predicted_reshaped[:, 0] = predicted.numpy()  # 'close' 값만 사용

        return self.scaler.inverse_transform(predicted_reshaped)[0, 0]

    def save_model(self, path='bitcoin_forecast_model'):
        try:
            # 모델 상태 저장
            torch.save(self.model.state_dict(), f"{path}_weights.pth")

            # 스케일러 저장
            joblib.dump(self.scaler, f"{path}_scaler.joblib")

            # 기타 정보 저장
            metadata = {
                'is_fitted': self.is_fitted,
                'seq_length': self.seq_length,
                'features': self.features
            }
            with open(f"{path}_metadata.json", 'w') as f:
                json.dump(metadata, f)

            logger.info(f"모델이 {path}에 저장되었습니다.")
        except Exception as e:
            logger.error(f"모델 저장 중 오류 발생: {e}")

    def load_model(self, path='bitcoin_forecast_model'):
        try:
            # 모델 상태 로드
            self.create_model(len(self.features))
            self.model.load_state_dict(torch.load(f"{path}_weights.pth", map_location=torch.device('cpu')))

            # 스케일러 로드
            self.scaler = joblib.load(f"{path}_scaler.joblib")

            # 기타 정보 로드
            with open(f"{path}_metadata.json", 'r') as f:
                metadata = json.load(f)

            self.is_fitted = metadata['is_fitted']
            self.seq_length = metadata['seq_length']
            self.features = metadata['features']

            logger.info(f"모델이 {path}에서 로드되었습니다.")
        except Exception as e:
            logger.error(f"모델 로딩 중 오류 발생: {e}")
            self.model = None
            self.is_fitted = False


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)


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
