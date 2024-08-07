from collections import deque
import random
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.metrics import mean_squared_error


class MLPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.last_accuracy = 0
        self.last_loss = 0

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare the data for training the model."""
        features = data[['open', 'high', 'low', 'close', 'volume']].values
        features = self.scaler.fit_transform(features)
        labels = np.where(data['close'].shift(-1) > data['close'], 1, 0)[:-1]
        return features[:-1], labels

    def train(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Train the RandomForest model using the provided data."""
        X, y = self.prepare_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        # Calculate accuracy
        self.last_accuracy = self.model.score(X_test, y_test) * 100

        # Calculate loss (for Random Forest, we'll use mean squared error)
        y_pred = self.model.predict(X_test)
        self.last_loss = mean_squared_error(y_test, y_pred)

        print(f"Model accuracy: {self.last_accuracy:.2f}%")
        return self.last_accuracy, self.last_loss

    def predict(self, data: pd.DataFrame) -> int:
        """Predict the next action based on the model."""
        features = self.scaler.transform(data[['open', 'high', 'low', 'close', 'volume']].values)
        prediction = self.model.predict(features[-1].reshape(1, -1))
        return prediction[0]


class RLAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Limit memory size
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.batch_size = 64  # Increased batch size for faster learning
        self.recent_rewards = deque(maxlen=100)  # Store recent rewards for monitoring

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.recent_rewards.append(reward)

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size: int):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            states.append(state[0])
            targets_f.append(target_f[0])
        self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0, batch_size=self.batch_size)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name: str):
        self.model.load_weights(name)

    def save(self, name: str):
        self.model.save_weights(name)

    def get_average_reward(self):
        if len(self.recent_rewards) == 0:
            return 0
        return sum(self.recent_rewards) / len(self.recent_rewards)