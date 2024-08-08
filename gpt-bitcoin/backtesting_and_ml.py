import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from collections import deque
import random
from typing import List, Dict, Callable, Any, Tuple
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, data: pd.DataFrame, initial_balance: float, fee_rate: float):
        self.data = data
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.trades_per_day = 144  # 10분 간격으로 1일 144회 거래 가능

    def run_walkforward_analysis(self, strategy: Callable, window_size: int, step_size: int) -> Dict:
        if len(self.data) < window_size:
            logger.warning(f"Not enough data for walk-forward analysis. Using all data for single run.")
            return self.single_run_backtest(strategy)

        results = []
        for start in range(0, len(self.data) - window_size, step_size):
            end = start + window_size
            train_data = self.data.iloc[start:end]
            test_data = self.data.iloc[end:min(end + step_size, len(self.data))]

            if len(test_data) == 0:
                break

            optimized_params = self.optimize_strategy(strategy, train_data)
            test_result = self.backtest(strategy, test_data, optimized_params)
            results.append(test_result)

        return self.aggregate_results(results)

    def single_run_backtest(self, strategy: Callable) -> Dict:
        params = {'sma_short': 5, 'sma_long': 10}  # Default params for small dataset
        result = self.backtest(strategy, self.data, params)
        return {
            'mean_return': result['total_return'],
            'std_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': self.calculate_max_drawdown([result['total_return']])
        }

    def optimize_strategy(self, strategy: Callable, data: pd.DataFrame) -> Dict:
        def objective(params):
            int_params = [max(1, int(round(p))) for p in params]
            result = self.backtest(strategy, data, {'sma_short': int_params[0], 'sma_long': int_params[1]})
            return -result['total_return']

        initial_params = [5, 10]
        bounds = [(2, min(20, len(data) // 2)), (5, min(50, len(data) - 1))]

        result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)

        optimized_params = [max(1, int(round(p))) for p in result.x]
        return dict(zip(['sma_short', 'sma_long'], optimized_params))

    def backtest(self, strategy: Callable, data: pd.DataFrame, params: Dict) -> Dict:
        balance = self.initial_balance
        position = 0
        trades = []

        for i in range(1, len(data)):
            signal = strategy(data.iloc[:i], params)
            current_price = data['close'].iloc[i]

            if signal == 1 and balance > 0:  # Buy signal
                buy_amount = balance * 0.99  # Consider fees
                position += buy_amount / current_price * (1 - self.fee_rate)
                balance = 0
                trades.append(('buy', current_price, position))
            elif signal == -1 and position > 0:  # Sell signal
                balance += position * current_price * (1 - self.fee_rate)
                position = 0
                trades.append(('sell', current_price, balance))

        final_balance = balance + position * data['close'].iloc[-1]
        total_return = (final_balance - self.initial_balance) / self.initial_balance * 100

        return {
            'final_balance': final_balance,
            'total_return': total_return,
            'num_trades': len(trades)
        }

    def aggregate_results(self, results: List[Dict]) -> Dict:
        if not results:
            logger.warning("No results to aggregate.")
            return {'mean_return': 0, 'std_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}

        returns = [r['total_return'] for r in results]
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0,
            'max_drawdown': self.calculate_max_drawdown(returns)
        }

    def calculate_max_drawdown(self, returns: List[float]) -> float:
        if not returns:
            return 0
        cumulative_returns = np.cumprod(1 + np.array(returns) / 100)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        return np.max(drawdown) * 100

    def identify_market_regime(self, regime: str) -> pd.DataFrame:
        if regime not in ['bull', 'bear', 'sideways']:
            raise ValueError("Invalid regime. Choose 'bull', 'bear', or 'sideways'.")

        self.data['SMA_20'] = self.data['close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['close'].rolling(window=50).mean()
        self.data['ATR'] = self.calculate_atr(14)
        self.data['ADX'] = self.calculate_adx(14)

        bull_condition = (self.data['close'] > self.data['SMA_20']) & (self.data['SMA_20'] > self.data['SMA_50']) & (self.data['ADX'] > 25)
        bear_condition = (self.data['close'] < self.data['SMA_20']) & (self.data['SMA_20'] < self.data['SMA_50']) & (self.data['ADX'] > 25)
        sideways_condition = (self.data['ADX'] <= 25) | ((self.data['SMA_20'] - self.data['SMA_50']).abs() / self.data['SMA_50'] < 0.02)

        if regime == 'bull':
            return self.data[bull_condition]
        elif regime == 'bear':
            return self.data[bear_condition]
        else:  # sideways
            return self.data[sideways_condition]

    def calculate_atr(self, period: int) -> pd.Series:
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window=period).mean()

    def calculate_adx(self, period: int) -> pd.Series:
        plus_dm = self.data['high'].diff()
        minus_dm = self.data['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr = self.calculate_atr(1)
        plus_di = 100 * (plus_dm.rolling(window=period).sum() / tr.rolling(window=period).sum())
        minus_di = 100 * (minus_dm.rolling(window=period).sum() / tr.rolling(window=period).sum())

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx

    def calculate_market_volatility(self, window: int = 20) -> float:
        returns = self.data['close'].pct_change()
        volatility = returns.rolling(window=window).std()
        return volatility.iloc[-1] * np.sqrt(252)  # Annualized volatility


class MLPredictor:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42)
        }
        self.model_weights = {'random_forest': 0.4, 'gradient_boosting': 0.4, 'svm': 0.2}
        self.scaler = StandardScaler()
        self.min_data_points = 30  # 최소 필요 데이터 포인트 수 설정
        self.last_accuracy = 0
        self.last_loss = 0

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # 데이터 복사
        df = data.copy()

        if len(data) < 1381:
            raise ValueError(f"최소 1381개의 데이터 포인트가 필요합니다. 현재 데이터 포인트: {len(data)}")

        # 기술적 지표 추가
        df['SMA'] = df['close'].rolling(window=min(20, len(df))).mean()
        df['EMA'] = df['close'].ewm(span=min(20, len(df)), adjust=False).mean()
        df['RSI'] = self.calculate_rsi(df['close'], window=min(14, len(df)))
        df['BB_Upper'], df['BB_Lower'] = self.calculate_bollinger_bands(df['close'], window=min(20, len(df)))
        df['ATR'] = self.calculate_atr(df, window=min(14, len(df)))
        df['MACD'], df['Signal_Line'] = self.calculate_macd(df['close'])
        df['Volume_SMA'] = df['volume'].rolling(window=min(20, len(df))).mean()
        df['Price_Change'] = df['close'].pct_change()

        features = [
            'open', 'high', 'low', 'close', 'volume',
            'SMA', 'EMA', 'RSI', 'BB_Upper', 'BB_Lower', 'ATR',
            'MACD', 'Signal_Line', 'Volume_SMA', 'Price_Change'
        ]

        # NaN 값 제거
        df = df.dropna()

        if len(df) == 0:
            raise ValueError("데이터프레임이 비어 있습니다. 충분한 데이터가 없습니다.")

        X = df[features].values
        y = (df['close'].shift(-1) > df['close']).astype(int).values[:-1]
        return X[:-1], y  # X의 마지막 행은 y에 대응하는 레이블이 없으므로 제거

    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[
        pd.Series, pd.Series]:
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band

    def calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window=window).mean()

    def calculate_macd(self, prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> \
    Tuple[pd.Series, pd.Series]:
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal

    def train(self, data: pd.DataFrame):
        X, y = self.prepare_data(data)
        if len(X) != len(y):
            logger.warning(f"X와 y의 길이가 일치하지 않습니다. X: {len(X)}, y: {len(y)}")
            # X와 y의 길이를 맞춥니다
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        performance = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            performance[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }

        overall_accuracy = np.mean([perf['accuracy'] for perf in performance.values()])
        overall_loss = 1 - overall_accuracy
        self.last_accuracy = overall_accuracy
        self.last_loss = overall_loss
        return overall_accuracy, overall_loss, performance

    def predict(self, data: pd.DataFrame) -> Tuple[int, float]:
        try:
            X, _ = self.prepare_data(data)
            if X.shape[0] == 0:
                raise ValueError("예측을 위한 데이터가 비어 있습니다.")

            X_scaled = self.scaler.transform(X)

            predictions = {}
            for name, model in self.models.items():
                predictions[name] = model.predict_proba(X_scaled[-1].reshape(1, -1))[:, 1]

            weighted_prediction = sum(self.model_weights[name] * pred for name, pred in predictions.items())
            confidence = abs(weighted_prediction - 0.5) * 2

            return 1 if weighted_prediction > 0.5 else 0, confidence[0]
        except ValueError as e:
            logger.warning(f"ML 예측 실패: {e}")
            return 0, 0.0  # 기본값 반환
        except Exception as e:
            logger.error(f"ML 예측 중 예상치 못한 오류 발생: {e}")
            return 0, 0.0  # 기본값 반환

    def get_accuracy(self):
        return self.last_accuracy

    def get_loss(self):
        return self.last_loss


class RLAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.batch_size = 64
        self.recent_rewards = deque(maxlen=100)

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


def run_backtest(historical_data: pd.DataFrame):
    backtester = Backtester(historical_data, initial_balance=10_000_000, fee_rate=0.0005)
    data_length = len(historical_data)
    window_size = min(180, max(5, data_length // 2))
    step_size = max(1, data_length // 10)

    logger.info(f"Running backtest with window_size={window_size}, step_size={step_size}")
    results = backtester.run_walkforward_analysis(example_strategy, window_size=window_size, step_size=step_size)
    logger.info(f"Backtest results: {results}")
    return results


def example_strategy(data: pd.DataFrame, params: Dict) -> int:
    if len(data) < params['sma_long']:
        return 0

    sma_short = data['close'].rolling(window=params['sma_short']).mean().iloc[-1]
    sma_long = data['close'].rolling(window=params['sma_long']).mean().iloc[-1]

    if sma_short > sma_long:
        return 1  # Buy signal
    elif sma_short < sma_long:
        return -1  # Sell signal
    else:
        return 0  # Hold


def train_and_evaluate_ml_model(historical_data: pd.DataFrame) -> Dict[str, Any]:
    ml_predictor = MLPredictor()
    accuracy, loss, performance = ml_predictor.train(historical_data)

    logger.info(f"ML model performance: Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
    logger.info(f"Detailed performance: {performance}")

    recent_data = historical_data.tail(1)
    prediction, confidence = ml_predictor.predict(recent_data)

    logger.info(f"Recent prediction: {'Up' if prediction == 1 else 'Down'}, Confidence: {confidence:.2f}")

    return {
        'accuracy': accuracy,
        'loss': loss,
        'performance': performance,
        'recent_prediction': prediction,
        'confidence': confidence
    }


def prepare_rl_state(data: pd.DataFrame) -> np.ndarray:
    # 상태를 준비하는 로직 구현
    # 예: 가격, 거래량, 기술적 지표 등을 포함
    state = np.array([
        data['close'].iloc[-1],
        data['volume'].iloc[-1],
        data['SMA'].iloc[-1],
        data['RSI'].iloc[-1],
        data['ATR'].iloc[-1]
    ]).reshape(1, -1)
    return state


def calculate_rl_reward(action: int, next_price: float, current_price: float) -> float:
    if action == 0:  # buy
        return (next_price - current_price) / current_price
    elif action == 1:  # sell
        return (current_price - next_price) / current_price
    else:  # hold
        return 0


def train_rl_agent(historical_data: pd.DataFrame, episodes: int = 100) -> RLAgent:
    state_size = 5  # 상태의 특성 수에 맞게 조정
    action_size = 3  # 매수, 매도, 홀드
    rl_agent = RLAgent(state_size, action_size)

    for episode in range(episodes):
        state = prepare_rl_state(historical_data.iloc[0:1])
        total_reward = 0

        for i in range(1, len(historical_data)):
            action = rl_agent.act(state)
            next_state = prepare_rl_state(historical_data.iloc[i:i + 1])
            reward = calculate_rl_reward(action, historical_data['close'].iloc[i], historical_data['close'].iloc[i - 1])
            done = i == len(historical_data) - 1

            rl_agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(rl_agent.memory) > rl_agent.batch_size:
                rl_agent.replay(rl_agent.batch_size)

        logger.info(f"Episode: {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}")

    return rl_agent