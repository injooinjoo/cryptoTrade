import numpy as np
import pandas as pd
import torch
from torch import nn
from datetime import datetime

from tradingview.fetch_data import update_data

# 1. Constants and Inputs
BUY = 1
SELL = -1
CLEAR = 0

# 기본 입력 변수들
base_k = 252
ema200_period = 200
ema_fast_period = 10
ema_slow_period = 20
swing_period = 10

k = int(np.floor(np.sqrt(base_k)))

# 2. kNN을 위한 변수 초기화
features = []
directions = []

# 3. 거래 상태
in_long_trade = False
in_short_trade = False
break_even_triggered = False
entry_price = 0.0
stop_loss = 0.0
take_profit = 0.0


# 4. EMA 계산 함수 (직접 구현)
def calculate_ema(data, period):
    ema = np.zeros_like(data)
    ema[0] = data[0]
    alpha = 2 / (period + 1)

    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

    return ema


# 5. RSI 계산 함수 (직접 구현)
def calculate_rsi(data, period):
    # 가격 변화 계산
    delta = np.diff(data)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.zeros_like(data)
    avg_loss = np.zeros_like(data)

    # 초기 평균 이득과 손실 계산
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])

    # 이동 평균 계산
    for i in range(period + 1, len(data)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period

    # avg_loss가 0인 경우 처리
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rs[avg_loss == 0] = np.inf  # 손실이 없으면 rs를 무한대로 설정
        rsi = 100 - (100 / (1 + rs))

    # NaN 값이 발생할 경우 RSI를 50으로 설정
    rsi[np.isnan(rsi)] = 50

    # 초기 값 설정
    rsi[:period] = 50

    return rsi


# 6. PyTorch 기반의 kNN 구현
class KNNClassifier(nn.Module):
    def __init__(self, k):
        super(KNNClassifier, self).__init__()
        self.k = k

    def forward(self, x, train_data, train_labels):
        # 유클리디안 거리 계산
        distances = torch.cdist(x.unsqueeze(0), train_data)
        # 거리 기반으로 가까운 k개 선택
        _, indices = torch.topk(distances, self.k, largest=False)
        topk_labels = train_labels[indices.squeeze()]
        # 가장 많은 레이블을 예측 값으로 사용
        prediction = torch.mode(topk_labels).values.item()
        return prediction


# 7. 전략 구현
def backtest_strategy(data, k):
    global entry_price, stop_loss, take_profit, in_long_trade, in_short_trade, break_even_triggered

    close_prices = data['close'].values

    # 지표 계산
    ema200 = calculate_ema(close_prices, ema200_period)
    ema_fast = calculate_ema(close_prices, ema_fast_period)
    ema_slow = calculate_ema(close_prices, ema_slow_period)
    rsi_values = calculate_rsi(close_prices, ema_slow_period)

    # kNN 학습 데이터 준비
    f1 = calculate_rsi(close_prices, ema_slow_period)
    f2 = calculate_rsi(close_prices, ema_fast_period)
    class_label = np.sign(np.diff(close_prices))

    train_data = []
    train_labels = []

    for i in range(1, len(f1)):
        features.append([f1[i], f2[i]])
        directions.append(class_label[i - 1])
        train_data.append([f1[i], f2[i]])
        train_labels.append(class_label[i - 1])

    # 데이터를 torch 텐서로 변환
    train_data = torch.tensor(train_data).float()
    train_labels = torch.tensor(train_labels).long()

    # kNN 모델 생성
    knn_model = KNNClassifier(k)

    # 예측을 위한 현재 값
    current_data = torch.tensor([f1[-1], f2[-1]]).float()

    # PyTorch 기반 예측
    prediction = knn_model(current_data, train_data, train_labels)

    # 거래 조건 정의
    price_above_ema200 = close_prices[-1] > ema200[-1]
    ribbon_above_ema200 = ema_fast[-1] > ema200[-1] and ema_slow[-1] > ema200[-1]
    ribbon_green = ema_fast[-1] > ema_slow[-1]
    price_pullback_long = close_prices[-1] > ema_slow[-1]
    ml_buy_signal = prediction == BUY
    rsi_oversold = rsi_values[-1] < 30

    long_condition = price_above_ema200 and ribbon_above_ema200 and ribbon_green and price_pullback_long and ml_buy_signal and rsi_oversold

    # Short 조건
    price_below_ema200 = close_prices[-1] < ema200[-1]
    ribbon_below_ema200 = ema_fast[-1] < ema200[-1] and ema_slow[-1] < ema200[-1]
    ribbon_red = ema_fast[-1] < ema_slow[-1]
    price_pullback_short = close_prices[-1] < ema_slow[-1]
    ml_sell_signal = prediction == SELL
    rsi_overbought = rsi_values[-1] > 70

    short_condition = price_below_ema200 and ribbon_below_ema200 and ribbon_red and price_pullback_short and ml_sell_signal and rsi_overbought

    # Long 포지션 진입
    if long_condition and not in_long_trade and not in_short_trade:
        entry_price = close_prices[-1]
        stop_loss = min(close_prices[-swing_period:])
        take_profit = entry_price + (entry_price - stop_loss) * 2
        in_long_trade = True
        break_even_triggered = False

    # Short 포지션 진입
    if short_condition and not in_short_trade and not in_long_trade:
        entry_price = close_prices[-1]
        stop_loss = max(close_prices[-swing_period:])
        take_profit = entry_price - (stop_loss - entry_price) * 2
        in_short_trade = True
        break_even_triggered = False

    # Long 포지션의 Break-Even 로직
    if in_long_trade and not break_even_triggered:
        if close_prices[-1] >= entry_price + (take_profit - entry_price) * 0.25:
            stop_loss = entry_price
            break_even_triggered = True

    # Short 포지션의 Break-Even 로직
    if in_short_trade and not break_even_triggered:
        if close_prices[-1] <= entry_price - (entry_price - take_profit) * 0.25:
            stop_loss = entry_price
            break_even_triggered = True

    # Long 포지션 종료 조건
    if in_long_trade and close_prices[-1] < ema200[-1]:
        in_long_trade = False  # 포지션 청산

    # Short 포지션 종료 조건
    if in_short_trade and close_prices[-1] > ema200[-1]:
        in_short_trade = False  # 포지션 청산

    # 결과 반환
    return {
        "long_condition": long_condition,
        "short_condition": short_condition,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "in_long_trade": in_long_trade,
        "in_short_trade": in_short_trade
    }


# 8. 데이터 불러오기 및 백테스트 실행
def load_data_from_csv(filename='crypto_data.csv'):
    update_data(filename, exchange_id='binance', symbol='BTC/USDT', timeframe='5m')
    try:
        data = pd.read_csv(filename, index_col='timestamp', parse_dates=True)
        if data.empty:
            print(f"No data fetched. Please check the data source or time range.")
            return None
        return data
    except FileNotFoundError:
        print(f"{filename} not found. Please make sure to run fetch_data.py first to collect data.")
        return None


if __name__ == "__main__":
    data = load_data_from_csv()
    if data is not None:
        result = backtest_strategy(data, k=k)
        print(result)
