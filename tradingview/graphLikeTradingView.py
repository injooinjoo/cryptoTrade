import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
from fetch_data import update_data
from datetime import timedelta

from tradingview.kNN import run_knn_strategy


def plot_tradingview_like(df, indicators=None):
    # 최근 8시간 데이터만 선택
    end_time = df.index[-1]
    start_time = end_time - timedelta(hours=8)
    df = df.loc[start_time:end_time]

    # 기본 스타일 설정
    mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='dotted', facecolor='#1e1e1e', edgecolor='#3f3f3f', figcolor='#1e1e1e', rc={'axes.edgecolor':'#3f3f3f'})

    # Y축 범위 설정
    price_range = df['Close'].max() - df['Close'].min()
    y_margin = price_range * 0.1  # 10% 여유 공간
    vmin = df['Close'].min() - y_margin
    vmax = df['Close'].max() + y_margin

    # 플롯 추가 설정
    add_plots = [mpf.make_addplot(df['Close'], color='dodgerblue', width=1.5)]

    if indicators is not None:
        for indicator, data in indicators.items():
            add_plots.append(mpf.make_addplot(data.loc[start_time:end_time], panel=0, secondary_y=False, ylabel=indicator))

    # Buy/Sell 신호 추가
    if 'Buy_Signal' in df.columns:
        buy_signals = df['Buy_Signal'].loc[start_time:end_time].dropna()
        if not buy_signals.empty:
            buy_data = df['Close'].where(df['Buy_Signal'].notna())
            add_plots.append(mpf.make_addplot(buy_data, type='scatter', markersize=100, marker='^', color='g'))

    if 'Sell_Signal' in df.columns:
        sell_signals = df['Sell_Signal'].loc[start_time:end_time].dropna()
        if not sell_signals.empty:
            sell_data = df['Close'].where(df['Sell_Signal'].notna())
            add_plots.append(mpf.make_addplot(sell_data, type='scatter', markersize=100, marker='v', color='r'))

    # 차트 그리기
    fig, axes = mpf.plot(df, type='line', style=s, addplot=add_plots, returnfig=True,
                         ylim=(vmin, vmax),
                         figsize=(12,8),
                         tight_layout=True,
                         show_nontrading=True)

    # 제목 추가
    plt.title('Crypto Price Chart (Last 8 Hours) with Indicators and Signals', fontsize=16, color='white')

    # Y축 레이블 색상 변경
    axes[0].tick_params(axis='y', colors='white')
    axes[0].tick_params(axis='x', colors='white')

    plt.show()

def prepare_data_for_plot(df):
    # mplfinance에 필요한 형식으로 데이터 준비
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # 소문자로 된 컬럼 이름을 대문자로 변경
    df.columns = [col.capitalize() for col in df.columns]

    # 필요한 컬럼이 없으면 더미 데이터로 채우기
    for col in required_columns:
        if col not in df.columns:
            if col == 'Volume':
                df[col] = np.random.randint(1000, 10000, size=len(df))
            else:
                df[col] = df['Close']  # Close 가격으로 채우기

    return df[required_columns]


if __name__ == "__main__":
    # 데이터 업데이트
    update_data()

    # 데이터 로드
    df = pd.read_csv('crypto_data.csv', index_col='timestamp', parse_dates=True)
    df = prepare_data_for_plot(df)

    # run_knn_strategy 호출하여 Buy/Sell 신호 가져오기
    start_date = pd.Timestamp('2024-06-01 00:00:00')
    stop_date = pd.Timestamp('2024-12-31 23:45:00')

    # kNN 전략 실행 후 신호 데이터 가져오기
    knn_signals = run_knn_strategy(df, start_date, stop_date)

    # kNN 신호를 원래 데이터프레임에 병합
    df['Buy_Signal'] = knn_signals['Buy_Signal']
    df['Sell_Signal'] = knn_signals['Sell_Signal']
    df['Clear_Signal'] = knn_signals['Clear_Signal']

    # RSI 계산 (실제 RSI 계산 로직으로 대체해야 함)
    df['RSI'] = pd.Series(np.random.randn(len(df)), index=df.index).rolling(window=14).mean()

    indicators = {'RSI': df['RSI']}
    plot_tradingview_like(df, indicators=indicators)
