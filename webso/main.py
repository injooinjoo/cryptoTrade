import json
import websocket
from datetime import datetime, timedelta
import threading
import logging
import sys
import io

# Windows 콘솔에서 한글 로그 메시지가 제대로 출력되도록 설정
# 이는 'UnicodeEncodeError'를 방지하기 위함입니다.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    handlers=[
        logging.FileHandler("backtest_log.log", encoding='utf-8'),  # 로그 파일을 utf-8로 설정
        logging.StreamHandler(sys.stdout)  # 콘솔 출력도 utf-8로 설정
    ]
)

# 가격 변동을 추적하기 위한 전역 변수
price_history = {}

# 최대 허용 수익률 (%) - 비정상적으로 높은 수익률을 방지
MAX_PROFIT_PCT = 1000  # 예: 1000% 초과 시 무시

class Strategy:
    """
    독립적인 투자 전략을 관리하는 클래스.
    각 전략은 자체적인 자본과 투자 상태를 관리합니다.
    """
    def __init__(self, name, time_limit_sec, buy_threshold_pct, sell_threshold_pct, investment_amount=25_000_000):
        self.name = name  # 전략 이름 (A, D, G, J)
        self.time_limit = timedelta(seconds=time_limit_sec)  # 시간 제한
        self.buy_threshold = buy_threshold_pct  # 매수 기준 변동률 (%)
        self.sell_threshold = sell_threshold_pct  # 매도 기준 변동률 (%)
        self.investment_amount = investment_amount  # 매수 시 투자 금액 (원)
        self.capital = 25_000_000  # 각 전략의 초기 자본 (원)
        self.invested = False  # 투자 상태 (True: 투자 중, False: 투자하지 않음)
        self.buy_price = None  # 매수가
        self.buy_time = None  # 매수 시각
        self.lock = threading.Lock()  # 투자 상태 접근 동기화

    def evaluate(self, ticker, current_time, current_price, price_history):
        """
        특정 티커의 가격 변동을 평가하여 매수/매도 조건을 확인하고 실행합니다.
        """
        if ticker not in price_history:
            return

        # 해당 전략의 시간 제한 내 가격 리스트
        relevant_prices = [
            (time, price) for time, price in price_history[ticker]
            if current_time - time <= self.time_limit
        ]

        if not relevant_prices:
            return

        first_time, first_price = relevant_prices[0]

        # 가격 변동률 계산 전에 첫 가격이 0이 아닌지 확인
        if first_price == 0:
            logging.warning(f"{self.name} 전략: {ticker}의 첫 가격이 0입니다. 건너뜁니다.")
            return

        # 현재 가격과 첫 가격을 기반으로 변동률 계산
        price_change = (current_price - first_price) / first_price * 100

        # 매수 조건 확인
        if not self.invested:
            if price_change >= self.buy_threshold and (current_time - first_time) <= self.time_limit:
                with self.lock:
                    # 투자 가능한 자본 확인
                    if self.capital < self.investment_amount:
                        logging.info(f"{self.name} 전략: 자본 부족으로 {ticker} 매수 불가.")
                        return

                    # 자본 할당 및 투자 상태 업데이트
                    self.capital -= self.investment_amount
                    self.invested = True
                    self.buy_price = current_price
                    self.buy_time = current_time

                logging.info(
                    f"{self.name} 전략: {ticker} {self.buy_threshold}% 이상 상승 ({price_change:.2f}%) -> 매수. "
                    f"매수가: {self.buy_price:.2f}원, 투자 금액: {self.investment_amount:,.2f}원, "
                    f"남은 자본: {self.capital:,.2f}원"
                )
        else:
            # 매도 조건 확인
            target_price = self.buy_price * (1 + self.sell_threshold / 100)
            if current_price >= target_price:
                # 수익률 계산
                profit_pct = (current_price - self.buy_price) / self.buy_price * 100

                # 비정상적으로 높은 수익률 방지
                if profit_pct > MAX_PROFIT_PCT:
                    logging.warning(
                        f"{self.name} 전략: {ticker}의 수익률 {profit_pct:.2f}%이 최대 허용 수익률 {MAX_PROFIT_PCT}%을 초과. 매도하지 않습니다."
                    )
                    return

                # 수익 금액 계산
                profit = self.investment_amount * (profit_pct / 100)

                with self.lock:
                    # 자본 업데이트
                    self.capital += self.investment_amount + profit
                    # 투자 상태 초기화
                    self.invested = False
                    self.buy_price = None
                    self.buy_time = None

                logging.info(
                    f"{self.name} 전략: {ticker} {self.sell_threshold}% 이상 상승 ({profit_pct:.2f}%) -> 매도. "
                    f"수익률: {profit_pct:.2f}%, 수익 금액: {profit:,.2f}원, 총 자본: {self.capital:,.2f}원"
                )

def initialize_strategies():
    """
    4개의 독립적인 전략(A, D, G, J)을 초기화합니다.
    각 전략은 고유한 설정을 가지고 있으며, 고정된 투자 금액을 사용합니다.
    """
    strategies = []
    strategy_labels = ['A', 'D', 'G', 'J']  # 4개의 전략
    time_limits = [10, 20, 30, 40]  # 초 단위
    buy_thresholds = [0.5, 1, 2]    # % 단위
    sell_threshold = 3             # 모든 전략에 대해 동일한 매도 기준 (%)

    label_index = 0
    for label in strategy_labels:
        # 각 전략마다 다른 시간 제한과 매수 기준을 설정
        if label == 'A':
            time_limit = 10
            buy_threshold = 0.5
        elif label == 'D':
            time_limit = 20
            buy_threshold = 1
        elif label == 'G':
            time_limit = 30
            buy_threshold = 2
        elif label == 'J':
            time_limit = 40
            buy_threshold = 1  # 예시로 설정, 필요에 따라 조정 가능
        else:
            time_limit = 10
            buy_threshold = 0.5

        strategy = Strategy(
            name=label,
            time_limit_sec=time_limit,
            buy_threshold_pct=buy_threshold,
            sell_threshold_pct=sell_threshold,
            investment_amount=25_000_000  # 각 전략당 고정 투자 금액 (25,000,000원)
        )
        strategies.append(strategy)

    return strategies

# WebSocket 메시지 처리 함수
def on_message(ws, message):
    global strategies, price_history

    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        logging.error(f"JSON 디코딩 에러: {message}")
        return

    current_time = datetime.now()

    if 'trade_price' in data and 'code' in data:
        ticker = data['code']  # 티커 정보
        price = data['trade_price']  # 실시간 가격 정보

        # trade_price가 올바른 타입인지 확인
        if not isinstance(price, (int, float)):
            logging.error(f"{ticker}의 trade_price가 숫자가 아닙니다: {price}")
            return

        # 티커별로 가격 리스트 유지
        if ticker not in price_history:
            price_history[ticker] = []

        # 현재 가격과 시간을 리스트에 저장
        price_history[ticker].append((current_time, price))

        # 오래된 데이터 제거 (최대 시간 제한에 따라)
        max_time_limit = max(strategy.time_limit for strategy in strategies)
        price_history[ticker] = [
            (time, p) for time, p in price_history[ticker]
            if current_time - time <= max_time_limit
        ]

        # 가격 변동 분석 및 로그
        if len(price_history[ticker]) > 1:
            first_time, first_price = price_history[ticker][0]
            last_time, last_price = price_history[ticker][-1]

            # 첫 가격이 0이 아닌지 확인
            if first_price == 0:
                logging.warning(f"{ticker}의 첫 가격이 0입니다. 가격 변동을 계산하지 않습니다.")
            else:
                price_change = (last_price - first_price) / first_price * 100
                if price_change >= 0.5:
                    logging.info(f"{ticker} 가격 변동: {price_change:.2f}%")

        # 각 전략에 대해 조건 평가
        for strategy in strategies:
            strategy.evaluate(ticker, current_time, price, price_history)
    else:
        logging.info(f"받은 데이터에 'trade_price' 또는 'code' 없음: {data}")

def on_error(ws, error):
    logging.error(f"에러 발생: {error}")

def on_close(ws):
    logging.info("WebSocket 연결 종료")

def on_open(ws):
    """
    WebSocket 연결 시 구독 메시지를 전송하여 실시간 체결 정보를 받습니다.
    """
    subscribe_msg = [
        {"ticket": "test"},
        {
            "type": "ticker",
            "codes": [
                'KRW-BTC', 'KRW-ETH', 'KRW-NEO', 'KRW-MTL', 'KRW-XRP', 'KRW-ETC', 'KRW-SNT', 'KRW-WAVES',
                'KRW-XEM', 'KRW-QTUM', 'KRW-LSK', 'KRW-STEEM', 'KRW-XLM', 'KRW-ARDR', 'KRW-ARK', 'KRW-STORJ',
                'KRW-GRS', 'KRW-ADA', 'KRW-SBD', 'KRW-POWR', 'KRW-BTG', 'KRW-ICX', 'KRW-EOS', 'KRW-TRX', 'KRW-SC',
                'KRW-ONT', 'KRW-ZIL', 'KRW-POLYX', 'KRW-ZRX', 'KRW-LOOM', 'KRW-BCH', 'KRW-BAT', 'KRW-IOST',
                'KRW-CVC', 'KRW-IQ', 'KRW-IOTA', 'KRW-HIFI', 'KRW-ONG', 'KRW-GAS', 'KRW-UPP', 'KRW-ELF',
                'KRW-KNC', 'KRW-BSV', 'KRW-THETA', 'KRW-QKC', 'KRW-BTT', 'KRW-MOC', 'KRW-TFUEL', 'KRW-MANA',
                'KRW-ANKR', 'KRW-AERGO', 'KRW-ATOM', 'KRW-TT', 'KRW-GAME2', 'KRW-MBL', 'KRW-WAXP', 'KRW-HBAR',
                'KRW-MED', 'KRW-MLK', 'KRW-STPT', 'KRW-ORBS', 'KRW-VET', 'KRW-CHZ', 'KRW-STMX', 'KRW-DKA',
                'KRW-HIVE', 'KRW-KAVA', 'KRW-AHT', 'KRW-LINK', 'KRW-XTZ', 'KRW-BORA', 'KRW-JST', 'KRW-CRO',
                'KRW-TON', 'KRW-SXP', 'KRW-HUNT', 'KRW-DOT', 'KRW-MVL', 'KRW-STRAX', 'KRW-AQT', 'KRW-GLM',
                'KRW-META', 'KRW-FCT2', 'KRW-CBK', 'KRW-SAND', 'KRW-HPO', 'KRW-DOGE', 'KRW-STRIKE', 'KRW-PUNDIX',
                'KRW-FLOW', 'KRW-AXS', 'KRW-STX', 'KRW-XEC', 'KRW-SOL', 'KRW-POL', 'KRW-AAVE', 'KRW-1INCH',
                'KRW-ALGO', 'KRW-NEAR', 'KRW-AVAX', 'KRW-T', 'KRW-CELO', 'KRW-GMT', 'KRW-APT', 'KRW-SHIB',
                'KRW-MASK', 'KRW-ARB', 'KRW-EGLD', 'KRW-SUI', 'KRW-GRT', 'KRW-BLUR', 'KRW-IMX', 'KRW-SEI',
                'KRW-MINA', 'KRW-CTC', 'KRW-ASTR', 'KRW-ID', 'KRW-PYTH', 'KRW-MNT', 'KRW-AKT', 'KRW-ZETA',
                'KRW-AUCTION', 'KRW-STG', 'KRW-BEAM', 'KRW-TAIKO', 'KRW-USDT', 'KRW-ONDO', 'KRW-ZRO', 'KRW-BLAST',
                'KRW-JUP', 'KRW-ENS', 'KRW-G', 'KRW-PENDLE', 'KRW-ATH', 'KRW-USDC', 'KRW-UXLINK', 'KRW-BIGTIME',
                'KRW-CKB'
            ],
            "isOnlyRealtime": True  # 실시간 체결 정보만 받기
        }
    ]
    ws.send(json.dumps(subscribe_msg))
    logging.info("WebSocket 구독 메시지 전송 완료")

if __name__ == "__main__":
    # 전략 초기화
    strategies = initialize_strategies()
    logging.info(f"초기화된 전략 수: {len(strategies)}")
    for strategy in strategies:
        logging.info(
            f"전략 {strategy.name}: 시간 제한 {strategy.time_limit.seconds}초, "
            f"매수 기준 {strategy.buy_threshold}%, 매도 기준 {strategy.sell_threshold}%, "
            f"투자 금액: {strategy.investment_amount:,}원"
        )

    # WebSocket 연결 설정
    websocket_url = "wss://api.upbit.com/websocket/v1"
    ws = websocket.WebSocketApp(
        websocket_url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.on_open = on_open
    ws.run_forever()
