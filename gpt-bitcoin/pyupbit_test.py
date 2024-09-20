import pyupbit
import pandas as pd
from datetime import datetime, timedelta

def check_pyupbit_data_structure():
    # 현재 시간부터 200개의 10분 간격 데이터 가져오기
    end_date = datetime.now()
    df = pyupbit.get_ohlcv("KRW-BTC", interval="minute10", to=end_date, count=200)

    print("1. DataFrame 기본 정보:")
    print(df.info())
    print("\n2. DataFrame 처음 5행:")
    print(df.head())
    print("\n3. DataFrame 컬럼:")
    print(df.columns)
    print("\n4. DataFrame 인덱스 정보:")
    print(df.index)
    print(f"인덱스 타입: {type(df.index)}")
    print(f"인덱스의 첫 번째 요소 타입: {type(df.index[0])}")
    print("\n5. DataFrame 데이터 타입:")
    print(df.dtypes)
    print("\n6. DataFrame 통계 정보:")
    print(df.describe())

    # 인덱스를 컬럼으로 변환
    df_reset = df.reset_index()
    print("\n7. 인덱스를 컬럼으로 변환한 후 DataFrame 정보:")
    print(df_reset.info())
    print("\n8. 변환 후 처음 5행:")
    print(df_reset.head())

check_pyupbit_data_structure()