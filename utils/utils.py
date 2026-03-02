import logging
import os
import json
from datetime import datetime, timedelta, time as dt_time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

import config

logger = logging.getLogger("auto_trade.utils")


def save_to_json(data, filename):
    """데이터를 JSON 파일로 저장

    Args:
        data: 저장할 데이터
        filename (str): 파일명
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"JSON 저장 중 오류 발생: {str(e)}")
        return False


def load_from_json(filename):
    """JSON 파일에서 데이터 로드

    Args:
        filename (str): 파일명

    Returns:
        dict: 로드된 데이터
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"JSON 로드 중 오류 발생: {str(e)}")
        return None


def save_to_csv(df, filename):
    """데이터프레임을 CSV 파일로 저장

    Args:
        df (DataFrame): 저장할 데이터프레임
        filename (str): 파일명
    """
    try:
        df.to_csv(filename, index=False, encoding="utf-8")
        return True
    except Exception as e:
        logger.error(f"CSV 저장 중 오류 발생: {str(e)}")
        return False


def load_from_csv(filename):
    """CSV 파일에서 데이터프레임 로드

    Args:
        filename (str): 파일명

    Returns:
        DataFrame: 로드된 데이터프레임
    """
    try:
        df = pd.read_csv(filename, encoding="utf-8")
        return df
    except Exception as e:
        logger.error(f"CSV 로드 중 오류 발생: {str(e)}")
        return None


def calculate_returns(prices):
    """수익률 계산

    Args:
        prices (array-like): 가격 배열

    Returns:
        array: 수익률 배열
    """
    returns = np.diff(prices) / prices[:-1]
    return returns


def plot_stock_chart(ohlcv_data, ticker, save_path=None):
    """주식 차트 그리기

    Args:
        ohlcv_data (DataFrame): OHLCV 데이터
        ticker (str): 종목코드
        save_path (str): 저장 경로 (None이면 저장하지 않음)
    """
    try:
        # 데이터 준비
        df = ohlcv_data.copy()

        if "date" not in df.columns:
            logger.error("날짜 컬럼이 없습니다.")
            return

        # 그래프 설정
        fig, axs = plt.subplots(
            2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )

        # 가격 차트
        axs[0].plot(df["date"], df["close"], label="종가")

        if "ma5" in df.columns:
            axs[0].plot(df["date"], df["ma5"], label="5일선", linestyle="--")

        if "ma20" in df.columns:
            axs[0].plot(df["date"], df["ma20"], label="20일선", linestyle="-.")

        axs[0].set_title(f"{ticker} 가격 차트")
        axs[0].set_ylabel("가격")
        axs[0].grid(True)
        axs[0].legend()

        # 거래량 차트
        axs[1].bar(df["date"], df["volume"], label="거래량")
        axs[1].set_ylabel("거래량")
        axs[1].grid(True)

        # x축 날짜 포맷 설정
        date_form = DateFormatter("%Y-%m-%d")
        axs[1].xaxis.set_major_formatter(date_form)
        fig.autofmt_xdate()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"차트 저장 완료: {save_path}")
        else:
            plt.show()

        plt.close()

    except Exception as e:
        logger.error(f"차트 그리기 중 오류 발생: {str(e)}")


def calculate_indicators(df):
    """기술적 지표 계산

    Args:
        df (DataFrame): OHLCV 데이터

    Returns:
        DataFrame: 지표가 추가된 데이터프레임
    """
    # 이동평균선
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["ma60"] = df["close"].rolling(window=60).mean()

    # 볼린저 밴드
    df["ma20_std"] = df["close"].rolling(window=20).std()
    df["upper_band"] = df["ma20"] + (df["ma20_std"] * 2)
    df["lower_band"] = df["ma20"] - (df["ma20_std"] * 2)

    # MACD
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    return df


def calculate_strategy_performance(df, positions):
    """전략 성과 계산

    Args:
        df (DataFrame): OHLCV 데이터
        positions (DataFrame): 포지션 데이터 (진입/청산 정보)

    Returns:
        dict: 성과 지표
    """
    try:
        # 수익률 계산
        total_returns = []

        for _, position in positions.iterrows():
            entry_date = position["entry_date"]
            exit_date = position["exit_date"]
            entry_price = position["entry_price"]
            exit_price = position["exit_price"]

            # 수익률
            returns = (exit_price / entry_price) - 1
            total_returns.append(returns)

        if not total_returns:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_return": 0,
                "max_return": 0,
                "min_return": 0,
                "std_return": 0,
                "sharpe_ratio": 0,
            }

        # 성과 지표 계산
        total_trades = len(total_returns)
        win_trades = sum(1 for r in total_returns if r > 0)
        win_rate = win_trades / total_trades if total_trades > 0 else 0

        avg_return = np.mean(total_returns)
        max_return = np.max(total_returns)
        min_return = np.min(total_returns)
        std_return = np.std(total_returns)

        # 샤프 비율
        risk_free_rate = 0.01  # 연 1%로 가정
        daily_risk_free = (1 + risk_free_rate) ** (1 / 252) - 1
        sharpe_ratio = (
            (avg_return - daily_risk_free) / std_return if std_return > 0 else 0
        )

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "max_return": max_return,
            "min_return": min_return,
            "std_return": std_return,
            "sharpe_ratio": sharpe_ratio,
        }
    except Exception as e:
        logger.error(f"전략 성과 계산 중 오류 발생: {str(e)}")
        return {
            "total_trades": 0,
            "win_rate": 0,
            "avg_return": 0,
            "error": str(e),
        }


# 거래 시간 관리 관련 유틸리티 함수


def is_market_open():
    """현재 시장이 열려있는지 확인

    Returns:
        bool: 시장 개장 여부
    """
    # 주말 체크
    now = datetime.now()
    if now.weekday() >= 5:  # 5: 토요일, 6: 일요일
        return False

    # 공휴일 체크
    today_str = now.strftime("%Y%m%d")
    holidays = load_holidays()
    if today_str in holidays:
        return False

    # 시간 체크
    current_time = now.time()
    market_open = dt_time(9, 0)  # 09:00
    market_close = dt_time(15, 30)  # 15:30

    return market_open <= current_time <= market_close


def is_trading_time():
    """현재 매매 가능 시간인지 확인 (장 중 시간)

    Returns:
        bool: 매매 가능 여부
    """
    if not is_market_open():
        return False

    # 장 시작 직후 5분 제외 (동시호가 시간)
    now = datetime.now()
    current_time = now.time()
    after_opening = dt_time(9, 5)  # 09:05
    before_closing = dt_time(15, 20)  # 15:20

    return after_opening <= current_time <= before_closing


def is_regular_trading_hours():
    """정규장 시간인지 확인

    Returns:
        bool: 정규장 여부 (동시호가 시간 제외)
    """
    if not is_market_open():
        return False

    # 동시호가 시간 제외
    now = datetime.now()
    current_time = now.time()
    morning_auction_end = dt_time(9, 0)  # 09:00
    closing_auction_start = dt_time(15, 20)  # 15:20

    return morning_auction_end <= current_time <= closing_auction_start


def load_holidays():
    """공휴일 목록 로드

    Returns:
        list: 공휴일 목록 (YYYYMMDD 형식)
    """
    holiday_file = os.path.join(config.DATA_DIR, "holidays.json")

    if os.path.exists(holiday_file):
        holidays = load_from_json(holiday_file)
        if holidays:
            return holidays

    # 기본 공휴일 목록 (해마다 업데이트 필요)
    current_year = datetime.now().year
    default_holidays = [
        f"{current_year}0101",  # 신정
        f"{current_year}0301",  # 삼일절
        f"{current_year}0505",  # 어린이날
        f"{current_year}0606",  # 현충일
        f"{current_year}0815",  # 광복절
        f"{current_year}1003",  # 개천절
        f"{current_year}1009",  # 한글날
        f"{current_year}1225",  # 크리스마스
    ]

    # TODO: 설날, 추석 등 음력 기반 공휴일은 별도 계산 필요

    # 파일에 저장
    save_to_json(default_holidays, holiday_file)

    return default_holidays


def update_holidays_from_api(api_client):
    """한국투자증권 API를 통해 휴장일 정보 업데이트

    Args:
        api_client (KisAPI): 한국투자증권 API 클라이언트

    Returns:
        bool: 업데이트 성공 여부
    """
    try:
        # API로 휴장일 조회
        holiday_data = api_client.get_holidays()

        if not holiday_data:
            logger.warning("휴장일 정보를 가져오는데 실패했습니다.")
            return False

        # 휴장일 목록 추출
        holidays = [item["date"] for item in holiday_data]

        # 파일에 저장
        holiday_file = os.path.join(config.DATA_DIR, "holidays.json")
        save_to_json(holidays, holiday_file)

        logger.info(f"휴장일 정보 업데이트 완료: {len(holidays)}개")
        return True

    except Exception as e:
        logger.error(f"휴장일 정보 업데이트 중 오류 발생: {str(e)}")
        return False


def get_trading_time_status():
    """현재 거래 시간 상태 확인

    Returns:
        str: 거래 시간 상태
            - 'CLOSED': 장 마감 (주말, 공휴일 포함)
            - 'PRE_MARKET': 장 시작 전 (09:00 이전)
            - 'OPENING_AUCTION': 동시호가 (09:00~09:05)
            - 'REGULAR': 정규장 (09:05~15:20)
            - 'CLOSING_AUCTION': 동시호가 (15:20~15:30)
            - 'POST_MARKET': 장 마감 후 (15:30 이후)
    """
    now = datetime.now()

    # 주말 체크
    if now.weekday() >= 5:  # 5: 토요일, 6: 일요일
        return "CLOSED"

    # 공휴일 체크
    today_str = now.strftime("%Y%m%d")
    holidays = load_holidays()
    if today_str in holidays:
        return "CLOSED"

    # 시간 체크
    current_time = now.time()
    pre_market_end = dt_time(9, 0)  # 09:00
    opening_auction_end = dt_time(9, 5)  # 09:05
    closing_auction_start = dt_time(15, 20)  # 15:20
    post_market_start = dt_time(15, 30)  # 15:30

    if current_time < pre_market_end:
        return "PRE_MARKET"
    elif current_time < opening_auction_end:
        return "OPENING_AUCTION"
    elif current_time < closing_auction_start:
        return "REGULAR"
    elif current_time < post_market_start:
        return "CLOSING_AUCTION"
    else:
        return "POST_MARKET"
