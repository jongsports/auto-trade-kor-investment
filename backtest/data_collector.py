"""
백테스팅용 OHLCV 데이터 수집기.

KIS API에서 과거 데이터를 수집하고 로컬 CSV 캐시로 관리합니다.
캐시가 있으면 API 호출 없이 재사용합니다.
"""
import asyncio
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("backtest.data_collector")

CACHE_DIR = Path("data/backtest_cache")


class BacktestDataCollector:
    """
    백테스팅용 OHLCV 데이터 수집 및 캐시 관리.

    사용법:
        collector = BacktestDataCollector(api_client)
        data = collector.get_ohlcv("005930", "2023-01-01", "2023-12-31")
    """

    def __init__(self, api_client=None):
        """
        Args:
            api_client: AsyncKisAPI 인스턴스 (None이면 캐시 전용 모드)
        """
        self.api_client = api_client
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str) -> Path:
        return CACHE_DIR / f"{ticker}.csv"

    def _load_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(ticker)
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            df = df.sort_values("date").reset_index(drop=True)
            logger.debug(f"[{ticker}] 캐시 로드: {len(df)}행")
            return df
        except Exception as e:
            logger.warning(f"[{ticker}] 캐시 로드 실패: {e}")
            return None

    def _save_cache(self, ticker: str, df: pd.DataFrame) -> None:
        path = self._cache_path(ticker)
        df.to_csv(path, index=False)
        logger.debug(f"[{ticker}] 캐시 저장: {len(df)}행 → {path}")

    def get_ohlcv(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        OHLCV 데이터 반환 (캐시 우선, 없으면 KIS API 호출).

        Args:
            ticker: 종목코드 (예: "005930")
            start_date: 시작일 "YYYY-MM-DD"
            end_date: 종료일 "YYYY-MM-DD"
            force_refresh: True이면 캐시 무시하고 API 재호출

        Returns:
            pd.DataFrame: date/open/high/low/close/volume/amount 컬럼
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        cached = self._load_cache(ticker) if not force_refresh else None

        if cached is not None:
            # 캐시에 필요한 날짜 범위가 포함되어 있는지 확인
            cache_start = cached["date"].min()
            cache_end = cached["date"].max()

            # 충분한 초기 지표 계산을 위해 start_date보다 120일 이전 데이터까지 필요
            required_start = start_dt - timedelta(days=180)

            # 주말/공휴일로 캐시 마지막 날이 end_dt보다 최대 7일 이전일 수 있음
            if cache_start <= required_start and cache_end >= end_dt - timedelta(days=7):
                mask = (cached["date"] >= required_start) & (cached["date"] <= end_dt)
                result = cached[mask].reset_index(drop=True)
                logger.info(f"[{ticker}] 캐시 사용: {len(result)}행")
                return result

        # API 호출 필요
        if self.api_client is None:
            logger.error(f"[{ticker}] API 클라이언트 없음, 캐시도 없음")
            return pd.DataFrame()

        df = asyncio.run(self._fetch_from_api(ticker, start_date, end_date))
        if not df.empty:
            self._save_cache(ticker, df)

        if df.empty:
            return df

        required_start = start_dt - timedelta(days=180)
        mask = (df["date"] >= required_start) & (df["date"] <= end_dt)
        return df[mask].reset_index(drop=True)

    async def _fetch_from_api(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        KIS API에서 OHLCV 수집 (TR: FHKST03010100).

        이슈 #1 명세에 따라 inquire-daily-itemchartprice 엔드포인트 사용.
        지표 워밍업을 위해 start_date보다 200일 이전부터 수집합니다.
        """
        try:
            # 지표 계산 워밍업: 시작일 200일(달력) 이전부터 수집
            fetch_start = (
                pd.to_datetime(start_date) - timedelta(days=200)
            ).strftime("%Y%m%d")
            fetch_end = pd.to_datetime(end_date).strftime("%Y%m%d")

            df = await self.api_client.get_ohlcv_by_range(
                ticker=ticker,
                start_date=fetch_start,
                end_date=fetch_end,
                period_code="D",
            )

            if df.empty:
                logger.warning(f"[{ticker}] API 반환 데이터 없음")
                return pd.DataFrame()

            logger.info(f"[{ticker}] API 수집 완료: {len(df)}행")
            return df

        except Exception as e:
            logger.error(f"[{ticker}] API 수집 실패: {e}", exc_info=True)
            return pd.DataFrame()

    def batch_collect(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        여러 종목 OHLCV 일괄 수집.

        Returns:
            dict: {ticker: DataFrame}
        """
        result = {}
        total = len(tickers)

        for i, ticker in enumerate(tickers, 1):
            logger.info(f"[{i}/{total}] {ticker} 데이터 수집 중...")
            df = self.get_ohlcv(ticker, start_date, end_date, force_refresh)
            if not df.empty:
                result[ticker] = df
            else:
                logger.warning(f"[{ticker}] 데이터 없음 — 건너뜀")

        logger.info(f"배치 수집 완료: {len(result)}/{total}개 종목")
        return result

    def save_sample_data(self, ticker: str, df: pd.DataFrame) -> None:
        """테스트용 샘플 데이터 저장."""
        self._save_cache(ticker, df)

    @staticmethod
    def generate_sample_data(
        ticker: str = "005930",
        start_date: str = "2022-01-01",
        end_date: str = "2023-12-31",
        initial_price: float = 60000.0,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        테스트용 합성 OHLCV 데이터 생성 (API 없이 백테스트 검증용).

        랜덤워크 기반으로 현실감 있는 주가 데이터를 생성합니다.
        """
        import numpy as np

        rng = np.random.default_rng(seed)
        dates = pd.bdate_range(start=start_date, end=end_date)  # 영업일만

        n = len(dates)
        # 일간 수익률: 평균 0.03%, 표준편차 1.5% (코스피 단타 종목 수준)
        daily_returns = rng.normal(loc=0.0003, scale=0.015, size=n)
        close_prices = initial_price * (1 + daily_returns).cumprod()

        # 고가/저가/시가 생성
        intraday_range = rng.uniform(0.005, 0.03, size=n)
        high = close_prices * (1 + intraday_range * rng.uniform(0.3, 0.7, size=n))
        low = close_prices * (1 - intraday_range * rng.uniform(0.3, 0.7, size=n))
        open_prices = low + (high - low) * rng.uniform(0.2, 0.8, size=n)

        # 거래량: 기본 + 급등 이벤트
        base_volume = rng.integers(500_000, 2_000_000, size=n).astype(float)
        volume_spike = rng.choice([1.0, 2.5, 5.0], size=n, p=[0.85, 0.1, 0.05])
        volume = base_volume * volume_spike

        df = pd.DataFrame(
            {
                "date": dates,
                "open": open_prices.round(0).astype(int),
                "high": high.round(0).astype(int),
                "low": low.round(0).astype(int),
                "close": close_prices.round(0).astype(int),
                "volume": volume.astype(int),
                "amount": (close_prices * volume).astype(int),
            }
        )
        df["ticker"] = ticker
        return df.reset_index(drop=True)
