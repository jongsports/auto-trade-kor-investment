import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime

from typing import List, Dict, Any, Optional
import config
from core.trader_api import AsyncKisAPI
from data.async_news_analyzer import AsyncNewsAnalyzer

logger = logging.getLogger("auto_trade.stock_screener")


class AsyncStockScreener:
    """주식 스크리닝 클래스 — 모멘텀 단타 최적화 (Async Version)"""

    def __init__(self, api_client: AsyncKisAPI):
        self.api_client = api_client
        self.market_codes = {"KOSPI": "J", "KOSDAQ": "K"}
        self.candidate_stocks = []

        # 스크리닝 설정
        self.momentum_days = config.MOMENTUM_DAYS
        self.min_gap_up = config.MIN_GAP_UP
        self.min_volume_ratio = config.MIN_VOLUME_RATIO
        self.min_amount_ratio = config.MIN_AMOUNT_RATIO
        self.min_ma5_ratio = config.MIN_MA5_RATIO

        self.news_analyzer = None

    # ------------------------------------------------------------------
    # 시장 종목 조회
    # ------------------------------------------------------------------

    async def _affordable_max_price(self) -> int:
        """계좌 규모로 1주라도 살 수 있는 가격 상한.

        사이징은 종목당 (max_stock_ratio × 0.3~1.5)를 배정하므로 상한은 1.5배 기준이다.
        이보다 비싼 종목은 진입 단계에서 반드시 탈락하니 스크리닝할 이유가 없다.
        조회 실패 시 0 (상한 없음) — 진입 단계가 한도를 다시 확인한다.
        """
        account = await self.api_client.get_account_summary()
        equity = account.get("total_evaluated_amount", 0) if account else 0
        return int(equity * config.MAX_STOCK_RATIO * 1.5)

    async def get_market_stocks(self, market="KOSPI", max_price: int = 0) -> List[str]:
        """거래대금 상위 종목 (시장당 최대 30건 — KIS 순위분석 1회 응답 상한)."""
        code = {"KOSPI": "0001", "KOSDAQ": "1001"}.get(market)
        if code is None:
            return []
        return await self.api_client.get_top_market_stocks(
            code, min_price=config.MIN_PRICE, max_price=max_price)

    async def get_volume_surge_stocks(self) -> List[tuple]:
        """장중 거래량 급증 종목 (거래증가율 순).

        Returns:
            [(ticker, market), ...] 형태. 연속시그널 스캐너가 기대하는 포맷.
        """
        max_price = await self._affordable_max_price()
        pairs: List[tuple] = []
        for market, code in (("KOSPI", "0001"), ("KOSDAQ", "1001")):
            tickers = await self.api_client.get_volume_surge_stocks(
                code, min_price=config.MIN_PRICE, max_price=max_price)
            pairs.extend((t, market) for t in tickers)
        return pairs

    # ------------------------------------------------------------------
    # 당일 봉 처리
    # ------------------------------------------------------------------

    @staticmethod
    def _completed_bars(ohlcv: pd.DataFrame, today=None) -> pd.DataFrame:
        """오늘 날짜의 (진행 중인) 봉을 뺀 일봉."""
        today = pd.Timestamp(today or datetime.now().date())
        return ohlcv[pd.to_datetime(ohlcv["date"]).dt.normalize() < today]

    @classmethod
    def _with_today_bar(cls, ohlcv: pd.DataFrame, price_data: Dict[str, Any], today=None) -> pd.DataFrame:
        """실시간 시세로 당일 봉을 만들어 붙인다.

        KIS 일봉 응답이 당일 봉을 포함하든 아니든 결과가 같도록, 기존 당일 행을 지운
        뒤 하나만 붙인다. 예전에는 무조건 append 해서 당일 봉이 두 번 들어갈 수 있었다.
        """
        today = pd.Timestamp(today or datetime.now().date())
        row = {
            "date": today,
            "open": price_data["open"],
            "high": price_data["high"],
            "low": price_data["low"],
            "close": price_data["price"],
            "volume": price_data["volume"],
            "amount": price_data["amount"],
        }
        return pd.concat([cls._completed_bars(ohlcv, today), pd.DataFrame([row])], ignore_index=True)

    @staticmethod
    def _session_elapsed_fraction(now: Optional[datetime] = None) -> float:
        """정규장(09:00~15:30) 중 경과 비율. 장 시작 전 0, 마감 후 1."""
        now = now or datetime.now()
        elapsed = (now.hour * 60 + now.minute) - 9 * 60
        return min(1.0, max(0.0, elapsed / 390))

    # ------------------------------------------------------------------
    # 시초가 갭 검증
    # ------------------------------------------------------------------

    async def validate_opening_candidates(self, candidates: list) -> list:
        """09:05 장 시작 직후 시초가 갭 필터링.

        갭 하락 > OPENING_GAP_DOWN_THRESHOLD(3%) 종목 제외 — 당일 악재 반영.
        갭 상승 > OPENING_GAP_UP_THRESHOLD(5%) 종목 제외 — 추격 매수 방지.
        """
        gap_down_limit = getattr(config, "OPENING_GAP_DOWN_THRESHOLD", 0.03)
        gap_up_limit   = getattr(config, "OPENING_GAP_UP_THRESHOLD",   0.05)
        validated = []

        for c in candidates:
            ticker = c.get("ticker", "")
            if not ticker:
                continue
            try:
                ohlcv = await self.api_client.get_ohlcv(ticker, count=5)
                price_data = await self.api_client.get_current_price(ticker)
                completed = self._completed_bars(ohlcv) if not ohlcv.empty else ohlcv
                if completed.empty or not price_data or not price_data.get("open"):
                    logger.info(f"[갭 검증 불가] {ticker}: 시세 없음 — 제외")
                    continue

                prev_close = float(completed["close"].iloc[-1])
                today_open = float(price_data["open"])

                if prev_close <= 0:
                    continue

                gap = (today_open - prev_close) / prev_close

                if gap < -gap_down_limit:
                    logger.info(f"[갭하락 제외] {ticker}: gap={gap:.2%} (< -{gap_down_limit:.0%})")
                    continue
                if gap > gap_up_limit:
                    logger.info(f"[갭상승 추격방지] {ticker}: gap={gap:.2%} (> {gap_up_limit:.0%})")
                    continue

                c = c.copy()
                c["opening_gap"] = round(gap, 4)
                validated.append(c)

            except Exception as e:
                # 갭을 확인하지 못한 종목은 사지 않는다.
                logger.error(f"[갭 검증 오류] {ticker}: {e} — 제외")

        logger.info(f"[갭 필터] {len(candidates)}종목 → {len(validated)}종목 통과")
        return validated

    # ------------------------------------------------------------------
    # 기술적 지표 계산
    # ------------------------------------------------------------------

    def calculate_technical_indicators(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """
        OHLCV DataFrame에 기술적 지표 컬럼을 추가하여 반환.

        기존 지표: MA5/20/60, disparity_ma5, volume_ma20/amount_ma20, RSI14,
                   MACD/MACD Signal, BB upper/std
        추가 지표: MA120, BB lower, MACD Histogram, Stochastic(5,3,3),
                   ADX(14), ATR(14)/ATR ratio, OBV/OBV slope, VWAP proxy
        """
        df = ohlcv_data.copy()

        # ── 이동평균 ─────────────────────────────────────────────────────
        df["ma5"]   = df["close"].rolling(window=5).mean()
        df["ma20"]  = df["close"].rolling(window=20).mean()
        df["ma60"]  = df["close"].rolling(window=60).mean()
        df["ma120"] = df["close"].rolling(window=120).mean()  # 기관 참조선

        df["disparity_ma5"] = (df["close"] / df["ma5"] - 1) * 100

        # ── 거래량/거래대금 ───────────────────────────────────────────────
        df["volume_ma20"] = df["volume"].rolling(window=20).mean()
        df["amount_ma20"] = df["amount"].rolling(window=20).mean()

        # ── RSI(14) ──────────────────────────────────────────────────────
        delta = df["close"].diff()
        gain  = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["rsi14"] = 100 - (100 / (1 + rs))

        # ── MACD (12/26/9) ───────────────────────────────────────────────
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"]        = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]  # MACD Histogram

        # ── 볼린저 밴드 (20, 2σ) ─────────────────────────────────────────
        df["bb_std"]   = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["ma20"] + (df["bb_std"] * 2)
        df["bb_lower"] = df["ma20"] - (df["bb_std"] * 2)

        # ── Stochastic (5, 3, 3) — 한국 단타 핵심 지표 ───────────────────
        rolling_low5  = df["low"].rolling(window=5).min()
        rolling_high5 = df["high"].rolling(window=5).max()
        stoch_range   = rolling_high5 - rolling_low5
        df["stoch_k"] = np.where(
            stoch_range > 0,
            100.0 * (df["close"] - rolling_low5) / stoch_range,
            50.0   # 가격 변동 없을 때 중립값
        )
        df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

        # ── ATR(14) — Wilder EWM ─────────────────────────────────────────
        # Wilder 평활: S_i = S_{i-1} * 13/14 + TR_i * 1/14 → ewm(alpha=1/14)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"]  - df["close"].shift(1)).abs()
        ], axis=1).max(axis=1)
        df["atr14"]     = tr.ewm(alpha=1 / 14, adjust=False).mean()   # Wilder ATR (곱셈 없음)
        df["atr_ratio"] = (df["atr14"] / df["close"].replace(0, np.nan)) * 100

        # ── ADX(14) + DI+/DI- — Wilder EWM ──────────────────────────────
        plus_dm_raw  = df["high"].diff()
        minus_dm_raw = -df["low"].diff()
        plus_dm  = plus_dm_raw.where((plus_dm_raw  > minus_dm_raw) & (plus_dm_raw  > 0), 0.0)
        minus_dm = minus_dm_raw.where((minus_dm_raw > plus_dm_raw) & (minus_dm_raw > 0), 0.0)

        # 분자/분모 모두 같은 스무딩 → 비율 계산 시 배수 상쇄됨
        smooth_atr14    = tr.ewm(alpha=1 / 14, adjust=False).mean()
        smooth_plus_dm  = plus_dm.ewm(alpha=1 / 14, adjust=False).mean()
        smooth_minus_dm = minus_dm.ewm(alpha=1 / 14, adjust=False).mean()

        df["di_plus"]  = 100 * smooth_plus_dm  / smooth_atr14.replace(0, np.nan)
        df["di_minus"] = 100 * smooth_minus_dm / smooth_atr14.replace(0, np.nan)
        di_sum  = (df["di_plus"] + df["di_minus"]).replace(0, np.nan)
        dx      = (df["di_plus"] - df["di_minus"]).abs() / di_sum * 100
        df["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

        # ── OBV (On-Balance Volume) ───────────────────────────────────────
        price_chg = df["close"].diff()
        obv_dir   = np.where(price_chg > 0, 1, np.where(price_chg < 0, -1, 0))
        df["obv"]       = (obv_dir * df["volume"]).cumsum()
        df["obv_slope"] = df["obv"].diff(5)   # 5일 OBV 기울기

        # ── VWAP proxy (5일 rolling — 일봉 기준 근사치) ──────────────────
        vol_sum    = df["volume"].rolling(5).sum().replace(0, np.nan)
        df["vwap_proxy"] = df["amount"].rolling(5).sum() / vol_sum

        return df

    # ------------------------------------------------------------------
    # 기존 Check 메서드 (유지)
    # ------------------------------------------------------------------

    def check_momentum(self, df: pd.DataFrame) -> bool:
        """MACD 상향 + RSI 안전구간 + 볼린저밴드 상단 근접."""
        if len(df) < 20:
            return False
        is_macd_bullish  = df["macd"].iloc[-1] > df["macd_signal"].iloc[-1]
        is_rsi_safe      = 40 <= df["rsi14"].iloc[-1] <= 70
        is_breaking_bb   = df["close"].iloc[-1] >= (df["bb_upper"].iloc[-1] * 0.98)
        return is_macd_bullish and is_rsi_safe and is_breaking_bb

    def check_volume_surge(self, df: pd.DataFrame, elapsed_fraction: float = 1.0) -> bool:
        """거래량이 20일 MA의 2배 이상.

        마지막 봉이 진행 중이면 누적 거래량을 하루치로 환산해 비교한다. 환산 없이
        10:30 의 누적 거래량을 하루치 평균과 비교하면 오전에는 거의 통과할 수 없다.
        장 초반 과대 환산을 막기 위해 경과 비율 하한을 20%로 둔다.
        """
        if len(df) < 21:
            return False
        vol_ma20_prev = df["volume_ma20"].iloc[-2]
        if not vol_ma20_prev or pd.isna(vol_ma20_prev):
            return False
        projected = df["volume"].iloc[-1] / max(elapsed_fraction, 0.2)
        return projected / vol_ma20_prev >= 2.0

    def check_moving_average(self, df: pd.DataFrame) -> bool:
        """정배열 (종가 > MA20 > MA60) + 이격도 과도 확장 아님."""
        if len(df) < 60:
            return False
        is_above_ma20       = df["close"].iloc[-1] > df["ma20"].iloc[-1]
        is_ma20_above_ma60  = df["ma20"].iloc[-1] > df["ma60"].iloc[-1]
        is_disparity_ok     = df["disparity_ma5"].iloc[-1] <= 110
        return is_above_ma20 and is_ma20_above_ma60 and is_disparity_ok

    # ------------------------------------------------------------------
    # 신규 Check 메서드 (6개)
    # ------------------------------------------------------------------

    def check_stochastic(self, df: pd.DataFrame) -> bool:
        """
        Stochastic K%가 D% 상향돌파 (과매도 구간 <40에서 올라옴).
        한국 단타 트레이더가 가장 많이 참고하는 단기 모멘텀 지표.
        """
        if len(df) < 8 or "stoch_k" not in df.columns:
            return False
        try:
            k_now,  d_now  = df["stoch_k"].iloc[-1], df["stoch_d"].iloc[-1]
            k_prev, d_prev = df["stoch_k"].iloc[-2], df["stoch_d"].iloc[-2]

            if any(pd.isna(v) for v in [k_now, d_now, k_prev, d_prev]):
                return False

            crossed_up   = (k_prev <= d_prev) and (k_now > d_now)   # K가 D 상향돌파
            was_oversold = df["stoch_k"].iloc[-5:-1].min() < 40      # 최근 과매도 경험
            both_rising  = (k_now > k_prev) and (d_now > d_prev)     # 양선 동반 상승
            return crossed_up and was_oversold and both_rising
        except Exception:
            return False

    def check_adx(self, df: pd.DataFrame) -> bool:
        """
        ADX > 20 AND DI+ > DI-: 상승 추세 확인 (횡보장 필터).
        ADX가 낮으면 변동성 없는 횡보 → 단타에 불리.
        """
        if len(df) < 28 or "adx" not in df.columns:
            return False
        try:
            adx      = df["adx"].iloc[-1]
            di_plus  = df["di_plus"].iloc[-1]
            di_minus = df["di_minus"].iloc[-1]
            if any(pd.isna(v) for v in [adx, di_plus, di_minus]):
                return False
            return (adx > 20) and (di_plus > di_minus)
        except Exception:
            return False

    def check_obv_trend(self, df: pd.DataFrame) -> bool:
        """
        OBV 기울기 > 0 AND 가격도 5일간 상승: 매집 확인.
        OBV와 가격 다이버전스 = 분산 신호 → 진입 금지.
        """
        if len(df) < 10 or "obv_slope" not in df.columns:
            return False
        try:
            obv_slope    = df["obv_slope"].iloc[-1]
            price_chg_5d = df["close"].iloc[-1] - df["close"].iloc[-6]
            if pd.isna(obv_slope):
                return False
            return (obv_slope > 0) and (price_chg_5d > 0)
        except Exception:
            return False

    def check_gap_up(self, df: pd.DataFrame) -> float:
        """
        당일 갭업 비율(%) 반환. 갭없음 또는 데이터 부족 시 0.0.
        오늘 시가 > 전일 종가 비율로 계산.
        """
        if len(df) < 2:
            return 0.0
        try:
            today_open      = df["open"].iloc[-1]
            yesterday_close = df["close"].iloc[-2]
            if yesterday_close <= 0:
                return 0.0
            return (today_open - yesterday_close) / yesterday_close * 100
        except Exception:
            return 0.0

    def check_vwap_position(self, df: pd.DataFrame) -> bool:
        """종가 > VWAP proxy: 당일 매수세 우위 확인."""
        if len(df) < 5 or "vwap_proxy" not in df.columns:
            return False
        try:
            vwap  = df["vwap_proxy"].iloc[-1]
            close = df["close"].iloc[-1]
            if pd.isna(vwap) or vwap <= 0:
                return False
            return close > vwap
        except Exception:
            return False

    def check_atr_range(self, df: pd.DataFrame) -> bool:
        """
        ATR ratio 1.5%~4%: 단타 적정 변동성 구간.
        너무 낮으면 수익 기회 없음, 너무 높으면 리스크 과도.
        """
        if len(df) < 14 or "atr_ratio" not in df.columns:
            return False
        try:
            atr_ratio = df["atr_ratio"].iloc[-1]
            if pd.isna(atr_ratio):
                return False
            return 1.5 <= atr_ratio <= 4.0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 공시 리스크 (유지)
    # ------------------------------------------------------------------

    async def check_disclosure_risk(self, ticker: str) -> bool:
        return True

    # ------------------------------------------------------------------
    # 세션별 진입 임계값
    # ------------------------------------------------------------------

    def get_entry_threshold(self, is_overnight_window: bool = False) -> int:
        """
        현재 거래 세션에 따라 최소 진입 점수 반환.

        오버나이트 창은 낮은 임계값이지만 수급(order_flow) 필수 조건이 별도 부과됨.
        """
        if is_overnight_window:
            return 65  # 승률 18% → 조건 대폭 강화 (기존 45)

        now_str = datetime.now().strftime("%H:%M")

        if "09:00" <= now_str < "10:00":
            return 60   # 모닝 러시: 강한 모멘텀만
        elif "14:00" <= now_str < "15:10":
            return 50   # 장마감 전: 후반 모멘텀 포착
        else:
            return 55   # 일반 시간대

    # ------------------------------------------------------------------
    # 100점 Confluence 스코어링 시스템
    # ------------------------------------------------------------------

    def calculate_stock_score(
        self,
        ticker: str,
        ohlcv_data: pd.DataFrame,
        investor_trend: dict,
        is_overnight_window: bool = False,
        is_intraday: bool = False,
        intraday_data: Dict[str, Any] = None,
        news_score: float = 0.0,
        elapsed_fraction: float = 1.0,
    ) -> dict:
        """
        100점 만점 모멘텀 Confluence 스코어링.

        카테고리별 배점:
          기술적 지표  max 40 (MACD+Histogram 12 / Stochastic 10 / ADX 8 / OBV 10)
          거래량 품질  max 20 (거래량급증 10 / MA20 위 5 / ATR범위 5)
          수급 (외기관) max 30 (외국인 15 / 기관 15)
          뉴스          max 10
          오버나이트보너스 max +20 (야간 창 한정)

        Returns:
            dict: {total, technical, volume, order_flow, news, overnight_bonus}
        """
        # 당일 봉 병합은 호출부(_process_ticker)가 _with_today_bar 로 끝낸 상태다.
        df = self.calculate_technical_indicators(ohlcv_data)
        if len(df) < 5: return {"total": 0}

        technical_score  = 0   # max 40
        volume_score     = 0   # max 20
        order_flow_score = 0   # max 30
        # 비대칭 가중 (2026-05-17): 부정 뉴스가 가격에 더 큰 영향 → -10 ~ +5
        news_pts         = max(-10, min(5, int(news_score)))
        overnight_bonus  = 0
        intraday_bonus   = 0

        # ── CATEGORY 1: 기술적 지표 (max 40) ─────────────────────────────

        # MACD 상향 + Histogram 양수 & 증가 (12pts, 부분점수 6pts)
        if len(df) >= 26:
            macd_bullish  = df["macd"].iloc[-1] > df["macd_signal"].iloc[-1]
            hist_positive = df["macd_hist"].iloc[-1] > 0
            hist_growing  = df["macd_hist"].iloc[-1] > df["macd_hist"].iloc[-2]
            if macd_bullish and hist_positive and hist_growing:
                technical_score += 12
            elif macd_bullish and hist_positive:
                technical_score += 6   # 상승세이지만 가속도 아직

        # Stochastic K>D 과매도 돌파 (10pts)
        if self.check_stochastic(df):
            technical_score += 10

        # ADX>20, DI+>DI- 추세 확인 (8pts)
        if self.check_adx(df):
            technical_score += 8

        # OBV 매집 확인 (10pts)
        if self.check_obv_trend(df):
            technical_score += 10

        # ── CATEGORY 2: 거래량 품질 (max 20) ─────────────────────────────

        # 거래량 급증 ≥ MA20 × 2배 (10pts)
        volume_surge = self.check_volume_surge(df, elapsed_fraction)
        if volume_surge:
            volume_score += 10

        # 종가 > MA20 정배열 기본 (5pts)
        if len(df) >= 20 and df["close"].iloc[-1] > df["ma20"].iloc[-1]:
            volume_score += 5

        # ATR ratio 1.5~4% 단타 적정 변동성 (5pts)
        if self.check_atr_range(df):
            volume_score += 5

        # ── CATEGORY 3: 수급 — 실제 KIS API (max 30) ─────────────────────

        foreign_buy     = investor_trend.get("foreign_net_buy", 0)
        institution_buy = investor_trend.get("institution_net_buy", 0)

        if foreign_buy > 0:
            order_flow_score += 15
        if institution_buy > 0:
            order_flow_score += 15

        # ── INTRADAY MOMENTUM BOOST ────────────────────────────────────
        if is_intraday and intraday_data:
            # 시가 대비 상승 중이며 거래량이 폭발하는 경우 보너스
            current_price = intraday_data["price"]
            open_price = intraday_data["open"]
            
            # 1. 시가 돌파 및 유지 (Bullish) - 2% 이상 상승 시 보너스
            if open_price > 0 and current_price >= open_price * 1.02:
                intraday_bonus += 10
            
            # 2. 거래량 강도 — 하루치로 환산한 거래량이 평균을 넘는 속도인지
            avg_volume = df["volume"].iloc[:-1].tail(20).mean()
            projected = intraday_data["volume"] / max(elapsed_fraction, 0.2)
            if avg_volume > 0 and projected > avg_volume:
                intraday_bonus += 10

            # 배수를 곱하지 않는다. 1.5배 후 100으로 자르면 후보의 3분의 1이 100점
            # 동점이 되어 임계값도 순위도 의미가 없어진다.
            total = technical_score + volume_score + order_flow_score + news_pts + intraday_bonus
        else:
            # ── OVERNIGHT WINDOW 특수 로직 (강화: 승률 18%→개선 목표) ──────────
            if is_overnight_window and len(df) > 0:
                current     = df.iloc[-1]
                candle_size = current["high"] - current["low"]

                if candle_size > 0:
                    close_position = (current["close"] - current["low"]) / candle_size

                    if close_position < 0.85:
                        # 상위 15% 마감만 허용 (기존 0.7→0.85)
                        technical_score = 0
                        volume_score    = 0
                    elif technical_score < 20:
                        # 기술적 지표 최소 20점 필수
                        technical_score = 0
                        volume_score    = 0
                    else:
                        overnight_bonus = 15
                        if close_position >= 0.9 and volume_surge:
                            overnight_bonus = 20

                        # 외국인+기관 동시 순매수 필수
                        if order_flow_score < 30:
                            technical_score = 0
                            volume_score    = 0
                            overnight_bonus = 0
                else:
                    # 고가=저가 (거래 없음/상하한가 고정) — 종가 위치를 판단할 수 없다
                    technical_score = 0
                    volume_score    = 0

            total = technical_score + volume_score + order_flow_score + news_pts + overnight_bonus

        return {
            "total":           max(0, min(100, total)),
            "technical":       technical_score,
            "volume":          volume_score,
            "order_flow":      order_flow_score,
            "news":            news_pts,
            "overnight_bonus": overnight_bonus,
            "intraday_bonus":  intraday_bonus,
        }

    def _generate_confluence_reason(self, score_dict: dict) -> str:
        """분석된 점수 구성을 바탕으로 한글 요약 사유 생성."""
        reasons = []
        if score_dict.get("technical", 0) >= 20: 
            reasons.append("기술적 지표 강세")
        elif score_dict.get("technical", 0) >= 10:
            reasons.append("차트 반등 시그널")

        if score_dict.get("order_flow", 0) >= 30:
            reasons.append("외인/기관 동반 매수")
        elif score_dict.get("order_flow", 0) >= 15:
            reasons.append("수급 유입 확인")

        if score_dict.get("volume", 0) >= 10:
            reasons.append("거래량 급증")

        if score_dict.get("news", 0) >= 7:
            reasons.append("뉴스 모멘텀 긍정")

        if score_dict.get("overnight_bonus", 0) > 0:
            reasons.append("오버나이트 적합성 높음")

        if score_dict.get("intraday_bonus", 0) > 0:
            reasons.append("실시간 모멘텀 가속")

        if not reasons:
            return "종합 모멘텀 분석"
        
        return ", ".join(reasons)

    # ------------------------------------------------------------------
    # 단일 종목 처리 (핵심 파이프라인)
    # ------------------------------------------------------------------

    async def _process_ticker(self, ticker: str, market: str,
                             is_overnight_window: bool = False,
                             is_intraday: bool = False) -> dict:
        """단일 종목 비동기 처리: OHLCV → 수급 → 뉴스 → 점수 → 반환.

        Issue #10-A: 비인트라데이 모드에서 get_current_price() API 호출 제거 (API 절약).
                     현재가는 OHLCV 마지막 종가를 사용.
        Issue #10-B: 뉴스 분석 5초 타임아웃으로 파이프라인 블로킹 방지.
        Issue #9-C:  market_code 파라미터를 get_investor_trend()에 전달 (KOSDAQ 버그 수정).
        Issue #9-D:  reason 태그를 "Overnight"/"Intraday"/"Momentum" 으로 통일.
        """
        try:
            # 1+3. OHLCV & 수급 병렬 조회 — 독립적이므로 gather (P7)
            market_code = self.market_codes.get(market, "J")
            ohlcv_data, investor_trend = await asyncio.gather(
                self.api_client.get_ohlcv(ticker, period_code="D", count=100),
                self.api_client.get_investor_trend(ticker, market_code=market_code),
            )

            if ohlcv_data.empty:
                logger.warning(f"[{ticker}] OHLCV 데이터 없음 - 건너뜀")
                return {}
            if len(ohlcv_data) < 20:
                logger.warning(f"[{ticker}] 데이터 부족 ({len(ohlcv_data)}행) - 건너뜀")
                return {}

            # 수급을 확인할 수 없으면 후보로 올리지 않는다. 점수의 30%가 수급이고
            # Overnight 은 외국인·기관 동시 순매수가 필수 조건이다.
            if not investor_trend.get("data_available", True):
                return {}
            # 장중 판단(장중 모멘텀·오버나이트)에는 당일 수급이 필요하다. 추정치가 비어
            # 확정치로 폴백하면 전 거래일 값이 온다 — 그것으로 "오늘 외국인·기관이
            # 사고 있다"고 판단하지 않는다.
            if (is_intraday or is_overnight_window) and investor_trend.get("source") == "confirmed" \
                    and investor_trend.get("as_of") != datetime.now().strftime("%Y%m%d"):
                return {}

            # 2. 장중 판단(장중 모멘텀·오버나이트)은 당일 봉이 필요하다. 실시간 시세로
            #    당일 봉을 만들어 붙인다. 프리마켓은 완성된 일봉만 쓴다 (Issue #10-A).
            current_price = float(ohlcv_data["close"].iloc[-1])
            intraday_data = None
            elapsed_fraction = 1.0
            if is_intraday or is_overnight_window:
                price_data = await self.api_client.get_current_price(ticker)
                if not price_data or not price_data.get("price"):
                    logger.info(f"[{ticker}] 실시간 시세 조회 실패 - 건너뜀")
                    return {}
                current_price = price_data["price"]
                ohlcv_data = self._with_today_bar(ohlcv_data, price_data)
                elapsed_fraction = self._session_elapsed_fraction()
                if is_intraday:
                    intraday_data = price_data
            else:
                ohlcv_data = self._completed_bars(ohlcv_data)
                if len(ohlcv_data) < 20:
                    return {}

            stock_name = self.api_client.stock_names.get(ticker, ticker)

            # 4. 뉴스 점수 — 5초 타임아웃으로 파이프라인 블로킹 방지 (Issue #10-B)
            news_score_raw = 0.0
            if self.news_analyzer is not None:
                try:
                    news_days = 1 if is_intraday else 2
                    news_result = await asyncio.wait_for(
                        self.news_analyzer.analyze_stock_news(ticker, stock_name, days=news_days),
                        timeout=5.0
                    )
                    news_score_raw = float(news_result.get("score", 0.0))
                except asyncio.TimeoutError:
                    logger.debug(f"[{ticker}] 뉴스 분석 타임아웃 (5s) - 점수 0 처리")
                except Exception as e:
                    logger.debug(f"[{ticker}] 뉴스 분석 실패: {e}")

            # 5. 100점 스코어링
            score_dict = self.calculate_stock_score(
                ticker=ticker,
                ohlcv_data=ohlcv_data,
                investor_trend=investor_trend,
                is_overnight_window=is_overnight_window,
                is_intraday=is_intraday,
                intraday_data=intraday_data,
                news_score=news_score_raw,
                elapsed_fraction=elapsed_fraction,
            )
            total_score = score_dict["total"]

            # 6. 세션별 임계값 필터
            if total_score < self.get_entry_threshold(is_overnight_window):
                return {}

            # 7. reason 태그 통일 — async_trader 필터링과 전략 청산 로직에서 정확히 매칭 (Issue #9-D)
            if is_overnight_window:
                reason_tag = "Overnight"
            elif is_intraday:
                reason_tag = "Intraday"
            else:
                reason_tag = "Momentum"

            detail = self._generate_confluence_reason(score_dict)
            logger.debug(f"[{ticker}] reason={reason_tag} score={total_score} detail={detail}")

            return {
                "ticker":  ticker,
                "name":    stock_name,
                "score":   total_score,
                "price":   current_price,
                "reason":  reason_tag,
                "market":  market,
                "source":  "confluence",
                "score_breakdown":     score_dict,
                "foreign_net_buy":     investor_trend.get("foreign_net_buy", 0),
                "institution_net_buy": investor_trend.get("institution_net_buy", 0),
                "_ohlcv_snapshot":     ohlcv_data,   # 매수 단계 재조회 방지 (P4)
            }

        except Exception as e:
            logger.error(f"[{ticker}] _process_ticker 오류: {e}", exc_info=True)

        return {}

    # ------------------------------------------------------------------
    # 병렬 스크리닝 실행
    # ------------------------------------------------------------------

    async def run_screening_async(self, market_list=["KOSPI", "KOSDAQ"], is_intraday: bool = False,
                                  market_regime: str = "NORMAL") -> list:
        """
        asyncio.gather를 사용한 병렬 스크리닝.
        """
        tasks = []

        max_price = await self._affordable_max_price()
        logger.info(f"스크리닝 시작 (시장: {market_list}, 실시간 모멘텀: {is_intraday}, "
                    f"가격 상한: {max_price:,}원)")
        for market in market_list:
            tickers = await self.get_market_stocks(market, max_price=max_price)
            logger.info(f"{market}: {len(tickers)}개 종목 발견")
            for t in tickers:
                tasks.append((t, market))

        if not tasks:
            logger.warning("스크리닝할 종목이 없습니다.")
            return []

        total = len(tasks)
        logger.info(f"총 {total}개 종목 스크리닝 시작")

        # 코루틴 동시 실행 제한 (demo 모드 권장치 준수)
        concurrent_limit = 2 if self.api_client.demo_mode else 10
        semaphore = asyncio.Semaphore(concurrent_limit)

        async def process_with_limit(ticker: str, market: str) -> dict:
            async with semaphore:
                # 8시 스크리닝은 overnight_window=False, is_intraday=False
                # 9시 이후는 상황에 따라 조정
                now_str = datetime.now().strftime("%H:%M")
                is_overnight = config.OVERNIGHT_BUY_START <= now_str <= config.OVERNIGHT_BUY_END
                
                result = await self._process_ticker(ticker, market, 
                                                    is_overnight_window=is_overnight,
                                                    is_intraday=is_intraday)
                # API 부하 분산
                await asyncio.sleep(1.0 if self.api_client.demo_mode else 0.2)
                return result

        raw_results = await asyncio.gather(
            *[process_with_limit(t, m) for t, m in tasks],
            return_exceptions=True
        )

        valid_results = []
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                logger.error(f"[{tasks[i][0]}] 처리 중 예외 발생: {r}")
                continue
            if r and isinstance(r, dict) and r.get("ticker"):
                valid_results.append(r)

        valid_results.sort(key=lambda x: x["score"], reverse=True)
        self.candidate_stocks = valid_results

        logger.info(f"스크리닝 완료: {len(valid_results)}/{total}개 후보 선별")
        return valid_results

    def get_final_candidates(self, limit=5):
        return self.candidate_stocks[:limit] if self.candidate_stocks else []
