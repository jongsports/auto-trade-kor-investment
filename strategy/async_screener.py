import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime

from typing import List, Dict
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

    async def get_market_stocks(self, market="KOSPI") -> List[str]:
        """거래량/거래대금 상위 종목 조회 (Dynamic)."""
        if market == "KOSPI":
            return await self.api_client.get_top_market_stocks("0001", count=30)
        elif market == "KOSDAQ":
            return await self.api_client.get_top_market_stocks("1001", count=20)
        return []

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

    def check_volume_surge(self, df: pd.DataFrame) -> bool:
        """거래량이 20일 MA의 2배 이상."""
        if len(df) < 20:
            return False
        try:
            vol_ma20_prev = max(df["volume_ma20"].iloc[-2], 1.0)
            return df["volume"].iloc[-1] / vol_ma20_prev >= 2.0
        except Exception:
            return False

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
            return 45

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
        news_score: float = 0.0
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
        df = self.calculate_technical_indicators(ohlcv_data)

        technical_score  = 0   # max 40
        volume_score     = 0   # max 20
        order_flow_score = 0   # max 30
        news_pts         = max(0, min(10, int(news_score)))
        overnight_bonus  = 0

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
        if self.check_volume_surge(df):
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

        # ── OVERNIGHT WINDOW 특수 로직 ────────────────────────────────────

        if is_overnight_window and len(df) > 0:
            current     = df.iloc[-1]
            candle_size = current["high"] - current["low"]

            if candle_size > 0:
                close_position = (current["close"] - current["low"]) / candle_size

                if close_position < 0.7:
                    # 고가 부근에서 마감 안 함 → 오버나이트 부적격
                    technical_score = 0
                    volume_score    = 0
                else:
                    overnight_bonus = 15
                    # 종가가 일봉 상위 20% + 거래량 급증 → 추가 보너스
                    if close_position >= 0.8 and self.check_volume_surge(df):
                        overnight_bonus = 20

                    # 오버나이트 필수 조건: 외국인 또는 기관 순매수 반드시 있어야 함
                    if order_flow_score == 0:
                        technical_score = 0
                        volume_score    = 0
                        overnight_bonus = 0

        total = technical_score + volume_score + order_flow_score + news_pts + overnight_bonus

        return {
            "total":           total,
            "technical":       technical_score,
            "volume":          volume_score,
            "order_flow":      order_flow_score,
            "news":            news_pts,
            "overnight_bonus": overnight_bonus,
        }

    # ------------------------------------------------------------------
    # 단일 종목 처리 (핵심 파이프라인)
    # ------------------------------------------------------------------

    async def _process_ticker(self, ticker: str, market: str) -> dict:
        """단일 종목 비동기 처리: OHLCV → 수급 → 뉴스 → 점수 → 반환."""
        try:
            now_str = datetime.now().strftime("%H:%M")
            is_overnight_window = config.OVERNIGHT_BUY_START <= now_str <= config.OVERNIGHT_BUY_END

            # 1. OHLCV 조회
            ohlcv_data = await self.api_client.get_ohlcv(ticker, period_code="D", count=100)

            if ohlcv_data.empty:
                logger.warning(f"[{ticker}] OHLCV 데이터 없음 - 건너뜀")
                return {}

            if len(ohlcv_data) < 20:
                logger.warning(f"[{ticker}] 데이터 부족 ({len(ohlcv_data)}행, 최소 20 필요) - 건너뜀")
                return {}

            logger.info(f"[{ticker}] 데이터 준비 ({len(ohlcv_data)}행)")

            # 2. 수급 조회 (OHLCV 성공 시만 — API 호출 절약)
            investor_trend = await self.api_client.get_investor_trend(ticker)

            # 3. 뉴스 점수 (news_analyzer 없으면 스킵)
            news_score_raw = 0.0
            if self.news_analyzer is not None:
                try:
                    stock_name  = f"STK_{ticker}"
                    news_result = await self.news_analyzer.analyze_stock_news(ticker, stock_name, days=1)
                    raw         = news_result.get("score", 0.0)
                    news_score_raw = min(10.0, max(0.0, float(raw) * 0.5))
                except Exception as e:
                    logger.debug(f"[{ticker}] 뉴스 분석 실패 (비필수): {e}")

            # 4. 100점 스코어링
            score_dict = self.calculate_stock_score(
                ticker=ticker,
                ohlcv_data=ohlcv_data,
                investor_trend=investor_trend,
                is_overnight_window=is_overnight_window,
                news_score=news_score_raw,
            )
            total_score = score_dict["total"]

            logger.info(
                f"[점수 요약] {ticker} ({market}) | "
                f"기술:{score_dict['technical']} 거래량:{score_dict['volume']} "
                f"수급:{score_dict['order_flow']} 뉴스:{score_dict['news']} "
                f"야간보너스:{score_dict['overnight_bonus']} | 총점:{total_score}"
            )

            # 5. 세션별 임계값 필터
            threshold = self.get_entry_threshold(is_overnight_window)
            if total_score < threshold:
                return {}

            # 6. 현재가 및 종목명 조회
            mrkt_code = "J" if market == "KOSPI" else "K"
            params    = {"FID_COND_MRKT_DIV_CODE": mrkt_code, "FID_INPUT_ISCD": ticker}
            info_res  = await self.api_client._fetch(
                "GET", "/uapi/domestic-stock/v1/quotations/inquire-price", "FHKST01010100",
                params=params
            )

            if info_res.get("rt_cd") == "0" and "output" in info_res:
                current_price = int(info_res["output"].get("stck_prpr", 0))
                stock_name    = info_res["output"].get("hts_kor_isnm", f"STK_{ticker}")
            else:
                current_price = int(ohlcv_data["close"].iloc[-1])
                stock_name    = f"STK_{ticker}"

            return {
                # 기존 인터페이스 유지
                "ticker":  ticker,
                "name":    stock_name,
                "score":   total_score,
                "price":   current_price,
                "reason":  "Overnight" if is_overnight_window else "Momentum",
                "market":  market,
                "source":  "confluence",
                # 추가 정보 (하위 호환 — 기존 코드는 무시 가능)
                "score_breakdown":     score_dict,
                "foreign_net_buy":     investor_trend.get("foreign_net_buy", 0),
                "institution_net_buy": investor_trend.get("institution_net_buy", 0),
            }

        except Exception as e:
            logger.error(f"[{ticker}] _process_ticker 오류: {e}", exc_info=True)

        return {}

    # ------------------------------------------------------------------
    # 병렬 스크리닝 실행
    # ------------------------------------------------------------------

    async def run_screening_async(self, market_list=["KOSPI", "KOSDAQ"]) -> list:
        """
        asyncio.gather를 사용한 병렬 스크리닝.

        demo_mode: API 내부 Semaphore(1)이 HTTP를 직렬화하므로 실질적 병렬 처리는
                   실전 모드에서 이루어짐. demo 모드에서는 Python 코루틴 스케줄링 이점만.
        """
        tasks = []

        logger.info(f"스크리닝 시작 - 시장: {market_list}")
        for market in market_list:
            tickers = await self.get_market_stocks(market)
            logger.info(f"{market}: {len(tickers)}개 종목 발견")
            for t in tickers:
                tasks.append((t, market))

        if not tasks:
            logger.warning("스크리닝할 종목이 없습니다.")
            return []

        total = len(tasks)
        logger.info(f"총 {total}개 종목 스크리닝 시작")

        # 코루틴 동시 실행 제한 (demo: API 내부 Semaphore(1)와 협력)
        concurrent_limit = 3 if self.api_client.demo_mode else 10
        semaphore        = asyncio.Semaphore(concurrent_limit)

        async def process_with_limit(ticker: str, market: str) -> dict:
            async with semaphore:
                result = await self._process_ticker(ticker, market)
                # VTS 속도 제한 준수 (demo: 1.5s 추가, 실전: 0.3s)
                await asyncio.sleep(1.5 if self.api_client.demo_mode else 0.3)
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
