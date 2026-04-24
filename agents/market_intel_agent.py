"""
MarketIntelligenceAgent — 시장 정보 에이전트 (시장 정보의 최전선)

담당 역할:
  1. 시장 체제 판단: BULL / BEAR / NORMAL / VOLATILE
  2. 시장 폭 (Market Breadth): 상승/하락 종목 수, 등락비율
  3. 수급 흐름: 외국인·기관·프로그램 일별 순매수
  4. 섹터 강/약 추적 → 핫 테마 식별
  5. MarketContext를 실시간 업데이트해 다른 에이전트에 제공

업데이트 주기:
  - 시장 폭/수급: 15분마다
  - 시장 체제: 30분마다
  - 섹터 강도: 30분마다
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent, AgentSignal, MarketContext

logger = logging.getLogger("agent.market_intel")

# KIS API를 통해 조회할 KOSPI 대표 ETF / 섹터 ETF
KOSPI_PROXY = "069500"          # KODEX 200
KOSDAQ_PROXY = "229200"         # KODEX 코스닥150
SECTOR_ETFS = {
    "반도체":     "091160",     # KODEX 반도체
    "IT":         "266360",     # KODEX Fn성장IT
    "바이오":     "244580",     # KODEX 바이오
    "2차전지":    "305720",     # KODEX 2차전지산업
    "방산":       "364970",     # KODEX 방위산업Plus
    "금융":       "091170",     # KODEX 은행
    "에너지":     "322400",     # KODEX 에너지화학
    "자동차":     "091180",     # KODEX 자동차
}

# G2: 핫 테마 추적용 ETF (단타 매매에 직결되는 주요 테마)
THEME_ETFS = {
    "AI":         "379800",     # KODEX 미국S&P500 (AI 관련 proxy)
    "로봇":       "278540",     # KODEX 미국로봇(ROBO)
    "원자력":     "367770",     # TIGER 원자력테마
    "조선":       "466930",     # KODEX 조선
    "수소":       "367380",     # KODEX K-미래차(수소 포함)
    "리튬":       "305720",     # KODEX 2차전지산업
    "항공":       "251340",     # KODEX 운송
    "게임":       "300640",     # KODEX 게임
}

# 대체 후보 ETF — 1차 티커 실패 시 순서대로 시도
_ETF_FALLBACKS: Dict[str, List[str]] = {
    "반도체":   ["091160", "395160", "472160"],
    "IT":       ["266360", "315480", "102780"],
    "바이오":   ["244580", "253280", "261060"],
    "2차전지":  ["305720", "371460", "455890"],
    "방산":     ["364970", "425080", "409820"],
    "금융":     ["091170", "102970", "091220"],
    "에너지":   ["322400", "117460", "261220"],
    "자동차":   ["091180", "204420", "381170"],
    "AI":       ["379800", "381170", "360750"],
    "로봇":     ["278540", "396500", "315480"],
    "원자력":   ["367770", "456840", "302190"],
    "조선":     ["466930", "140700", "241560"],
    "수소":     ["367380", "381180", "371460"],
    "리튬":     ["305720", "446650", "455890"],
    "항공":     ["251340", "234310", "140710"],
    "게임":     ["300640", "102780", "091180"],
}

# 시장 폭 계산을 위한 KOSPI 상위 종목 샘플 (실제로는 API로 조회)
BREADTH_SAMPLE_COUNT = 100


class MarketIntelligenceAgent(BaseAgent):
    """
    시장 전체 상황을 분석해 MarketContext를 유지·공급합니다.
    다른 에이전트들은 이 컨텍스트를 기반으로 전략을 보정합니다.
    """

    def __init__(self, api_client=None, config=None):
        super().__init__("MarketIntel", api_client, config)
        self.context = MarketContext()

        # 캐시
        self._sector_cache: Dict[str, float] = {}       # sector → 수익률(%)
        self._theme_cache: Dict[str, float] = {}        # theme → 수익률(%)
        self._breadth_cache: Dict[str, Any] = {}
        self._last_regime_update: Optional[datetime] = None
        self._last_breadth_update: Optional[datetime] = None
        self._last_sector_update: Optional[datetime] = None
        self._last_theme_update: Optional[datetime] = None
        self._last_flow_update: Optional[datetime] = None

    # ─────────────────────────────────────────────
    # ETF 티커 자동 검증 & 교체
    # ─────────────────────────────────────────────

    async def _validate_etf_tickers(self) -> None:
        """시작 시 모든 ETF 티커의 OHLCV 데이터 존재를 검증.

        데이터가 없는 티커는 _ETF_FALLBACKS에서 대체 후보를 순차 시도하여
        유효한 티커로 자동 교체합니다.
        """
        if not self.api_client:
            return

        global SECTOR_ETFS, THEME_ETFS
        for etf_dict, label in [(SECTOR_ETFS, "섹터"), (THEME_ETFS, "테마")]:
            for name, ticker in list(etf_dict.items()):
                try:
                    df = await self.api_client.get_ohlcv(ticker, period_code="D", count=5)
                    if df is not None and len(df) >= 2:
                        continue  # 정상
                except Exception:
                    pass

                # 현재 티커 실패 → 대체 후보 순차 시도
                fallbacks = _ETF_FALLBACKS.get(name, [])
                replaced = False
                for alt in fallbacks:
                    if alt == ticker:
                        continue
                    try:
                        df = await self.api_client.get_ohlcv(alt, period_code="D", count=5)
                        if df is not None and len(df) >= 2:
                            etf_dict[name] = alt
                            logger.warning(
                                f"[ETF검증] {label} '{name}' 티커 교체: "
                                f"{ticker} → {alt} (데이터 확인됨)"
                            )
                            replaced = True
                            break
                    except Exception:
                        continue

                if not replaced:
                    # 대체 후보도 전부 실패 → 해당 항목 제거
                    del etf_dict[name]
                    logger.warning(
                        f"[ETF검증] {label} '{name}' 모든 후보 실패 → 추적 목록에서 제거"
                    )

        logger.info(
            f"[ETF검증] 완료 — 섹터 {len(SECTOR_ETFS)}개, 테마 {len(THEME_ETFS)}개 활성"
        )

    # ─────────────────────────────────────────────
    # 라이프사이클
    # ─────────────────────────────────────────────

    async def initialize(self) -> None:
        """초기 시장 데이터 적재."""
        # ETF 티커 유효성 검증 — 데이터 없는 티커는 대체 후보로 자동 교체
        await self._validate_etf_tickers()

        try:
            await self._update_regime()
            await self._update_breadth()
            await self._update_sector_rotation()
            await self._update_hot_themes()
            await self._update_program_flow()
        except Exception as e:
            logger.warning(f"초기화 중 오류 (계속 진행): {e}")

        # 백그라운드 주기적 업데이트
        self._spawn(self._safe_loop(self._update_regime, 1800, "시장체제"))        # 30분
        self._spawn(self._safe_loop(self._update_breadth, 900, "시장폭"))          # 15분
        self._spawn(self._safe_loop(self._update_sector_rotation, 1800, "섹터"))   # 30분
        self._spawn(self._safe_loop(self._update_hot_themes, 1800, "핫테마"))      # 30분
        self._spawn(self._safe_loop(self._update_program_flow, 900, "프로그램수급")) # 15분

    async def analyze(
        self, context: MarketContext, candidates: List[Dict]
    ) -> List[AgentSignal]:
        """MarketContext를 공유하는 역할 — 직접 매매 신호는 발행하지 않음."""
        # 컨텍스트는 백그라운드 루프가 유지; analyze() 호출 시 최신화만 확인
        signals: List[AgentSignal] = []

        # 극단적 위험 신호 발행
        if self.context.regime in ("VOLATILE", "VOLATILE_UP", "VOLATILE_DOWN") and self.context.kospi_volatility > 0.45:
            signals.append(self._make_signal(
                "RISK", None, confidence=0.9,
                strategy="extreme_volatility",
                metadata={"volatility": self.context.kospi_volatility, "regime": "VOLATILE"},
            ))

        # 시장 급락 경보 (breadth 20 미만)
        if self.context.breadth_score < 20:
            signals.append(self._make_signal(
                "RISK", None, confidence=0.85,
                strategy="breadth_collapse",
                metadata={"breadth": self.context.breadth_score},
            ))

        # 수급 극단 매수 기회 (외국인+기관 > 5000억)
        if self.context.net_institutional_flow() > 500:  # 억 단위
            signals.append(self._make_signal(
                "INFO", None, confidence=0.7,
                strategy="massive_inflow",
                metadata={"net_flow": self.context.net_institutional_flow()},
            ))

        return signals

    # ─────────────────────────────────────────────
    # 시장 체제 (Regime Detection)
    # ─────────────────────────────────────────────

    async def _update_regime(self) -> None:
        """KOSPI / KOSDAQ 데이터로 시장 체제 판단."""
        if not self.api_client:
            return

        try:
            kospi_df = await self.api_client.get_ohlcv(KOSPI_PROXY, period_code="D", count=120)
            if kospi_df is None or len(kospi_df) < 60:
                return

            close = kospi_df["close"]
            vol = kospi_df["volume"]

            ma20 = close.rolling(20).mean().iloc[-1]
            ma60 = close.rolling(60).mean().iloc[-1]
            ma120 = close.rolling(120).mean().iloc[-1]
            current = close.iloc[-1]

            # 연간화 변동성
            ret = close.pct_change().dropna()
            kospi_vol = float(ret.rolling(20).std().iloc[-1] * np.sqrt(252))
            self.context.kospi_volatility = kospi_vol

            # 체제 결정
            regime, confidence = self._classify_regime(current, ma20, ma60, ma120, kospi_vol)
            self.context.regime = regime
            self.context.regime_confidence = confidence
            self._last_regime_update = datetime.now()

            logger.info(
                f"[MarketIntel] 시장체제={regime} (신뢰도={confidence:.2f}) "
                f"KOSPI변동성={kospi_vol:.1%} MA20={ma20:.0f} MA60={ma60:.0f}"
            )

        except Exception as e:
            logger.error(f"시장체제 업데이트 실패: {e}")

    def _classify_regime(
        self, price: float, ma20: float, ma60: float, ma120: float, vol: float
    ) -> tuple[str, float]:
        """
        규칙 기반 시장 체제 분류.
        Returns: (regime, confidence)
        """
        score = 0.0  # -3 ~ +3 (양수=강세, 음수=약세)
        evidence = 0

        if price > ma20:
            score += 1; evidence += 1
        else:
            score -= 1; evidence += 1

        if ma20 > ma60:
            score += 1; evidence += 1
        else:
            score -= 1; evidence += 1

        if ma60 > ma120:
            score += 0.5; evidence += 1
        else:
            score -= 0.5; evidence += 1

        # 변동성 체크: 방향에 따라 VOLATILE_UP / VOLATILE_DOWN 세분화
        if vol > 0.40:
            direction = "UP" if price >= ma20 else "DOWN"
            return f"VOLATILE_{direction}", min(0.5 + vol - 0.40, 0.95)

        confidence = abs(score) / (evidence + 1e-9)

        if score >= 2.0:
            return "BULL", min(confidence, 0.9)
        elif score <= -2.0:
            return "BEAR", min(confidence, 0.9)
        else:
            return "NORMAL", 0.5 + abs(score) * 0.1

    # ─────────────────────────────────────────────
    # 시장 폭 (Market Breadth)
    # ─────────────────────────────────────────────

    async def _update_breadth(self) -> None:
        """상승/하락 종목 비율로 시장 폭 계산."""
        if not self.api_client:
            return
        try:
            # 상위 종목 목록
            kospi_tickers = await self.api_client.get_top_market_stocks("J", BREADTH_SAMPLE_COUNT)
            if not kospi_tickers:
                return

            # 병렬 현재가 조회
            tasks = [self.api_client.get_current_price(t) for t in kospi_tickers[:50]]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            advances = declines = unchanged = 0
            for r in results:
                if isinstance(r, Exception) or not r:
                    continue
                chg = r.get("change_rate", 0.0)  # 전일대비 등락률
                if chg > 0:
                    advances += 1
                elif chg < 0:
                    declines += 1
                else:
                    unchanged += 1

            total = advances + declines + unchanged
            if total == 0:
                return

            # 유효 데이터 없음 — change_rate 키가 없거나 모의투자 환경에서
            # 전일대비율이 0으로 반환되어 advances+declines==0인 경우,
            # 기존 breadth_score를 유지해 breadth_collapse 오경보 방지
            if advances + declines < 3:
                logger.debug(
                    f"[MarketIntel] 시장폭 — 유효 등락 데이터 부족 "
                    f"(advances={advances}, declines={declines}, unchanged={unchanged}), "
                    f"breadth_score 유지 ({self.context.breadth_score})"
                )
                return

            ad_ratio = advances / (declines + 1)
            breadth = (advances / total) * 100  # 0~100

            self.context.breadth_score = round(breadth, 1)
            self.context.advance_decline_ratio = round(ad_ratio, 2)
            self._last_breadth_update = datetime.now()

            logger.info(
                f"[MarketIntel] 시장폭={breadth:.1f}% "
                f"상승={advances} 하락={declines} A/D={ad_ratio:.2f}"
            )

        except Exception as e:
            logger.error(f"시장폭 업데이트 실패: {e}")

    # ─────────────────────────────────────────────
    # 섹터 로테이션
    # ─────────────────────────────────────────────

    async def _update_sector_rotation(self) -> None:
        """섹터 ETF 성과로 강/약세 섹터 파악."""
        if not self.api_client:
            return
        try:
            perf: Dict[str, float] = {}
            tasks = {name: self.api_client.get_ohlcv(etf, period_code="D", count=5)
                     for name, etf in SECTOR_ETFS.items()}

            for name, coro in tasks.items():
                try:
                    df = await coro
                    if df is not None and len(df) >= 2:
                        ret_5d = (df["close"].iloc[-1] / df["close"].iloc[-5] - 1) * 100
                        perf[name] = round(float(ret_5d), 2)
                except Exception:
                    pass

            if not perf:
                return

            sorted_sectors = sorted(perf.items(), key=lambda x: x[1], reverse=True)
            self._sector_cache = perf

            # 상위 3 → 핫 섹터, 하위 2 → 약세 섹터
            self.context.sector_leaders = [s for s, _ in sorted_sectors[:3]]
            self.context.sector_laggards = [s for s, _ in sorted_sectors[-2:]]
            self._last_sector_update = datetime.now()

            logger.info(
                f"[MarketIntel] 섹터 강세: {self.context.sector_leaders} "
                f"/ 약세: {self.context.sector_laggards}"
            )

        except Exception as e:
            logger.error(f"섹터 로테이션 업데이트 실패: {e}")

    # ─────────────────────────────────────────────
    # G2: 핫 테마 추적
    # ─────────────────────────────────────────────

    async def _update_hot_themes(self) -> None:
        """테마 ETF 5일 성과로 핫 테마 파악."""
        if not self.api_client:
            return
        try:
            perf: Dict[str, float] = {}
            for name, etf in THEME_ETFS.items():
                try:
                    df = await self.api_client.get_ohlcv(etf, period_code="D", count=5)
                    if df is not None and len(df) >= 2:
                        ret_5d = (df["close"].iloc[-1] / df["close"].iloc[-5] - 1) * 100
                        perf[name] = round(float(ret_5d), 2)
                except Exception:
                    pass

            if not perf:
                return

            sorted_themes = sorted(perf.items(), key=lambda x: x[1], reverse=True)
            self._theme_cache = perf

            # 양수 수익률인 테마만 상위 3개 핫 테마로 선정
            self.context.hot_themes = [t for t, v in sorted_themes[:3] if v > 0]
            self._last_theme_update = datetime.now()

            logger.info(
                f"[MarketIntel] 핫 테마: {self.context.hot_themes} "
                f"(상위: {sorted_themes[:3]})"
            )

        except Exception as e:
            logger.error(f"핫 테마 업데이트 실패: {e}")

    # ─────────────────────────────────────────────
    # G3: 프로그램 수급 추정
    # ─────────────────────────────────────────────

    async def _update_program_flow(self) -> None:
        """외국인·기관·개인 수급을 시장 전체 투자자매매동향 API로 직접 조회.

        Bug fix (2026-04-24): 기존 KODEX200(069500) proxy는 Primary API
        (HHPTJ04160200)가 ETF를 미지원하여 `output2` 빈 리스트 → 수급이 항상 +0억.
        신규: FHPTJ04030000 (inquire-investor-time-by-market)로 코스피+코스닥
        시장 전체 순매수 금액을 1회 호출로 집계 → 정확한 aggregate 신호.
        """
        if not self.api_client:
            return
        try:
            import asyncio as _asyncio
            kospi_res, kosdaq_res = await _asyncio.gather(
                self.api_client.get_market_investor_flow("KOSPI"),
                self.api_client.get_market_investor_flow("KOSDAQ"),
                return_exceptions=True,
            )

            total_foreign_bn = 0.0
            total_inst_bn = 0.0
            total_personal_bn = 0.0
            available_markets: list[str] = []

            for mkt_name, res in (("KOSPI", kospi_res), ("KOSDAQ", kosdaq_res)):
                if isinstance(res, dict) and res.get("data_available"):
                    total_foreign_bn += float(res.get("foreign_net_amount_bn", 0) or 0)
                    total_inst_bn += float(res.get("institution_net_amount_bn", 0) or 0)
                    total_personal_bn += float(res.get("personal_net_amount_bn", 0) or 0)
                    available_markets.append(mkt_name)

            if not available_markets:
                logger.warning("[MarketIntel] 시장 수급 조회: 양 시장 모두 실패")
                return

            self.context.institution_flow_bn = round(total_inst_bn, 1)
            self.context.foreign_flow_bn = round(total_foreign_bn, 1)
            self.context.program_flow_bn = round(total_inst_bn + total_foreign_bn, 1)
            self._last_flow_update = datetime.now()

            logger.info(
                f"[MarketIntel] 수급({'+'.join(available_markets)}): "
                f"기관={self.context.institution_flow_bn:+.0f}억 "
                f"외국인={self.context.foreign_flow_bn:+.0f}억 "
                f"합={self.context.program_flow_bn:+.0f}억 "
                f"(개인={total_personal_bn:+.0f}억)"
            )

        except Exception as e:
            logger.error(f"프로그램 수급 업데이트 실패: {e}")

    # ─────────────────────────────────────────────
    # 퍼블릭 헬퍼
    # ─────────────────────────────────────────────

    def get_context(self) -> MarketContext:
        """최신 MarketContext 스냅샷 반환 (동시 수정 방지)."""
        import dataclasses
        self.context.updated_at = datetime.now()
        return dataclasses.replace(self.context)

    def sector_performance(self) -> Dict[str, float]:
        """섹터별 5일 수익률 dict."""
        return dict(self._sector_cache)

    def is_hot_sector(self, sector_name: str) -> bool:
        """해당 섹터가 현재 강세 섹터인지 확인."""
        return sector_name in self.context.sector_leaders

    def regime_multiplier(self) -> float:
        """
        시장 체제에 따른 포지션 배율.
        BULL=1.2, NORMAL=1.0, BEAR=0.6, VOLATILE=0.4
        """
        multipliers = {
            "BULL": 1.2, "NORMAL": 1.0, "BEAR": 0.6,
            "VOLATILE": 0.4, "VOLATILE_UP": 0.9, "VOLATILE_DOWN": 0.6,
        }
        return multipliers.get(self.context.regime, 1.0)
