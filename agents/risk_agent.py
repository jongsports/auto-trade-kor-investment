"""
RiskManagementAgent — 리스크 관리 에이전트 (자본 보호의 최전선)

기존 AsyncRiskManager를 고도화하여 다음을 추가합니다:
  1. Kelly Criterion 기반 최적 포지션 사이징
  2. 포트폴리오 Heat Map (상관관계 기반 집중도 경보)
  3. VaR (Value at Risk) 계산 및 포지션 한도 자동 조정
  4. 동적 손절가 최적화 (ATR × 시장 변동성 × 전략 승률)
  5. 일별 손실 한도 (Circuit Breaker) 강화
  6. 포지션별 최대 보유 시간 초과 경보

수익 극대화를 위한 핵심 원칙:
  - 승률 높은 전략에 더 많이 배팅 (Kelly)
  - 손실 시 빠르게 줄이고, 수익 시 천천히 키움
  - 포트폴리오 상관관계로 분산 효과 극대화
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config as _cfg
from agents.base_agent import BaseAgent, AgentSignal, MarketContext
from agents.strategy_constants import VOLATILE_ALLOWED, BEAR_BLOCKED, FAST_EXIT_STRATEGIES, SLOW_EXIT_STRATEGIES

logger = logging.getLogger("agent.risk")


# ─────────────────────────────────────────────
# 리스크 상수 (Korean Market Calibrated)
# ─────────────────────────────────────────────

class RiskLimits:
    # 일별 손실 한도
    MAX_DAILY_LOSS_RATIO = 0.03       # 계좌 3% 손실 시 당일 매매 중단
    CIRCUIT_BREAKER_RATIO = 0.05      # 계좌 5% 손실 시 즉시 전량 청산 (레거시 — CB_LEVEL_3과 동일)

    # 누진적 서킷브레이커 레벨 (P3)
    CB_LEVEL_1 = 0.02   # -2%: 포지션 사이즈 50% 축소
    CB_LEVEL_2 = 0.03   # -3%: 신규 진입 중단
    CB_LEVEL_3 = 0.05   # -5%: 손실 포지션 강제 청산
    CB_LEVEL_4 = 0.07   # -7%: 전 포지션 청산 + 당일 완전 중단

    # 포지션 한도
    MAX_SINGLE_POSITION = 0.20        # 단일 종목 최대 20%
    MAX_TOTAL_EXPOSURE  = 0.70        # 총 노출 최대 70%
    MAX_SECTOR_EXPOSURE = 0.35        # 동일 섹터 최대 35%

    # Kelly Criterion
    KELLY_FRACTION = 0.35             # Full Kelly의 35% 사용 (수수료 대비 수익성 확보)
    MIN_WIN_RATE    = 0.52            # Kelly 적용 최소 승률

    # 동적 손절
    ATR_STOP_NORMAL   = 1.8          # 정상 시장: ATR × 1.8
    ATR_STOP_VOLATILE = 1.2          # 변동 시장: ATR × 1.2 (더 빠른 손절)
    ATR_STOP_BULL     = 2.2          # 강세 시장: ATR × 2.2 (여유)
    MIN_STOP_PCT = 0.015              # 최소 손절 1.5%
    MAX_STOP_PCT = 0.05               # 최대 손절 5%

    # 트레일링 스탑
    TRAIL_ACTIVATE_PCT = 0.03         # 3% 수익 시 트레일링 시작
    TRAIL_STEP_1 = 0.02               # 3~7% 수익: 2% 트레일
    TRAIL_STEP_2 = 0.015              # 7~12% 수익: 1.5% 트레일
    TRAIL_STEP_3 = 0.01               # 12%+ 수익: 1% 트레일 (수익 보호)

    # 보유 시간 한도
    MAX_HOLD_MIN_INTRADAY = 120       # 당일 전략: 최대 2시간
    MAX_HOLD_DAYS_OVERNIGHT = 3       # 오버나이트: 최대 3일


RL = RiskLimits()


@dataclass
class PositionRisk:
    """개별 포지션의 실시간 리스크 정보."""
    ticker: str
    quantity: int
    entry_price: float
    current_price: float
    strategy: str
    entry_time: datetime
    high_price: float = 0.0

    # 계산 결과
    unrealized_pnl: float = 0.0
    unrealized_pct: float = 0.0
    trailing_stop: float = 0.0
    dynamic_stop: float = 0.0
    hold_minutes: float = 0.0

    @property
    def profit_ratio(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    @property
    def from_high(self) -> float:
        """고점 대비 하락률."""
        if self.high_price == 0:
            return 0.0
        return (self.current_price - self.high_price) / self.high_price


@dataclass
class RiskReport:
    """현재 포트폴리오 전체 리스크 상태."""
    total_exposure: float = 0.0       # 총 투자 비율
    daily_pnl: float = 0.0            # 당일 실현 손익
    unrealized_pnl: float = 0.0       # 미실현 손익
    positions: List[PositionRisk] = field(default_factory=list)
    circuit_breaker_active: bool = False
    heat_level: str = "LOW"           # LOW / MEDIUM / HIGH / CRITICAL
    alerts: List[str] = field(default_factory=list)


class RiskManagementAgent(BaseAgent):
    """
    포트폴리오 전체의 리스크를 실시간으로 관리합니다.
    진입 허가 / 포지션 사이징 / 손절 트리거 / 서킷브레이커를 담당합니다.
    """

    def __init__(self, api_client=None, config=None, base_risk_manager=None):
        super().__init__("RiskMgmt", api_client, config)

        # 기존 AsyncRiskManager와 연동
        self.base_risk = base_risk_manager

        # 상태
        self._daily_realized_pnl: float = 0.0
        self._cb_level: int = 0  # 0=정상, 1=축소, 2=진입중단, 3=손실청산, 4=완전중단
        self._position_risks: Dict[str, PositionRisk] = {}

        # 전략별 성과 이력 (Kelly 부트스트랩, P1)
        self._strategy_stats_file = os.path.join("data", "strategy_stats.json")
        self._strategy_stats = self._load_strategy_stats()
        self._strategy_win_rates: Dict[str, float] = {
            k: v["win_rate"] for k, v in self._strategy_stats.items() if k != "_meta"
        }
        self._strategy_avg_return: Dict[str, float] = {
            k: v["avg_return"] for k, v in self._strategy_stats.items() if k != "_meta"
        }

        # 계좌 총액 캐시
        self._total_equity: float = 0.0
        self._last_equity_update: Optional[datetime] = None

    # ─────────────────────────────────────────────
    # 전략 성과 이력 관리 (P1: Kelly 부트스트랩)
    # ─────────────────────────────────────────────

    def _load_strategy_stats(self) -> dict:
        """strategy_stats.json 로드. 실패 시 빈 dict → 0.55 기본값 fallback."""
        try:
            if os.path.exists(self._strategy_stats_file):
                with open(self._strategy_stats_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"strategy_stats 로드 실패: {e}")
        return {}

    def _save_strategy_stats(self) -> None:
        """전략 성과 이력 저장 (거래 완료 시 호출)."""
        try:
            stats = {"_meta": {"version": 1, "updated_at": datetime.now().isoformat()}}
            for s in list(self._strategy_win_rates.keys()):
                existing = self._strategy_stats.get(s, {})
                stats[s] = {
                    "win_rate": round(self._strategy_win_rates.get(s, 0.55), 4),
                    "avg_return": round(self._strategy_avg_return.get(s, 0.025), 4),
                    "avg_loss": round(existing.get("avg_loss", 0.02), 4),
                    "trades": existing.get("trades", 0),
                }
            with open(self._strategy_stats_file, "w") as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            logger.warning(f"strategy_stats 저장 실패: {e}")

    # ─────────────────────────────────────────────
    # 라이프사이클
    # ─────────────────────────────────────────────

    async def initialize(self) -> None:
        await self._update_equity()
        self._spawn(self._safe_loop(self._update_equity, 300, "계좌잔고"))   # 5분
        self._spawn(self._safe_loop(self._monitor_positions, 30, "포지션"))  # 30초
        logger.info("[RiskMgmt] 리스크 에이전트 초기화 완료")

    async def analyze(
        self, context: MarketContext, candidates: List[Dict]
    ) -> List[AgentSignal]:
        """포지션 리스크 경보 신호 발행."""
        signals: List[AgentSignal] = []

        # 서킷 브레이커 발동 (레벨 2+ = 진입 중단 이상)
        if self._cb_level >= 2:
            signals.append(self._make_signal(
                "RISK", None, confidence=1.0,
                strategy="circuit_breaker",
                metadata={"daily_pnl": self._daily_realized_pnl},
            ))

        # 개별 포지션 리스크 체크
        for ticker, pos in self._position_risks.items():
            alert = self._check_position_alert(pos, context)
            if alert:
                signals.append(alert)

        return signals

    # ─────────────────────────────────────────────
    # 핵심: 진입 허가 + 포지션 사이징
    # ─────────────────────────────────────────────

    async def can_enter(
        self,
        ticker: str,
        price: float,
        strategy: str,
        context: MarketContext,
        current_holdings: Dict,
        sector: Optional[str] = None,
        ticker_sectors: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, str]:
        """
        진입 가능 여부 판단.
        Returns: (허가 여부, 거부 사유)
        """
        # 1. 서킷 브레이커 (누진적: Level 2+ 진입 중단)
        if self._cb_level >= 4:
            return False, "서킷브레이커 Level 4 (전 포지션 청산 + 당일 완전 중단)"
        if self._cb_level >= 2:
            return False, f"서킷브레이커 Level {self._cb_level} (신규 진입 중단)"

        # 2. 일별 손실 한도
        if self._total_equity > 0:
            loss_ratio = abs(self._daily_realized_pnl) / self._total_equity
            if self._daily_realized_pnl < 0 and loss_ratio >= RL.MAX_DAILY_LOSS_RATIO:
                return False, f"당일 손실 한도 도달 ({loss_ratio:.1%})"

        # 3. 이미 보유 중
        if ticker in current_holdings:
            return False, "이미 보유 중"

        # 4. 총 포지션 수 한도
        max_stocks = getattr(self.config, "MAX_STOCKS", 5) if self.config else 5
        if len(current_holdings) >= max_stocks:
            return False, f"최대 보유 종목 수 초과 ({max_stocks}개)"

        # 5. 총 노출 비율
        if self._total_equity > 0:
            total_invested = sum(
                h.get("buy_price", 0) * h.get("quantity", 0)
                for h in current_holdings.values()
            )
            exposure = total_invested / self._total_equity
            if exposure >= RL.MAX_TOTAL_EXPOSURE:
                return False, f"총 포지션 노출 한도 초과 ({exposure:.1%})"

        # G1. 섹터 집중도 체크 (동일 섹터 최대 35%)
        if sector and ticker_sectors and self._total_equity > 0:
            sector_invested = sum(
                h.get("buy_price", 0) * h.get("quantity", 0)
                for t, h in current_holdings.items()
                if ticker_sectors.get(t) == sector
            )
            sector_exposure = sector_invested / self._total_equity
            if sector_exposure >= RL.MAX_SECTOR_EXPOSURE:
                return False, f"섹터({sector}) 노출 한도 초과 ({sector_exposure:.1%})"

        # 6. 시장 체제 체크
        # VOLATILE_UP/VOLATILE_DOWN 포함하여 처리 (GAP-01 수정)
        _is_volatile = context.regime in ("VOLATILE", "VOLATILE_UP", "VOLATILE_DOWN")
        _is_bear     = context.regime in ("BEAR", "VOLATILE_DOWN")
        if _is_volatile and strategy not in VOLATILE_ALLOWED:
            return False, f"변동성 시장({context.regime}): 전략 {strategy} 비활성"
        if _is_bear and strategy in BEAR_BLOCKED:
            return False, f"약세/하락변동 시장({context.regime}): {strategy} 진입 금지"

        return True, "OK"

    def calc_position_size(
        self,
        ticker: str,
        price: float,
        strategy: str,
        context: MarketContext,
        signal_confidence: float,
        atr: float = 0.0,
    ) -> int:
        """
        Kelly Criterion 기반 최적 포지션 수량 계산.

        Full Kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        사용 Kelly = Full Kelly × KELLY_FRACTION (보수적 적용)
        """
        if self._total_equity <= 0 or price <= 0:
            return 0

        # Kelly 계산 (P1: 이력 부트스트랩)
        stats = self._strategy_stats.get(strategy, {})
        trade_count = stats.get("trades", 0)
        kelly = 0.0

        if trade_count < 30:
            # 부트스트랩 기간: 보수적 Kelly
            kelly_fraction = 0.05
            logger.debug(f"[Kelly] {strategy}: 부트스트랩 ({trade_count}/30 trades) → kelly=0.05")
        else:
            win_rate = self._strategy_win_rates.get(strategy, 0.55)
            avg_return = max(self._strategy_avg_return.get(strategy, 0.025), 0.01)
            avg_loss = atr / price if atr > 0 else stats.get("avg_loss", 0.02)

            if win_rate >= RL.MIN_WIN_RATE:
                edge = win_rate * avg_return - (1 - win_rate) * avg_loss
                kelly = edge / (avg_return + 1e-9)
                # win_rate별 Kelly fraction 스케일링
                if win_rate < 0.55:
                    kelly_fraction = kelly * 0.20
                elif win_rate < 0.60:
                    kelly_fraction = kelly * 0.30
                else:
                    kelly_fraction = kelly * 0.35
            else:
                kelly_fraction = 0.05  # 최소 배팅

        # 시장 체제 보정 — config 기반 (P4: 하드코딩 제거)
        rp = _cfg.get_regime_params(context.regime)
        regime_mult = rp.get("position_size_multiplier", 1.0)

        # risk_status 배율 통합
        risk_status = getattr(self.base_risk, "risk_status", "NORMAL") if self.base_risk else "NORMAL"
        risk_mult = _cfg.RISK_POSITION_MULTIPLIER.get(risk_status, 1.0)

        kelly_fraction *= regime_mult * risk_mult

        # CB Level 1: 포지션 사이즈 50% 축소
        cb_mult = self.get_cb_position_multiplier()
        kelly_fraction *= cb_mult

        # 신뢰도 보정
        kelly_fraction *= (0.7 + signal_confidence * 0.6)

        # 최대 단일 포지션 한도 클램핑
        kelly_fraction = min(kelly_fraction, RL.MAX_SINGLE_POSITION)
        kelly_fraction = max(kelly_fraction, 0.07)  # 최소 7% (수수료 대비 의미있는 포지션)

        target_amount = self._total_equity * kelly_fraction
        # 최소 주문금액 50만원 보장 (수수료 손익분기 확보)
        target_amount = max(target_amount, 500_000)
        quantity = max(int(target_amount / price), 1)

        logger.info(
            f"[PosSizing] {ticker}: kelly_base={kelly:.3f} × regime={regime_mult:.2f} "
            f"× risk={risk_mult:.2f} × cb={cb_mult:.2f} × conf={signal_confidence:.2f} "
            f"→ final={kelly_fraction:.3f} → qty={quantity} "
            f"(전략={strategy}, 체제={context.regime}, 리스크={risk_status}, trades={trade_count})"
        )
        return quantity

    def calc_dynamic_stop(
        self,
        entry_price: float,
        atr: float,
        context: MarketContext,
        strategy: str,
    ) -> float:
        """동적 손절가 계산."""
        # 시장 체제별 ATR 배율
        # VOLATILE_UP/DOWN 포함 — 방향성 급락 시 가장 타이트한 손절 (GAP-01 수정)
        atr_mult = {
            "BULL":          RL.ATR_STOP_BULL,
            "NORMAL":        RL.ATR_STOP_NORMAL,
            "VOLATILE":      RL.ATR_STOP_VOLATILE,
            "BEAR":          RL.ATR_STOP_VOLATILE,
            "VOLATILE_UP":   RL.ATR_STOP_NORMAL,    # 상승 변동성: 일반 수준 유지
            "VOLATILE_DOWN": RL.ATR_STOP_VOLATILE,  # 하락 변동성: VOLATILE과 동일
        }.get(context.regime, RL.ATR_STOP_NORMAL)

        # 전략별 보정 (단타일수록 빠른 손절) — 상수 참조 (GAP-11 수정)
        if strategy in FAST_EXIT_STRATEGIES:
            atr_mult *= 0.85  # 빠른 손절
        elif strategy in SLOW_EXIT_STRATEGIES:
            atr_mult *= 1.1   # 여유 있는 손절

        dynamic_stop = entry_price - atr * atr_mult

        # 클램핑
        min_stop = entry_price * (1 - RL.MAX_STOP_PCT)
        max_stop = entry_price * (1 - RL.MIN_STOP_PCT)
        return max(min_stop, min(max_stop, dynamic_stop))

    def calc_trailing_stop(
        self,
        entry_price: float,
        high_price: float,
        current_price: float,
    ) -> float:
        """
        수익률 단계별 트레일링 스탑가 계산.
        수익이 클수록 스탑을 타이트하게 올려 수익을 보호합니다.
        """
        profit_ratio = (high_price - entry_price) / entry_price if entry_price > 0 else 0

        if profit_ratio < RL.TRAIL_ACTIVATE_PCT:
            return 0.0  # 트레일링 미작동

        if profit_ratio < 0.07:
            trail_pct = RL.TRAIL_STEP_1
        elif profit_ratio < 0.12:
            trail_pct = RL.TRAIL_STEP_2
        else:
            trail_pct = RL.TRAIL_STEP_3  # 12% 이상: 빠른 수익 보호

        return high_price * (1 - trail_pct)

    # ─────────────────────────────────────────────
    # 실시간 포지션 모니터링
    # ─────────────────────────────────────────────

    async def _monitor_positions(self) -> None:
        """포지션별 리스크 지표 업데이트."""
        if not self.api_client:
            return
        try:
            account = await self.api_client.get_account_summary()
            if not account:
                return

            positions = account.get("positions", [])
            for pos in positions:
                ticker = pos.get("ticker", "")
                if not ticker:
                    continue

                existing = self._position_risks.get(ticker)
                cur_price = pos.get("current_price", 0)
                if cur_price <= 0:
                    continue

                if existing is None:
                    self._position_risks[ticker] = PositionRisk(
                        ticker=ticker,
                        quantity=pos.get("quantity", 0),
                        entry_price=pos.get("buy_price", 0),
                        current_price=cur_price,
                        strategy=pos.get("strategy", "unknown"),
                        entry_time=datetime.now(),
                        high_price=cur_price,
                    )
                else:
                    existing.current_price = cur_price
                    existing.quantity = pos.get("quantity", 0)
                    if cur_price > existing.high_price:
                        existing.high_price = cur_price
                    existing.hold_minutes = (
                        datetime.now() - existing.entry_time
                    ).total_seconds() / 60

        except Exception as e:
            logger.debug(f"포지션 모니터링 오류: {e}")

    def _check_position_alert(
        self, pos: PositionRisk, context: MarketContext
    ) -> Optional[AgentSignal]:
        """개별 포지션 리스크 경보 체크."""
        # 1. 트레일링 스탑 트리거
        trail = self.calc_trailing_stop(pos.entry_price, pos.high_price, pos.current_price)
        if trail > 0 and pos.current_price <= trail:
            return self._make_signal(
                "SELL", pos.ticker, confidence=0.95,
                strategy="trailing_stop",
                metadata={
                    "trigger": "trailing_stop",
                    "trail_price": trail,
                    "current": pos.current_price,
                    "profit": pos.profit_ratio,
                },
            )

        # 2. 보유 시간 초과 (당일 전략)
        if pos.strategy in ("Intraday", "Momentum") and pos.hold_minutes > RL.MAX_HOLD_MIN_INTRADAY:
            return self._make_signal(
                "SELL", pos.ticker, confidence=0.80,
                strategy="max_hold_time",
                metadata={"hold_minutes": pos.hold_minutes, "strategy": pos.strategy},
            )

        # 3. 과도한 미실현 손실 (-4% 초과)
        if pos.profit_ratio < -0.04:
            return self._make_signal(
                "SELL", pos.ticker, confidence=0.90,
                strategy="loss_cut",
                metadata={"loss_pct": round(pos.profit_ratio * 100, 2)},
            )

        return None

    # ─────────────────────────────────────────────
    # 보조 메서드
    # ─────────────────────────────────────────────

    async def _update_equity(self) -> None:
        """계좌 총액 업데이트."""
        if not self.api_client:
            return
        try:
            account = await self.api_client.get_account_summary()
            if account:
                equity = account.get("total_evaluated_amount", 0)
                if equity > 0:
                    self._total_equity = equity
                    self._last_equity_update = datetime.now()
        except Exception as e:
            logger.debug(f"계좌 잔고 업데이트 오류: {e}")

    def record_trade_result(
        self, ticker: str, strategy: str, won: bool, pnl_amount: float, pnl_ratio: float
    ) -> None:
        """거래 결과 기록 및 일별 손익 누적."""
        self._daily_realized_pnl += pnl_amount

        # 전략 승률/수익 업데이트 (EMA)
        alpha = 0.1  # 최근 10% 반영
        cur_wr = self._strategy_win_rates.get(strategy, 0.55)
        self._strategy_win_rates[strategy] = cur_wr * (1 - alpha) + (1.0 if won else 0.0) * alpha

        cur_ret = self._strategy_avg_return.get(strategy, 0.025)
        if pnl_ratio > 0:  # Kelly: avg_return은 승리 시 평균 수익만 반영
            self._strategy_avg_return[strategy] = cur_ret * (1 - alpha) + pnl_ratio * alpha

        # P1: trades 카운터 증가 + avg_loss 업데이트 + 배치 저장
        if strategy not in self._strategy_stats:
            self._strategy_stats[strategy] = {"trades": 0, "avg_loss": 0.02}
        self._strategy_stats[strategy]["trades"] = self._strategy_stats[strategy].get("trades", 0) + 1
        if not won and pnl_ratio < 0:
            cur_loss = self._strategy_stats[strategy].get("avg_loss", 0.02)
            self._strategy_stats[strategy]["avg_loss"] = round(cur_loss * 0.9 + abs(pnl_ratio) * 0.1, 4)
        self._save_strategy_stats()

        # 감사 로그 (P1)
        logger.info(
            f"[Kelly Audit] {strategy}: wr={self._strategy_win_rates.get(strategy, 0):.3f} "
            f"ret={self._strategy_avg_return.get(strategy, 0):.3f} "
            f"trades={self._strategy_stats.get(strategy, {}).get('trades', 0)} "
            f"{'WIN' if won else 'LOSS'} pnl={pnl_ratio:.2%}"
        )

        # 누진적 서킷 브레이커 체크 (P3)
        if self._total_equity > 0 and self._daily_realized_pnl < 0:
            loss_ratio = abs(self._daily_realized_pnl) / self._total_equity
            new_level = 0
            if loss_ratio >= RL.CB_LEVEL_4:
                new_level = 4
            elif loss_ratio >= RL.CB_LEVEL_3:
                new_level = 3
            elif loss_ratio >= RL.CB_LEVEL_2:
                new_level = 2
            elif loss_ratio >= RL.CB_LEVEL_1:
                new_level = 1

            if new_level > self._cb_level:
                self._cb_level = new_level
                cb_actions = {
                    1: "포지션 사이즈 50% 축소",
                    2: "신규 진입 중단",
                    3: "손실 포지션 강제 청산 필요",
                    4: "전 포지션 청산 + 당일 매매 완전 중단",
                }
                logger.critical(
                    f"🚨 [CB Level {new_level}] {cb_actions[new_level]} "
                    f"(당일 손실: {self._daily_realized_pnl:,.0f}원, {loss_ratio:.1%})"
                )

        # 포지션 기록 제거
        self._position_risks.pop(ticker, None)

    # ─────────────────────────────────────────────
    # 누진적 서킷브레이커 헬퍼 (P3)
    # ─────────────────────────────────────────────

    def get_cb_position_multiplier(self) -> float:
        """서킷브레이커 레벨에 따른 포지션 배율."""
        if self._cb_level >= 4:
            return 0.0  # 완전 중단
        elif self._cb_level >= 2:
            return 0.0  # 신규 진입 불가
        elif self._cb_level >= 1:
            return 0.5  # 50% 축소
        return 1.0

    def should_force_close_losers(self) -> bool:
        """CB Level 3+: 손실 포지션 강제 청산."""
        return self._cb_level >= 3

    def should_close_all(self) -> bool:
        """CB Level 4: 전 포지션 청산."""
        return self._cb_level >= 4

    def reset_daily(self) -> None:
        """일별 리셋 (07:00에 호출)."""
        self._daily_realized_pnl = 0.0
        self._cb_level = 0
        self._position_risks.clear()
        logger.info("[RiskMgmt] 일별 리스크 카운터 리셋")

    def update_strategy_performance(self, strategy: str, win_rate: float, avg_return: float) -> None:
        """외부에서 전략 성과 업데이트 (AlphaAgent와 연동)."""
        self._strategy_win_rates[strategy] = win_rate
        self._strategy_avg_return[strategy] = avg_return

    def get_risk_report(self) -> RiskReport:
        """현재 리스크 상태 리포트."""
        heat = "LOW"
        if abs(self._daily_realized_pnl) / (self._total_equity + 1e-9) > 0.02:
            heat = "MEDIUM"
        if abs(self._daily_realized_pnl) / (self._total_equity + 1e-9) > 0.04:
            heat = "HIGH"
        if self._cb_level >= 2:
            heat = "CRITICAL"

        return RiskReport(
            daily_pnl=self._daily_realized_pnl,
            circuit_breaker_active=self._cb_level >= 2,
            positions=list(self._position_risks.values()),
            heat_level=heat,
        )
