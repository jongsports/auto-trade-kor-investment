"""
BaseAgent — 모든 에이전트의 추상 기반 클래스.

에이전트는 각자의 전문 영역에서 독립적으로 분석하며,
AgentCoordinator가 신호를 취합해 최종 매매 결정을 내립니다.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────

@dataclass
class AgentSignal:
    """에이전트가 발행하는 매매 신호."""
    agent_name: str
    signal_type: str          # "BUY" | "SELL" | "HOLD" | "RISK" | "INFO"
    ticker: Optional[str]
    confidence: float         # 0.0 ~ 1.0
    score: float = 0.0        # 전략별 점수
    strategy: str = ""        # 발동 전략 이름
    entry_price: float = 0.0  # 권장 진입가 (0 = 시장가)
    target_price: float = 0.0 # 목표가
    stop_price: float = 0.0   # 손절가
    quantity_ratio: float = 1.0  # 기본 포지션 대비 배율 (0.5 ~ 2.0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        return (
            f"[{self.agent_name}] {self.signal_type} {self.ticker or 'MARKET'} "
            f"strategy={self.strategy} conf={self.confidence:.2f} score={self.score:.1f}"
        )


@dataclass
class MarketContext:
    """
    MarketIntelligenceAgent가 유지하는 공유 시장 컨텍스트.
    모든 에이전트가 읽기 전용으로 참조합니다.
    """
    # 시장 체제
    regime: str = "NORMAL"        # "BULL" | "BEAR" | "NORMAL" | "VOLATILE"
    regime_confidence: float = 0.5

    # 변동성
    kospi_volatility: float = 0.0  # 연간화 변동성
    kosdaq_volatility: float = 0.0

    # 시장 폭 (Market Breadth)
    breadth_score: float = 50.0    # 0(극약세) ~ 100(극강세)
    advance_decline_ratio: float = 1.0  # 상승/하락 종목 비율

    # 수급 (단위: 억 원)
    foreign_flow_bn: float = 0.0   # 외국인 순매수
    institution_flow_bn: float = 0.0  # 기관 순매수
    program_flow_bn: float = 0.0   # 프로그램 순매수

    # 섹터
    sector_leaders: List[str] = field(default_factory=list)  # 강세 섹터
    sector_laggards: List[str] = field(default_factory=list)  # 약세 섹터
    hot_themes: List[str] = field(default_factory=list)       # 핫 테마

    # 상태
    is_market_open: bool = False
    updated_at: Optional[datetime] = None

    def is_bullish(self) -> bool:
        return self.regime in ("BULL", "NORMAL") and self.breadth_score >= 50

    def is_risk_off(self) -> bool:
        return self.regime == "BEAR" or self.kospi_volatility > 0.35

    def net_institutional_flow(self) -> float:
        """외국인 + 기관 합산 순매수."""
        return self.foreign_flow_bn + self.institution_flow_bn


# ─────────────────────────────────────────────
# 성과 기록
# ─────────────────────────────────────────────

@dataclass
class SignalOutcome:
    """에이전트 신호의 사후 성과 기록."""
    signal: AgentSignal
    entry_price: float
    exit_price: float
    profit_ratio: float
    hold_time_min: float
    closed_at: datetime = field(default_factory=datetime.now)

    @property
    def is_win(self) -> bool:
        return self.profit_ratio > 0


# ─────────────────────────────────────────────
# 기반 에이전트 클래스
# ─────────────────────────────────────────────

class BaseAgent(ABC):
    """
    모든 에이전트의 추상 기반 클래스.

    구현 시 반드시 override:
      - initialize(): 초기화 로직
      - analyze(): 시장 분석 → 신호 리스트 반환
    """

    def __init__(self, name: str, api_client=None, config=None):
        self.name = name
        self.api_client = api_client
        self.config = config
        self.logger = logging.getLogger(f"agent.{name}")

        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._signal_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

        # 성과 추적
        self._outcomes: List[SignalOutcome] = []
        self._total_signals: int = 0

    # ── 추상 메서드 ──────────────────────────────

    @abstractmethod
    async def initialize(self) -> None:
        """에이전트 초기화 (상태 로드, 연결 확인 등)."""

    @abstractmethod
    async def analyze(
        self,
        context: MarketContext,
        candidates: List[Dict[str, Any]],
    ) -> List[AgentSignal]:
        """
        시장 상황과 후보 종목을 분석해 신호 리스트 반환.

        Args:
            context: MarketIntelligenceAgent가 제공하는 시장 컨텍스트
            candidates: 스크리너가 제공하는 후보 종목 dict 리스트
        Returns:
            AgentSignal 리스트 (빈 리스트 허용)
        """

    # ── 라이프사이클 ─────────────────────────────

    async def start(self) -> None:
        self._running = True
        await self.initialize()
        self.logger.info(f"[{self.name}] 에이전트 시작 ✅")

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self.logger.info(f"[{self.name}] 에이전트 종료 ⛔")

    # ── 신호 발행/수신 ────────────────────────────

    def emit(self, signal: AgentSignal) -> None:
        """신호를 큐에 게시."""
        self._total_signals += 1
        try:
            self._signal_queue.put_nowait(signal)
            self.logger.debug(f"신호 발행: {signal}")
        except asyncio.QueueFull:
            self.logger.warning(f"신호 큐 가득참 — 드롭: {signal}")

    async def drain_signals(self) -> List[AgentSignal]:
        """큐의 모든 신호를 수거해 리스트로 반환."""
        signals: List[AgentSignal] = []
        while not self._signal_queue.empty():
            try:
                signals.append(self._signal_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return signals

    # ── 백그라운드 태스크 ─────────────────────────

    def _spawn(self, coro) -> asyncio.Task:
        """백그라운드 코루틴 태스크 생성 및 추적."""
        task = asyncio.create_task(coro)
        self._tasks.append(task)
        task.add_done_callback(
            lambda t: self._tasks.remove(t) if t in self._tasks else None
        )
        return task

    async def _safe_loop(self, coro_factory, interval_sec: float, name: str) -> None:
        """
        예외가 발생해도 `interval_sec` 후 재시도하는 안전한 루프.
        에이전트가 중단될 때까지 계속 실행됩니다.
        """
        while self._running:
            try:
                await coro_factory()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"[{self.name}] {name} 오류 (재시도): {e}", exc_info=True)
            await asyncio.sleep(interval_sec)

    # ── 성과 추적 ─────────────────────────────────

    def record_outcome(self, outcome: SignalOutcome) -> None:
        """신호 성과 기록 (최근 200건 유지)."""
        self._outcomes.append(outcome)
        if len(self._outcomes) > 200:
            self._outcomes = self._outcomes[-200:]

    @property
    def win_rate(self) -> float:
        if not self._outcomes:
            return 0.5
        return sum(1 for o in self._outcomes if o.is_win) / len(self._outcomes)

    @property
    def avg_return(self) -> float:
        if not self._outcomes:
            return 0.0
        return sum(o.profit_ratio for o in self._outcomes) / len(self._outcomes)

    @property
    def sharpe_like(self) -> float:
        """단순 수익/변동성 비율 (Sharpe 대용)."""
        if len(self._outcomes) < 5:
            return 0.0
        returns = [o.profit_ratio for o in self._outcomes]
        import statistics
        mean_r = statistics.mean(returns)
        std_r = statistics.stdev(returns) if len(returns) > 1 else 1e-9
        return mean_r / (std_r + 1e-9)

    def performance_summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_signals": self._total_signals,
            "evaluated": len(self._outcomes),
            "win_rate": round(self.win_rate, 3),
            "avg_return": round(self.avg_return, 4),
            "sharpe_like": round(self.sharpe_like, 3),
        }

    # ── 헬퍼 ──────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    def _make_signal(
        self,
        signal_type: str,
        ticker: Optional[str],
        confidence: float,
        score: float = 0.0,
        strategy: str = "",
        **kwargs,
    ) -> AgentSignal:
        """편의 팩토리 메서드."""
        return AgentSignal(
            agent_name=self.name,
            signal_type=signal_type,
            ticker=ticker,
            confidence=confidence,
            score=score,
            strategy=strategy,
            **kwargs,
        )
