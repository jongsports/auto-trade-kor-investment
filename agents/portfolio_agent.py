"""
PortfolioAgent — 포트폴리오 관리 에이전트

포트폴리오 전체 최적화를 담당:
  1. 섹터 집중도 관리 — 동일 섹터 과집중 방지
  2. 전략 다양성 — 단일 전략 의존도 제한
  3. 자본 효율성 — 유휴 자본 최소화
  4. 수익 보호 — 누적 수익 구간별 포지션 축소 권장
  5. 일별 P&L 리포팅
"""

import logging
import math
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent, AgentSignal, MarketContext

logger = logging.getLogger("agent.portfolio")


class PortfolioAgent(BaseAgent):
    """포트폴리오 수준의 관리와 최적화를 담당합니다."""

    def __init__(self, api_client=None, config=None):
        super().__init__("Portfolio", api_client, config)

        # 전략 포지션 수 추적
        self._strategy_count: Dict[str, int] = defaultdict(int)
        # 일별 P&L 내역
        self._daily_trades: List[Dict] = []
        # 총 수익률 (log returns)
        self._cumulative_pnl: float = 0.0
        self._peak_pnl: float = 0.0
        # Step 9: 실제 자산(equity) 추적 — 1.0 기준 비율
        self._equity: float = 1.0

    async def initialize(self) -> None:
        logger.info("[Portfolio] 포트폴리오 에이전트 초기화 완료")

    async def analyze(
        self, context: MarketContext, candidates: List[Dict]
    ) -> List[AgentSignal]:
        """포트폴리오 수준 신호 발행."""
        signals: List[AgentSignal] = []

        # 누적 수익 높을 때 포지션 축소 권고
        if self._cumulative_pnl > 0.05:  # 5% 수익 달성
            drawdown = (self._peak_pnl - self._cumulative_pnl) / (self._peak_pnl + 1e-9)
            if drawdown > 0.02:  # 고점 대비 2% 후퇴
                signals.append(self._make_signal(
                    "RISK", None, confidence=0.65,
                    strategy="profit_protection",
                    metadata={
                        "cumulative_pnl": self._cumulative_pnl,
                        "drawdown": drawdown,
                        "message": "누적 수익 보호: 신규 진입 자제"
                    },
                ))

        return signals

    def check_portfolio_fit(
        self,
        ticker: str,
        strategy: str,
        holdings: Dict,
        sector: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        신규 포지션이 포트폴리오에 적합한지 확인.
        Returns: (허용 여부, 사유)
        """
        # 동일 전략 최대 2개 (전략 다양성 확보)
        max_per_strategy = 2
        if self._strategy_count.get(strategy, 0) >= max_per_strategy:
            return False, f"동일 전략({strategy}) 최대 포지션 초과"

        # 섹터 집중도 (미래 개선: 섹터 정보 활용)
        if sector:
            sector_count = sum(
                1 for h in holdings.values()
                if h.get("sector") == sector
            )
            if sector_count >= 2:
                return False, f"섹터({sector}) 최대 2개 포지션 초과"

        return True, "OK"

    def on_trade_open(self, ticker: str, strategy: str, price: float, quantity: int) -> None:
        """포지션 진입 기록."""
        self._strategy_count[strategy] = self._strategy_count.get(strategy, 0) + 1
        logger.debug(f"[Portfolio] 진입: {ticker} ({strategy}) @ {price:,.0f} × {quantity}")

    def on_trade_close(
        self,
        ticker: str,
        strategy: str,
        pnl_amount: float,
        pnl_ratio: float,
    ) -> None:
        """포지션 청산 기록 및 P&L 누적."""
        self._strategy_count[strategy] = max(
            0, self._strategy_count.get(strategy, 0) - 1
        )
        self._daily_trades.append({
            "ticker": ticker,
            "strategy": strategy,
            "pnl_amount": pnl_amount,
            "pnl_ratio": pnl_ratio,
            "closed_at": datetime.now().isoformat(),
        })
        # S3: log(1+r) — 비대칭 보정 (단순 합산 대비 정확한 누적 수익률)
        self._cumulative_pnl += math.log1p(pnl_ratio)
        self._equity *= (1 + pnl_ratio)  # Step 9: 실제 자산 추적
        self._peak_pnl = max(self._peak_pnl, self._cumulative_pnl)

        # HWM 기반 drawdown 체크
        drawdown = self._peak_pnl - self._cumulative_pnl
        if drawdown > 0.02:
            logger.warning(
                f"[Portfolio] HWM 경고: drawdown={drawdown:.2%} "
                f"(peak={self._peak_pnl:.2%}, current={self._cumulative_pnl:.2%})"
            )

    def daily_summary(self) -> Dict[str, Any]:
        """일별 거래 요약."""
        if not self._daily_trades:
            return {"trades": 0, "total_pnl": 0.0, "win_rate": 0.0}

        wins = sum(1 for t in self._daily_trades if t["pnl_ratio"] > 0)
        total_pnl = sum(t["pnl_amount"] for t in self._daily_trades)
        n = len(self._daily_trades)

        by_strategy = defaultdict(lambda: {"count": 0, "pnl": 0.0})
        for t in self._daily_trades:
            by_strategy[t["strategy"]]["count"] += 1
            by_strategy[t["strategy"]]["pnl"] += t["pnl_ratio"]

        return {
            "trades": n,
            "wins": wins,
            "win_rate": round(wins / n, 3),
            "total_pnl_amount": round(total_pnl, 0),
            "by_strategy": dict(by_strategy),
        }

    def reset_daily(self, current_holdings: Optional[Dict] = None) -> None:
        """일별 리셋.

        GAP-06 수정: 오버나이트 포지션이 있을 때 _strategy_count를 완전히 초기화하면
        해당 포지션이 카운트에서 누락되어 전략당 최대 2개 제한이 우회됨.
        current_holdings를 받아서 오버나이트 포지션의 전략 카운트를 재구성.
        """
        self._daily_trades.clear()
        self._strategy_count.clear()

        # 오버나이트 포지션의 전략 카운트 재구성
        if current_holdings:
            for holding in current_holdings.values():
                strat = holding.get("reason", "")
                if strat:
                    self._strategy_count[strat] = self._strategy_count.get(strat, 0) + 1
            logger.info(
                f"[Portfolio] 일별 포트폴리오 리셋 — 오버나이트 포지션 {len(current_holdings)}개 "
                f"전략 카운트 재구성: {dict(self._strategy_count)}"
            )
        else:
            logger.info("[Portfolio] 일별 포트폴리오 리셋")
