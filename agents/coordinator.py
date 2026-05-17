"""
AgentCoordinator — 에이전트 총괄 조율자 (두뇌)

모든 전문 에이전트를 통합 관리하고 최종 매매 결정을 내립니다:

  1. 에이전트 생명주기 관리 (start/stop)
  2. 시장 컨텍스트 공급 (MarketIntelligenceAgent → 전체)
  3. 신호 집계: AlphaAgent + SentimentAgent + RiskAgent 신호 가중 합산
  4. 최종 진입/청산 결정
  5. 포지션 사이징 결정 (Kelly Criterion 기반)
  6. 일별 성과 리포트 생성

의사결정 프로세스:
  Alpha 신호 (60%) + Sentiment 보정 (20%) + Risk/Market 필터 (20%)
  → 통합 신뢰도 > 임계값 → 진입 허가

에이전트 가중치 (성과에 따라 동적 조정):
  - AlphaAgent: 기본 60% (7개 전략의 핵심)
  - SentimentAgent: 기본 20%
  - MarketIntelAgent: 필터 역할 (가중치 없음, 차단/허용만)
  - RiskAgent: 서킷브레이커 역할
  - ExecutionAgent: 타이밍 보정
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import config as _cfg
from agents.base_agent import AgentSignal, MarketContext
from agents.market_intel_agent import MarketIntelligenceAgent
from agents.alpha_agent import AlphaGenerationAgent
from agents.risk_agent import RiskManagementAgent
from agents.sentiment_agent import SentimentAgent
from agents.execution_agent import ExecutionAgent
from agents.portfolio_agent import PortfolioAgent

logger = logging.getLogger("agent.coordinator")


# ─────────────────────────────────────────────
# 의사결정 임계값
# ─────────────────────────────────────────────

class DecisionThresholds:
    MIN_ALPHA_CONFIDENCE = 0.45       # Issue #24-H4: 0.55→0.45 (통과율 개선)
    MIN_COMBINED_CONFIDENCE = 0.50    # Issue #24-H4: 0.58→0.50 (7단계 AND 필터 완화)
    MIN_ALPHA_SCORE = 50.0            # Issue #24-H4: 60→50 (스크리너 임계값과 정렬)
    MAX_SIGNALS_PER_CYCLE = 5         # 한 사이클 최대 진입 신호
    ENTRY_COOLDOWN_SEC = 30           # 동일 종목 재진입 쿨다운


DT = DecisionThresholds()


@dataclass
class TradeDecision:
    """Coordinator의 최종 매매 결정."""
    action: str              # "BUY" | "SELL" | "HOLD"
    ticker: str
    quantity: int
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    strategy: str = ""
    combined_confidence: float = 0.0
    combined_score: float = 0.0
    signals_used: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    reason: str = ""

    def __str__(self) -> str:
        return (
            f"[{self.action}] {self.ticker} qty={self.quantity} "
            f"strategy={self.strategy} conf={self.combined_confidence:.2f}"
        )


class AgentCoordinator:
    """
    모든 에이전트를 통합 관리하는 최상위 조율자.
    AsyncAutoTrader가 이 클래스를 통해 매매 결정을 요청합니다.
    """

    def __init__(self, api_client=None, config=None, risk_manager=None, news_analyzer=None, strategy=None):
        self.api_client = api_client
        self.config = config
        self._strategy = strategy  # Step 11: Holdings 단일 소스 참조 (AsyncTradingStrategy)

        # 에이전트 인스턴스
        self.market_intel = MarketIntelligenceAgent(api_client, config)
        self.alpha        = AlphaGenerationAgent(api_client, config)
        self.risk         = RiskManagementAgent(api_client, config, base_risk_manager=risk_manager)
        self.sentiment    = SentimentAgent(api_client, config, news_analyzer=news_analyzer)
        self.execution    = ExecutionAgent(api_client, config)
        self.portfolio    = PortfolioAgent(api_client, config)

        self._all_agents = [
            self.market_intel, self.alpha, self.risk,
            self.sentiment, self.execution, self.portfolio,
        ]

        # 에이전트 가중치 (동적 조정)
        self._weights = {
            "AlphaGen":   0.60,
            "Sentiment":  0.20,
            "Execution":  0.10,
            "Portfolio":  0.10,
        }

        # 최근 진입 쿨다운 추적
        self._last_entry: Dict[str, datetime] = {}
        self._running = False

        # G1: 섹터 집중도 추적 (ticker → sector 매핑)
        self._ticker_sectors: Dict[str, str] = {}

    # ─────────────────────────────────────────────
    # 라이프사이클
    # ─────────────────────────────────────────────

    async def start(self) -> None:
        """모든 에이전트를 시작합니다."""
        self._running = True
        tasks = [agent.start() for agent in self._all_agents]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(
            f"✅ [Coordinator] {len(self._all_agents)}개 에이전트 시작 완료"
        )

    async def stop(self) -> None:
        """모든 에이전트를 종료합니다."""
        self._running = False
        tasks = [agent.stop() for agent in self._all_agents]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("⛔ [Coordinator] 모든 에이전트 종료")

    # ─────────────────────────────────────────────
    # 핵심: 매매 결정 생성
    # ─────────────────────────────────────────────

    async def generate_buy_decisions(
        self,
        candidates: List[Dict[str, Any]],
        current_holdings: Dict,
    ) -> List[TradeDecision]:
        """
        후보 종목 리스트를 분석해 매수 결정 리스트 반환.

        Flow:
          1. MarketIntel → 컨텍스트 확인 (필터)
          2. Alpha → 전략별 신호 수집
          3. Sentiment → 보정 점수
          4. Risk → 진입 허가 + 포지션 사이징
          5. Portfolio → 포트폴리오 적합성
          6. Execution → 타이밍 점수
          7. 통합 신뢰도 계산 → 임계값 초과 시 결정 반환
        """
        if not candidates:
            return []

        # 1. 시장 컨텍스트 취득
        context = self.market_intel.get_context()

        # Issue #24-H4: 극단 변동성/RISK 시 전면 차단 대신 로깅만 (포지션 사이징으로 대응)
        _is_extreme_volatile = (
            context.regime in ("VOLATILE", "VOLATILE_DOWN")
            or (context.regime == "VOLATILE_UP" and context.kospi_volatility > 0.45)
        )
        if _is_extreme_volatile and context.kospi_volatility > 0.45:
            logger.warning(
                f"[Coordinator] 고변동성({context.regime}, vol={context.kospi_volatility:.1%}) — 포지션 축소 적용"
            )
            # 전면 차단 제거: RiskManager의 position_size_multiplier가 처리

        # MarketIntel 신호 수집 — 시장폭 붕괴/극단 RISK 신호 체크
        market_signals = await self.market_intel.analyze(context, candidates)
        for sig in market_signals:
            if sig.signal_type == "RISK":
                logger.warning(
                    f"[Coordinator] 시장 RISK 신호 감지 (포지션 축소 적용): "
                    f"{sig.metadata.get('message', sig.strategy)}"
                )
                # Issue #24-H4: return [] 제거 — 전면 차단 대신 축소 진행

        # 2. Alpha 신호
        alpha_signals = await self.alpha.analyze(context, candidates)

        # 3. Sentiment 신호
        sentiment_signals = await self.sentiment.analyze(context, candidates)

        # 신호 그룹화 (ticker → signals)
        alpha_by_ticker = _group_by_ticker(alpha_signals)
        sentiment_by_ticker = _group_by_ticker(sentiment_signals)

        decisions: List[TradeDecision] = []

        for ticker, a_sigs in alpha_by_ticker.items():
            # Alpha 조건 필터
            best_alpha = max(a_sigs, key=lambda s: s.score)

            if best_alpha.score < DT.MIN_ALPHA_SCORE:
                continue
            if best_alpha.confidence < DT.MIN_ALPHA_CONFIDENCE:
                continue

            # 쿨다운 체크
            last = self._last_entry.get(ticker)
            if last and (datetime.now() - last).total_seconds() < DT.ENTRY_COOLDOWN_SEC:
                continue

            # Sentiment: 부정 뉴스 → 매수 차단 (S2: 6시간 감쇠 적용)
            s_sigs = sentiment_by_ticker.get(ticker, [])
            SENTIMENT_DECAY_HOURS = 6
            fresh_sell = any(
                s.signal_type == "SELL"
                and (datetime.now() - s.timestamp).total_seconds() < SENTIMENT_DECAY_HOURS * 3600
                for s in s_sigs
            )
            if fresh_sell:
                logger.info(f"[Coordinator] {ticker} 최근 부정 뉴스 ({SENTIMENT_DECAY_HOURS}h 이내) → 매수 제외")
                continue
            elif any(s.signal_type == "SELL" for s in s_sigs):
                logger.debug(f"[Coordinator] {ticker} 오래된 부정 뉴스 → 무시 (감쇠)")

            # 타이밍 점수
            timing_score, timing_desc = self.execution.get_entry_timing_score(best_alpha.strategy)
            timing_boost = (timing_score - 0.5) * 0.1  # -0.05 ~ +0.05

            # 통합 신뢰도 — Sentiment은 차단 필터, 가중치는 AlphaGen+Portfolio만 사용
            # 가중치 정규화: AlphaGen + Portfolio 합이 1.0 미만이면 보정
            w_alpha = self._weights["AlphaGen"]
            w_portfolio = self._weights["Portfolio"]
            w_sum = w_alpha + w_portfolio
            combined_conf = (
                best_alpha.confidence * (w_alpha / w_sum)
                + timing_boost
                + best_alpha.confidence * (w_portfolio / w_sum) * 0.5
            )
            combined_score = best_alpha.score

            if combined_conf < DT.MIN_COMBINED_CONFIDENCE:
                continue

            # G1: 후보 종목 섹터 추출 (alpha 신호 metadata 또는 후보 데이터에서)
            ticker_data_for_sector = next(
                (c for c in candidates if c.get("ticker") == ticker or c.get("code") == ticker), {}
            )
            sector = (
                best_alpha.metadata.get("sector")
                or ticker_data_for_sector.get("sector")
                or ""
            )

            # Risk: 진입 허가 체크 (G1 섹터 체크 포함)
            ok, reason = await self.risk.can_enter(
                ticker, best_alpha.entry_price or 0, best_alpha.strategy,
                context, current_holdings,
                sector=sector or None,
                ticker_sectors=self._ticker_sectors if sector else None,
            )
            if not ok:
                logger.debug(f"[Coordinator] {ticker} 리스크 거부: {reason}")
                continue

            # Portfolio: 적합성 체크
            pf_ok, pf_reason = self.portfolio.check_portfolio_fit(
                ticker, best_alpha.strategy, current_holdings
            )
            if not pf_ok:
                logger.debug(f"[Coordinator] {ticker} 포트폴리오 거부: {pf_reason}")
                continue

            # 포지션 사이징 (Kelly)
            ticker_data = next((c for c in candidates if c.get("ticker") == ticker or c.get("code") == ticker), {})
            ohlcv = ticker_data.get("_ohlcv_snapshot") or ticker_data.get("ohlcv")

            # GAP-02 수정: screener candidate에 "atr" 키 없음 → ohlcv에서 직접 계산
            atr = float(ticker_data.get("atr", 0))
            if atr == 0 and ohlcv is not None and len(ohlcv) >= 14:
                try:
                    high = ohlcv["high"]
                    low  = ohlcv["low"]
                    close_prev = ohlcv["close"].shift(1)
                    tr = (high - low).combine(
                        (high - close_prev).abs(), max
                    ).combine(
                        (low - close_prev).abs(), max
                    )
                    atr = float(tr.rolling(14).mean().iloc[-1])
                except Exception:
                    atr = 0

            price = best_alpha.entry_price or (
                float(ohlcv["close"].iloc[-1]) if ohlcv is not None and not ohlcv.empty else 0
            )
            if price <= 0:
                continue

            quantity = self.risk.calc_position_size(
                ticker=ticker,
                price=price,
                strategy=best_alpha.strategy,
                context=context,
                signal_confidence=combined_conf,
                atr=atr,
            )
            if quantity <= 0:
                continue

            # 손절가
            stop_price = best_alpha.stop_price or self.risk.calc_dynamic_stop(
                entry_price=price,
                atr=atr or price * 0.02,
                context=context,
                strategy=best_alpha.strategy,
            )

            decision = TradeDecision(
                action="BUY",
                ticker=ticker,
                quantity=quantity,
                entry_price=price,
                stop_price=round(stop_price, 0),
                target_price=round(best_alpha.target_price, 0),
                strategy=best_alpha.strategy,
                combined_confidence=round(combined_conf, 3),
                combined_score=round(combined_score, 1),
                signals_used=[str(s) for s in a_sigs[:3]],
                reason=f"Alpha={best_alpha.strategy} conf={combined_conf:.2f} timing={timing_desc}",
            )

            self._last_entry[ticker] = datetime.now()
            decisions.append(decision)

            # G1: 섹터 매핑 기록
            if sector:
                self._ticker_sectors[ticker] = sector

            logger.info(
                f"✅ [Coordinator] 매수 결정: {decision.ticker} "
                f"qty={quantity} strategy={best_alpha.strategy} "
                f"conf={combined_conf:.2f} score={combined_score:.1f}"
            )

            if len(decisions) >= DT.MAX_SIGNALS_PER_CYCLE:
                break

        # 통합 신뢰도 내림차순 정렬
        decisions.sort(key=lambda d: d.combined_confidence, reverse=True)
        return decisions

    async def generate_sell_decisions(
        self,
        current_holdings: Dict,
        context: Optional[MarketContext] = None,
    ) -> List[TradeDecision]:
        """
        현재 보유 포지션의 청산 결정 생성.
        Risk Agent의 신호 + 시장 컨텍스트 기반.
        """
        if context is None:
            context = self.market_intel.get_context()

        # Risk 에이전트 신호 수집
        risk_signals = await self.risk.analyze(context, [])

        decisions: List[TradeDecision] = []
        sell_tickers = set()

        for sig in risk_signals:
            if sig.signal_type == "SELL" and sig.ticker:
                sell_tickers.add(sig.ticker)
                decisions.append(TradeDecision(
                    action="SELL",
                    ticker=sig.ticker,
                    quantity=0,  # 전량 청산
                    strategy=sig.strategy,
                    combined_confidence=sig.confidence,
                    reason=f"Risk: {sig.strategy}",
                ))

        # 서킷브레이커 발동 시 전량 청산
        risk_report = self.risk.get_risk_report()
        if risk_report.circuit_breaker_active:
            for ticker, holding in current_holdings.items():
                if ticker not in sell_tickers:
                    decisions.append(TradeDecision(
                        action="SELL",
                        ticker=ticker,
                        quantity=0,
                        strategy="circuit_breaker",
                        combined_confidence=1.0,
                        reason="서킷브레이커 전량 청산",
                    ))

        return decisions

    # ─────────────────────────────────────────────
    # 거래 결과 피드백
    # ─────────────────────────────────────────────

    def on_trade_executed(
        self,
        ticker: str,
        action: str,
        strategy: str,
        price: float,
        quantity: int,
        pnl_ratio: float = 0.0,
        pnl_amount: float = 0.0,
    ) -> None:
        """
        실제 체결 후 모든 에이전트에 결과 피드백.
        이를 통해 에이전트들이 자신의 성과를 학습합니다.
        """
        if action == "BUY":
            self.portfolio.on_trade_open(ticker, strategy, price, quantity)
            # G1: 섹터 매핑 기록 (sector 정보가 있을 때만)
            # sector는 coordinator가 별도로 set하므로 여기서는 기존값 유지

        elif action == "SELL":
            # G1: 섹터 매핑 제거
            self._ticker_sectors.pop(ticker, None)
            won = pnl_ratio > 0

            # Alpha 에이전트 전략 성과 업데이트
            self.alpha.update_strategy_outcome(strategy, won, pnl_ratio)

            # Risk 에이전트 기록
            self.risk.record_trade_result(ticker, strategy, won, pnl_amount, pnl_ratio)

            # Portfolio 에이전트 기록
            self.portfolio.on_trade_close(ticker, strategy, pnl_amount, pnl_ratio)

            # S1: 전략별 성과 추적 (strategy_performance.json)
            self._update_strategy_performance(strategy, pnl_ratio, won=pnl_ratio > 0)

            # 에이전트 가중치 동적 조정 (성과 기반)
            self._adjust_weights()

            logger.info(
                f"📊 [Coordinator] 거래 결과: {ticker} {strategy} "
                f"{'✅ 수익' if won else '❌ 손실'} "
                f"{pnl_ratio:+.2%} ({pnl_amount:+,.0f}원)"
            )

    def notify_trade_event(self, event_type: str, ticker: str, sector: str = "") -> None:
        """Step 11: 진입/청산 이벤트 수신 — 섹터 매핑 동기화."""
        if event_type == "entry" and sector:
            self._ticker_sectors[ticker] = sector
        elif event_type == "exit":
            self._ticker_sectors.pop(ticker, None)

    def _update_strategy_performance(self, strategy: str, pnl_ratio: float, won: bool) -> None:
        """전략별 일일/누적 성과 업데이트 (S1)."""
        perf_file = os.path.join("data", "strategy_performance.json")
        try:
            perf = {}
            if os.path.exists(perf_file):
                with open(perf_file, "r") as f:
                    perf = json.load(f)
            if "daily" not in perf:
                perf["daily"] = {}
            if "cumulative" not in perf:
                perf["cumulative"] = {}

            # 일일 업데이트
            d = perf["daily"].setdefault(strategy, {"wins": 0, "losses": 0, "total_pnl": 0.0})
            d["wins" if won else "losses"] += 1
            d["total_pnl"] = round(d["total_pnl"] + pnl_ratio, 6)

            # 누적 업데이트
            c = perf["cumulative"].setdefault(strategy, {"wins": 0, "losses": 0, "total_pnl": 0.0})
            c["wins" if won else "losses"] += 1
            c["total_pnl"] = round(c["total_pnl"] + pnl_ratio, 6)
            total = c["wins"] + c["losses"]
            c["win_rate"] = round(c["wins"] / total, 3) if total > 0 else 0

            with open(perf_file, "w") as f:
                json.dump(perf, f, indent=2)
        except Exception as e:
            logger.warning(f"전략 성과 기록 실패: {e}")

    def _adjust_weights(self) -> None:
        """
        AlphaAgent 전략 성과에 따라 에이전트 가중치 동적 조정.
        Sentiment는 차단 필터 역할만 하므로 가중치 조정 대상에서 제외.
        """
        alpha_wr = self.alpha.win_rate
        # AlphaGen 가중치: 승률에 비례해 0.40~0.70 범위 조정
        self._weights["AlphaGen"] = max(0.40, min(0.70, alpha_wr * 0.70 + 0.30))

    # ─────────────────────────────────────────────
    # 보고
    # ─────────────────────────────────────────────

    def daily_report(self) -> Dict[str, Any]:
        """당일 에이전트 성과 종합 리포트."""
        return {
            "generated_at": datetime.now().isoformat(),
            "market_regime": self.market_intel.context.regime,
            "market_breadth": self.market_intel.context.breadth_score,
            "risk_status": self.risk.get_risk_report().__dict__,
            "portfolio": self.portfolio.daily_summary(),
            "alpha_strategies": self.alpha.strategy_stats(),
            "agent_performance": {
                a.name: a.performance_summary() for a in self._all_agents
            },
            "agent_weights": dict(self._weights),
            "execution": self.execution.execution_summary(),
        }

    def reset_daily(self, current_holdings: Optional[Dict] = None) -> None:
        """일별 리셋 (07:00에 호출).

        GAP-06 수정: 오버나이트 포지션 카운트 보존을 위해 current_holdings 전달.
        async_trader.py에서 호출 시 self.holdings를 넘겨줘야 함.
        """
        self.risk.reset_daily()
        self.portfolio.reset_daily(current_holdings=current_holdings)
        self._last_entry.clear()
        logger.info("[Coordinator] 일별 전체 리셋 완료")

    # ─────────────────────────────────────────────
    # 마켓 컨텍스트 퍼블릭 접근
    # ─────────────────────────────────────────────

    def get_market_context(self) -> MarketContext:
        return self.market_intel.get_context()

    def get_regime_multiplier(self) -> float:
        """현재 시장 체제에 따른 포지션 배율 반환 (config 기반)."""
        try:
            ctx = self.market_intel.get_context()
            regime = str(ctx.regime.value) if (ctx and hasattr(ctx.regime, "value")) else (
                str(ctx.regime) if ctx else "NORMAL"
            )
            rp = _cfg.get_regime_params(regime)
            return float(rp.get("position_size_multiplier", 1.0))
        except Exception:
            return 1.0


# ─────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────

def _group_by_ticker(signals: List[AgentSignal]) -> Dict[str, List[AgentSignal]]:
    """신호 리스트를 ticker별 dict로 그룹화."""
    result: Dict[str, List[AgentSignal]] = {}
    for sig in signals:
        if sig.ticker:
            result.setdefault(sig.ticker, []).append(sig)
    return result
