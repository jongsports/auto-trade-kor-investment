"""
ExecutionAgent — 체결 최적화 에이전트

매수/매도 체결 품질을 극대화하는 에이전트:
  1. VWAP 기반 진입 타이밍 최적화 — 평균 매수단가 절감
  2. 유동성 점검 — 슬리피지 사전 추정
  3. 분할 매수/매도 전략 — 충격비용 최소화
  4. 체결 슬리피지 추적 및 리포트
  5. 최적 거래 시간대 분석 (한국 시장 패턴)

한국 단타 매매의 최적 체결 패턴:
  - 09:00~09:30: 동시호가 + 시초가 — 갭 전략
  - 09:30~10:30: 모멘텀 가속 — 추격 진입 위험 구간
  - 10:30~13:00: 조정 구간 — 눌림목 매수 최적
  - 13:00~14:30: 재상승 — 2차 파동 포착
  - 14:30~15:20: 정리 — 청산 집중
  - 15:20~15:30: 동시호가 — 오버나이트만
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, time
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent, AgentSignal, MarketContext

logger = logging.getLogger("agent.execution")


# 한국 시장 시간대별 특성 (9:00 = 540분)
TIME_ZONE_PROFILES = {
    "opening_momentum":   (time(9, 0),  time(9, 30),  0.90, "시초가 모멘텀"),
    "chasing_danger":     (time(9, 30), time(10, 0),  0.60, "추격 매수 위험"),
    "primary_momentum":   (time(10, 0), time(10, 30), 0.85, "1차 모멘텀"),
    "dip_buy_optimal":    (time(10, 30),time(13, 0),  0.95, "눌림목 최적"),
    "secondary_wave":     (time(13, 0), time(14, 30), 0.80, "2차 파동"),
    "closing_cleanup":    (time(14, 30),time(15, 20), 0.70, "청산 집중"),
    "overnight_only":     (time(15, 20),time(15, 30), 0.40, "오버나이트만"),
}


class ExecutionAgent(BaseAgent):
    """체결 품질 최적화 및 슬리피지 최소화 에이전트."""

    def __init__(self, api_client=None, config=None):
        super().__init__("Execution", api_client, config)

        # 슬리피지 추적 (전략 → 슬리피지 목록)
        self._slippage_records: Dict[str, List[float]] = defaultdict(list)
        self._execution_log: List[Dict] = []

        # Step 13: 시간대별 슬리피지 버킷 (M3)
        self._slippage_by_hour: Dict[str, List[float]] = {
            "09:00-09:30": [],
            "09:30-10:30": [],
            "10:30-14:00": [],
            "14:00-15:30": [],
        }

    async def initialize(self) -> None:
        logger.info("[Execution] 체결 최적화 에이전트 초기화 완료")

    async def analyze(
        self, context: MarketContext, candidates: List[Dict]
    ) -> List[AgentSignal]:
        """체결 타이밍 및 유동성 경고 신호 발행."""
        signals: List[AgentSignal] = []
        now = datetime.now().time()

        # 추격 매수 위험 구간 경고
        if time(9, 30) <= now <= time(9, 55):
            signals.append(self._make_signal(
                "INFO", None, confidence=0.7,
                strategy="chasing_danger",
                metadata={"message": "추격 매수 위험 구간 — 신규 진입 자제"},
            ))

        return signals

    def get_entry_timing_score(self, strategy: str) -> Tuple[float, str]:
        """
        현재 시각 기준 진입 타이밍 점수.
        Returns: (0~1 점수, 설명)
        """
        now = datetime.now().time()
        for zone_name, (start, end, score, desc) in TIME_ZONE_PROFILES.items():
            if start <= now < end:
                # 전략별 보정
                if strategy in ("S3_gap_momentum", "S4_limit_up_chase") and zone_name == "opening_momentum":
                    score = min(score + 0.05, 1.0)
                if strategy == "S2_dip_buy" and zone_name == "dip_buy_optimal":
                    score = min(score + 0.05, 1.0)
                return score, desc
        return 0.5, "시장 외 시간"

    def estimate_slippage(self, ticker: str, quantity: int, price: float, volume_ma20: float) -> float:
        """
        예상 슬리피지 추정 (%).
        거래량이 적을수록, 수량이 많을수록 슬리피지 증가.
        """
        if volume_ma20 <= 0:
            return 0.005  # 0.5% 기본값

        # 거래 비중 (당일 평균 거래량 대비)
        trade_ratio = quantity / volume_ma20
        slippage = trade_ratio * 0.02  # 1% 비중 → 0.02% 슬리피지
        return min(slippage, 0.01)  # 최대 1% 슬리피지

    def recommend_split_orders(
        self, total_qty: int, price: float, volume_ma20: float
    ) -> List[Dict]:
        """
        분할 주문 추천.
        거래량 대비 비중이 크면 분할 주문으로 충격 비용 절감.
        """
        if volume_ma20 <= 0:
            return [{"qty": total_qty, "delay_sec": 0}]

        trade_ratio = total_qty / volume_ma20

        if trade_ratio < 0.01:  # 1% 미만 → 한 번에
            return [{"qty": total_qty, "delay_sec": 0}]
        elif trade_ratio < 0.05:  # 1~5% → 2분할
            half = total_qty // 2
            return [
                {"qty": half, "delay_sec": 0},
                {"qty": total_qty - half, "delay_sec": 30},
            ]
        else:  # 5% 이상 → 3분할
            third = total_qty // 3
            return [
                {"qty": third, "delay_sec": 0},
                {"qty": third, "delay_sec": 60},
                {"qty": total_qty - 2 * third, "delay_sec": 120},
            ]

    def record_execution(
        self,
        ticker: str,
        strategy: str,
        order_price: float,
        fill_price: float,
        quantity: int,
        action: str,  # "BUY" or "SELL"
    ) -> float:
        """체결 슬리피지 기록 및 반환."""
        if order_price == 0:
            return 0.0

        slippage = (fill_price - order_price) / order_price
        if action == "BUY":
            slippage = abs(slippage)  # 매수: 높게 체결 = 손실
        else:
            slippage = -min(slippage, 0)  # 매도: 낮게 체결 = 손실

        self._slippage_records[strategy].append(slippage)

        # Step 13: 시간대별 버킷에도 기록
        bucket = self._get_time_bucket(datetime.now().strftime("%H:%M"))
        self._slippage_by_hour[bucket].append(slippage)

        self._execution_log.append({
            "ticker": ticker,
            "strategy": strategy,
            "action": action,
            "order_price": order_price,
            "fill_price": fill_price,
            "slippage_pct": round(slippage * 100, 4),
            "quantity": quantity,
            "time": datetime.now().isoformat(),
        })

        if slippage > 0.003:  # 0.3% 이상 슬리피지 경고
            logger.warning(
                f"[Execution] ⚠️ 높은 슬리피지: {ticker} {action} "
                f"주문={order_price:,} 체결={fill_price:,} ({slippage:.2%})"
            )
        return slippage

    def _get_time_bucket(self, now_str: str) -> str:
        """HH:MM 형식 시각을 시간대 버킷으로 변환."""
        if now_str < "09:30":
            return "09:00-09:30"
        elif now_str < "10:30":
            return "09:30-10:30"
        elif now_str < "14:00":
            return "10:30-14:00"
        return "14:00-15:30"

    def get_recommended_order_type(self) -> str:
        """시간대별 슬리피지 기반 주문 방식 추천.
        슬리피지 0.3% 이상이면 지정가, 미만이면 시장가 추천.
        """
        import numpy as np
        now_str = datetime.now().strftime("%H:%M")
        bucket = self._get_time_bucket(now_str)
        records = self._slippage_by_hour[bucket]
        avg_slip = float(np.mean(records)) if records else 0.0
        return "LIMIT" if avg_slip > 0.003 else "MARKET"

    def execution_summary(self) -> Dict:
        """체결 품질 요약."""
        summary = {}
        for strategy, slippages in self._slippage_records.items():
            if slippages:
                summary[strategy] = {
                    "count": len(slippages),
                    "avg_slippage_pct": round(sum(slippages) / len(slippages) * 100, 3),
                    "max_slippage_pct": round(max(slippages) * 100, 3),
                }
        return summary
