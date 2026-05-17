"""
SentimentAgent — 뉴스 부정 키워드 필터 에이전트

KIS API 기반 뉴스 제목을 분석하여 위험 종목을 차단합니다:
  - 부정 키워드 감지 시 SELL 신호 발행 → 매수 차단
  - BUY 신호 없음 (긍정 뉴스로 신뢰도 조작 금지)
  - DART 공시 모니터링 제거 (ticker=None 버그 원인, 비활성화)
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List

from agents.base_agent import BaseAgent, AgentSignal, MarketContext

logger = logging.getLogger("agent.sentiment")


class SentimentAgent(BaseAgent):
    """뉴스 부정 키워드 기반 위험 종목 필터 에이전트."""

    def __init__(self, api_client=None, config=None, news_analyzer=None):
        super().__init__("Sentiment", api_client, config)
        self.news_analyzer = news_analyzer

        # 분석 캐시 (ticker → 마지막 결과, TTL 15분)
        self._analysis_cache: Dict[str, Dict] = {}
        self._cache_ttl_min = 15

    async def initialize(self) -> None:
        # GAP-08 수정: news_analyzer=None 시 명시적 경고 (무음 비활성화 방지)
        if self.news_analyzer is None:
            logger.warning(
                "[Sentiment] news_analyzer가 None — 뉴스 기반 부정 필터링 비활성화됨. "
                "부정 뉴스 종목도 매수 차단 없이 통과합니다."
            )
        else:
            logger.info("[Sentiment] 뉴스 부정 키워드 필터 에이전트 초기화 완료")

    async def analyze(
        self, context: MarketContext, candidates: List[Dict]
    ) -> List[AgentSignal]:
        """후보 종목의 뉴스 부정 키워드 체크."""
        if not candidates or not self.news_analyzer:
            return []

        tasks = [self._check_ticker_news(c) for c in candidates]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        signals: List[AgentSignal] = []
        for result in results:
            if isinstance(result, list):
                signals.extend(result)
        return signals

    async def _check_ticker_news(self, data: Dict) -> List[AgentSignal]:
        """단일 종목 뉴스 부정 키워드 체크."""
        ticker = data.get("ticker") or data.get("code", "")
        name   = data.get("name", ticker)
        if not ticker:
            return []

        # 캐시 확인
        cached = self._analysis_cache.get(ticker)
        if cached:
            age_min = (datetime.now() - cached["updated"]).total_seconds() / 60
            if age_min < self._cache_ttl_min:
                return cached.get("signals", [])

        signals: List[AgentSignal] = []
        try:
            result = await asyncio.wait_for(
                self.news_analyzer.analyze_stock_news(ticker, name),
                timeout=5.0,
            )
            if result and result.get("has_negative"):
                hits = result.get("negative_hits", [])
                signals.append(self._make_signal(
                    "SELL", ticker,
                    confidence=0.75,
                    score=80.0,
                    strategy="news_negative_keyword",
                    metadata={
                        "negative_hits": hits,
                        "news_count": result.get("news_mentions", 0),
                    },
                ))
                logger.info(
                    f"[Sentiment] {ticker}({name}) 부정 뉴스 → SELL 신호 | 키워드: {hits}"
                )
        except asyncio.TimeoutError:
            logger.debug(f"[Sentiment] {ticker} 뉴스 조회 타임아웃")
        except Exception as e:
            logger.debug(f"[Sentiment] {ticker} 뉴스 분석 오류: {e}")

        self._analysis_cache[ticker] = {
            "signals": signals,
            "updated": datetime.now(),
        }
        return signals
