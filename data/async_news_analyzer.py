"""
주식 뉴스 분석기 (KIS API 기반).

KIS OpenAPI의 news-title 엔드포인트(FHKST01011800)를 사용하여
종목 관련 뉴스 제목을 조회하고, 부정 키워드 필터링으로 위험 종목을 식별합니다.

Naver 크롤링 방식 → KIS 공식 API 방식으로 교체:
  - HTML 파싱 불안정 / robots.txt 위반 위험 제거
  - 긍정 감성 부스트 제거 (불필요한 신뢰도 조작 방지)
  - 부정 키워드 감지만 수행 → SELL 시그널 트리거
"""
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("auto_trade.news_analyzer")


class AsyncNewsAnalyzer:
    """KIS API 기반 뉴스 부정 키워드 필터."""

    # 부정 키워드 (종목 매수를 차단해야 할 리스크 사건)
    NEGATIVE: Dict[str, int] = {
        # 주가 약세
        "하한가": 3, "급락": 2, "52주 신저가": 2, "신저가": 2,
        # 실적 악화
        "적자전환": 3, "실적 악화": 3, "실적악화": 3, "영업손실": 2, "순손실": 2,
        "손실 확대": 2, "손실확대": 2, "실적 부진": 2,
        # 계약·소송
        "계약 해지": 2, "계약해지": 2, "소송": 2,
        # 경영 리스크
        "횡령": 3, "배임": 3, "사기": 3, "압수수색": 3, "검찰": 2, "제재": 2,
        # 주식 발행 (희석 압박)
        "유상증자": 2,
        # 상장·거래 위험
        "상장폐지": 3, "거래 정지": 3, "거래정지": 3, "영업 정지": 3, "영업정지": 3,
        # 재정 위기
        "파산": 3, "워크아웃": 3, "법정관리": 3,
        # 약한 부정 (단독 출현 시에만 고려)
        "공매도": 1, "부진": 1,
    }

    def __init__(self, api_client=None):
        self.api_client = api_client

    async def init_session(self) -> None:
        """KIS api_client가 세션을 관리하므로 별도 세션 불필요."""
        pass

    async def close(self) -> None:
        pass

    def check_negative_keywords(self, titles: List[str]) -> Tuple[bool, List[str]]:
        """
        뉴스 제목 리스트에서 부정 키워드 탐지.

        Returns:
            (부정 키워드 존재 여부, 발견된 키워드 목록)
        """
        hits: List[str] = []
        for title in titles:
            for kw in self.NEGATIVE:
                if kw in title and kw not in hits:
                    hits.append(kw)
        return bool(hits), hits

    async def analyze_stock_news(
        self, ticker: str, name: str, days: int = 1
    ) -> dict:
        """
        KIS API로 종목 뉴스 제목 조회 + 부정 키워드 필터링.

        Args:
            ticker: 종목코드
            name:   종목명 (로깅용)
            days:   (미사용, API 호환성 유지)

        Returns:
            {
              name, code, news_mentions, sentiment, score,
              has_negative, negative_hits
            }
            score: 부정 키워드 있으면 -5.0, 없으면 0.0
        """
        titles: List[str] = []
        if self.api_client and hasattr(self.api_client, "get_news_titles"):
            try:
                titles = await self.api_client.get_news_titles(ticker)
            except Exception as e:
                logger.warning(f"뉴스 조회 실패 {ticker}: {e}")

        if not titles:
            return {
                "name": name, "code": ticker,
                "news_mentions": 0, "sentiment": 0.0, "score": 0.0,
                "has_negative": False, "negative_hits": [],
            }

        has_negative, hits = self.check_negative_keywords(titles)
        score = -5.0 if has_negative else 0.0

        if hits:
            logger.info(f"[뉴스] {ticker}({name}) 부정 키워드 감지: {hits}")
        else:
            # 한시적 INFO 승격 (2026-04-24): 스크래핑 실제 동작 가시화
            logger.info(f"[뉴스] {ticker}: {len(titles)}건 — 부정 키워드 없음")

        return {
            "name": name, "code": ticker,
            "news_mentions": len(titles),
            "sentiment": -1.0 if has_negative else 0.0,
            "score": score,
            "has_negative": has_negative,
            "negative_hits": hits,
        }

    # ── 하위 호환성 메서드 (스크리너에서 직접 호출 시) ──────────────────────

    def analyze_sentiment(self, text: str) -> float:
        """단순 부정 키워드 체크. 있으면 -1.0, 없으면 0.0."""
        for kw in self.NEGATIVE:
            if kw in text:
                return -1.0
        return 0.0

    def calculate_news_score(self, news_list: List[Dict]) -> float:
        """하위 호환: 부정 키워드 있으면 -5, 없으면 0."""
        titles = [n.get("title", "") for n in news_list if isinstance(n, dict)]
        has_neg, _ = self.check_negative_keywords(titles)
        return -5.0 if has_neg else 0.0
