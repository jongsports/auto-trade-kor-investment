"""
주식 뉴스 비동기 감성 분석기.

네이버 뉴스 검색으로 최신 기사를 수집하고, 가중치 키워드 기반 감성 분석을 수행합니다.

설계 원칙:
- HTML 파싱: 네이버 뉴스 전용 CSS 클래스 우선, 폴백 셀렉터 사용
- 날짜: 상대(3분 전 / 1시간 전) 및 절대(2025.03.02) 형식 모두 파싱
- 감성: 가중치 키워드 + 부정어 문맥 감지 (키워드 앞 10자 내)
- 점수: count_score(0~5) + sentiment_score(0~5) = 0~10
- 네트워크: 재시도 2회, 타임아웃 8초, 네이버 적합 헤더
"""
import asyncio
import logging
import re
import urllib.parse
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger("auto_trade.news_analyzer")


class AsyncNewsAnalyzer:
    """주식 뉴스 비동기 감성 분석기."""

    # ── 가중치 키워드 (값이 클수록 강한 시그널) ─────────────────────────────
    POSITIVE: Dict[str, int] = {
        # 주가 강세
        "상한가": 3, "급등": 2, "52주 신고가": 3, "신고가": 2,
        # 실적
        "흑자전환": 3, "실적 개선": 3, "실적개선": 3, "호실적": 2, "실적 호전": 2,
        "영업이익 증가": 2, "순이익 증가": 2, "턴어라운드": 2,
        # 계약·수주
        "수주": 2, "계약 체결": 2, "계약체결": 2, "공급 계약": 2, "공급계약": 2,
        # M&A·사업
        "인수합병": 2, "신사업": 1, "사업 확대": 1,
        # 목표가·투자의견
        "목표가 상향": 3, "목표가상향": 3, "매수 의견": 2, "투자의견 상향": 3,
        # 주주환원
        "자사주 소각": 2, "자사주소각": 2, "무상증자": 2, "배당 확대": 2, "배당확대": 2,
        # 승인·특허
        "FDA 승인": 3, "FDA승인": 3, "신약 승인": 3, "식약처 승인": 3, "특허": 1,
        # 기타
        "수혜": 1, "강세": 1, "상승 돌파": 2,
    }

    NEGATIVE: Dict[str, int] = {
        # 주가 약세
        "하한가": 3, "급락": 2, "52주 신저가": 2, "신저가": 2,
        # 실적
        "적자전환": 3, "실적 악화": 3, "실적악화": 3, "영업손실": 2, "순손실": 2,
        "손실 확대": 2, "손실확대": 2, "실적 부진": 2,
        # 계약·소송
        "계약 해지": 2, "계약해지": 2, "소송": 2,
        # 경영 리스크
        "횡령": 3, "배임": 3, "사기": 3, "압수수색": 3, "검찰": 2, "제재": 2,
        # 주식 발행
        "유상증자": 2,
        # 상장 관련
        "상장폐지": 3, "거래 정지": 3, "거래정지": 3, "영업 정지": 3, "영업정지": 3,
        # 약한 부정
        "공매도": 1, "부진": 1, "불확실": 1, "우려": 1, "하락세": 1, "약세": 1,
        "파산": 3, "워크아웃": 3, "법정관리": 3,
    }

    # 부정어 패턴 (키워드 앞 10자 내에서 감지 시 해당 키워드 무효)
    NEGATION_PATTERNS: List[str] = ["아니", "않", "없", "못", "반대", "부정", "아님"]

    def __init__(self, api_client=None):
        self.api_client = api_client
        self.session: Optional[aiohttp.ClientSession] = None

    # ── 세션 관리 ────────────────────────────────────────────────────────

    async def init_session(self) -> None:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=False),
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
                    "Referer": "https://www.naver.com",
                },
            )

    async def close(self) -> None:
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    # ── 날짜 파싱 ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_naver_date(date_str: str) -> Optional[datetime]:
        """
        네이버 뉴스 날짜 문자열 → datetime 변환.

        지원 형식:
          - "3분 전"   / "1시간 전"  / "어제"
          - "2025.03.02."  /  "2025.03.02 14:30"
        """
        now = datetime.now()
        s = date_str.strip()

        # 상대 시간: N분 전
        m = re.search(r"(\d+)분\s*전", s)
        if m:
            return now - timedelta(minutes=int(m.group(1)))

        # 상대 시간: N시간 전
        m = re.search(r"(\d+)시간\s*전", s)
        if m:
            return now - timedelta(hours=int(m.group(1)))

        # 어제
        if "어제" in s:
            return now - timedelta(days=1)

        # 절대 날짜: YYYY.MM.DD (시분초 선택)
        m = re.search(r"(\d{4})\.(\d{2})\.(\d{2})", s)
        if m:
            try:
                return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass

        return None

    # ── HTML 파싱 ────────────────────────────────────────────────────────

    def _extract_news_from_html(
        self, html: str, cutoff: datetime
    ) -> List[Dict[str, str]]:
        """
        네이버 뉴스 검색 결과 HTML에서 제목·날짜 추출.

        셀렉터 전략 (우선순위):
          1. <a class="news_tit"> — 현행 네이버 뉴스 검색 구조
          2. 특정 data-clk 속성이 있는 <a> — 대안 구조
          3. 폴백: 길이 ≥15, 부모에 date 스팬 있는 <a> — 구조 변경 대비
        """
        soup = BeautifulSoup(html, "html.parser")
        results: List[Dict[str, str]] = []
        seen: set = set()

        # 전략 1·2: 네이버 전용 클래스
        title_tags = soup.find_all("a", class_=re.compile(r"news_tit"))
        if not title_tags:
            # 전략 2: data-clk 속성 (네이버 클릭 추적)
            title_tags = [
                a for a in soup.find_all("a", attrs={"data-clk": True})
                if len(a.get_text(strip=True)) >= 15
            ]
        if not title_tags:
            # 전략 3: 폴백 — href가 http로 시작하고 길이 ≥15인 <a>
            title_tags = [
                a for a in soup.find_all("a", href=re.compile(r"^https?://"))
                if len(a.get_text(strip=True)) >= 15
            ]

        for a_tag in title_tags:
            title = a_tag.get_text(strip=True)
            if len(title) < 10 or title in seen:
                continue
            seen.add(title)

            # 게재일 추출: 부모 컨테이너에서 날짜 span 탐색
            pub_date: Optional[datetime] = None
            container = a_tag.find_parent(
                class_=re.compile(r"bx|news_wrap|news_area|item|article")
            )
            if container:
                date_span = container.find(
                    "span", class_=re.compile(r"date|time|info|press")
                )
                if date_span:
                    pub_date = self._parse_naver_date(date_span.get_text(strip=True))

            # 날짜 미추출 시 현재 시각으로 폴백
            if pub_date is None:
                pub_date = datetime.now()

            # cutoff 이전 기사 제외
            if pub_date < cutoff:
                continue

            results.append({
                "title": title,
                "date": pub_date.strftime("%Y.%m.%d"),
            })

        return results

    # ── 뉴스 수집 ────────────────────────────────────────────────────────

    async def get_recent_news(
        self, ticker: str, name: str, days: int = 1
    ) -> List[Dict[str, str]]:
        """
        네이버 뉴스 검색으로 최신 기사 수집.

        - 최신순 정렬 (&sort=1)
        - 재시도 2회, 타임아웃 8초
        - days 기간 이전 기사는 수집 즉시 중단
        """
        await self.init_session()
        cutoff = datetime.now() - timedelta(days=days)
        news_list: List[Dict[str, str]] = []
        max_pages = 3
        max_retries = 2

        for page in range(1, max_pages + 1):
            start_offset = (page - 1) * 10 + 1
            url = (
                "https://search.naver.com/search.naver"
                f"?where=news&query={urllib.parse.quote(name)}"
                f"&start={start_offset}&sort=1"  # sort=1: 최신순
            )

            html = ""
            for attempt in range(max_retries):
                try:
                    async with self.session.get(
                        url, timeout=aiohttp.ClientTimeout(total=8)
                    ) as resp:
                        if resp.status != 200:
                            logger.debug(
                                f"[{ticker}] 뉴스 HTTP {resp.status} (p{page})"
                            )
                            break
                        html = await resp.text(encoding="utf-8", errors="replace")
                    break  # 성공
                except asyncio.TimeoutError:
                    logger.debug(
                        f"[{ticker}] 뉴스 타임아웃 (p{page}, attempt {attempt + 1})"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1.0)
                except Exception as e:
                    logger.warning(f"[{ticker}] 뉴스 수집 오류 (p{page}): {e}")
                    break

            if not html:
                break

            batch = self._extract_news_from_html(html, cutoff)
            news_list.extend(batch)

            # 이번 페이지 결과가 적으면 다음 페이지 불필요
            if len(batch) < 5:
                break

            await asyncio.sleep(0.3)

        logger.debug(f"[{ticker}] 뉴스 수집: {len(news_list)}건 (name={name})")
        return news_list

    # ── 감성 분석 ────────────────────────────────────────────────────────

    def analyze_sentiment(self, text: str) -> float:
        """
        가중치 키워드 기반 감성 점수 계산.

        - 부정어(아니/않/없 등)가 키워드 앞 10자 내에 있으면 해당 키워드 무효
        - 반환: -1.0 (매우 부정) ~ +1.0 (매우 긍정), 키워드 없으면 0.0
        """
        if not text:
            return 0.0

        pos_score = 0
        neg_score = 0

        for kw, weight in self.POSITIVE.items():
            idx = text.find(kw)
            if idx == -1:
                continue
            context_before = text[max(0, idx - 10): idx]
            if any(neg in context_before for neg in self.NEGATION_PATTERNS):
                continue  # 부정어 감지 → 이 키워드 무효
            pos_score += weight

        for kw, weight in self.NEGATIVE.items():
            idx = text.find(kw)
            if idx == -1:
                continue
            context_before = text[max(0, idx - 10): idx]
            if any(neg in context_before for neg in self.NEGATION_PATTERNS):
                continue  # "소송 없다" 류 → 이 키워드 무효
            neg_score += weight

        total = pos_score + neg_score
        if total == 0:
            return 0.0
        return (pos_score - neg_score) / total

    # ── 점수 계산 ────────────────────────────────────────────────────────

    def calculate_news_score(self, news_list: List[Dict[str, str]]) -> float:
        """
        뉴스 감성 점수 0~10 계산.

        구성:
          count_score     : 0~5점 (기사 1건당 1점, 최대 5건)
          sentiment_score : 0~5점 (평균 감성 × 5, 부정 감성은 0으로 처리)

        예시:
          기사 3건 + 평균 감성 0.6 → count=3, sentiment=3.0 → 합계 6.0
          기사 2건 + 평균 감성 1.0 → count=2, sentiment=5.0 → 합계 7.0
        """
        if not news_list:
            return 0.0

        count_score = min(5.0, float(len(news_list)))

        sentiments = [self.analyze_sentiment(n["title"]) for n in news_list]
        avg_sentiment = sum(sentiments) / len(sentiments)
        # 부정 감성(-1~0)은 0으로 처리 (별도 패널티 없음 — 기사 count는 유지)
        sentiment_score = max(0.0, avg_sentiment) * 5.0

        return round(min(10.0, count_score + sentiment_score), 2)

    # ── 단일 종목 분석 ───────────────────────────────────────────────────

    async def analyze_stock_news(
        self, ticker: str, name: str, days: int = 1
    ) -> dict:
        """
        단일 종목 뉴스 수집 + 감성 분석.

        Returns:
            {name, code, news_mentions, sentiment, score}
            score는 0~10 범위.
        """
        news_list = await self.get_recent_news(ticker, name, days)
        if not news_list:
            return {
                "name": name,
                "code": ticker,
                "news_mentions": 0,
                "sentiment": 0.0,
                "score": 0.0,
            }

        sentiments = [self.analyze_sentiment(n["title"]) for n in news_list]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        score = self.calculate_news_score(news_list)

        logger.debug(
            f"[{ticker}] 뉴스: {len(news_list)}건 | 감성: {avg_sentiment:.3f} | 점수: {score}"
        )
        return {
            "name": name,
            "code": ticker,
            "news_mentions": len(news_list),
            "sentiment": round(avg_sentiment, 3),
            "score": score,
        }

    # ── 다종목 병렬 분석 ─────────────────────────────────────────────────

    async def select_stocks_by_news_and_theme(
        self,
        stock_list: List[tuple],
        news_days: int = 1,
        top_n: int = 20,
    ) -> List[dict]:
        """
        여러 종목의 뉴스 감성을 비동기 병렬 분석 후 상위 종목 반환.

        Args:
            stock_list: [(ticker, name), ...] 형태 목록
            news_days:  최근 N일 뉴스 대상
            top_n:      score 기준 상위 N개 반환
        """
        logger.info(f"뉴스 분석 시작: {len(stock_list)}개 종목, {news_days}일치")
        tasks = [
            self.analyze_stock_news(code, name, news_days)
            for code, name in stock_list
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid = [
            r for r in results
            if isinstance(r, dict) and r.get("score", 0) > 0
        ]
        valid.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"뉴스 분석 완료: {len(valid)}/{len(stock_list)}개 양성 신호")
        return valid[:top_n]
