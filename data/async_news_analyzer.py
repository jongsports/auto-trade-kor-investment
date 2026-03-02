import logging
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import urllib.parse
from typing import List, Dict, Any

logger = logging.getLogger("auto_trade.news_analyzer")

class AsyncNewsAnalyzer:
    """Async News Sensing and Theme Analyzer."""

    def __init__(self, api_client=None):
        self.api_client = api_client
        self.session = None

        # Positive keywords for sentiment analysis
        self.positive_keywords = [
            '상한가', '급등', '실적', '흑자', '수주', '계약', '인수', '합병',
            '신사업', '공급', '돌파', '강세', '상승세', '목표가', '수혜',
            '배당', '자사주', '소각', '무상증자', 'FDA', '승인', '출시'
        ]

        # Negative keywords for sentiment analysis
        self.negative_keywords = [
            '하한가', '급락', '적자', '해지', '취소', '소송', '횡령', '배임',
            '유상증자', '하락세', '약세', '공매도', '매도', '주의', '경고',
            '정지', '상장폐지', '제재', '조사', ' 압수수색'
        ]

    async def init_session(self):
        if self.session is None or self.session.closed:
             connector = aiohttp.TCPConnector(ssl=False)
             self.session = aiohttp.ClientSession(
                connector=connector,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
             )

    async def close(self):
         if self.session and not self.session.closed:
             await self.session.close()

    async def get_recent_news(self, ticker: str, name: str, days: int = 1) -> List[Dict[str, str]]:
        """Fetch news titles asynchronously from Naver Search."""
        await self.init_session()
        news_list = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        start_date_str = start_date.strftime("%Y.%m.%d")
        
        # Searching naver news by name
        page = 1
        max_pages = 2
        
        try:
             while page <= max_pages:
                  start_offset = (page - 1) * 10 + 1
                  url = f"https://search.naver.com/search.naver?where=news&query={urllib.parse.quote(name)}&start={start_offset}"
                  async with self.session.get(url, timeout=5) as response:
                       if response.status != 200:
                            break
                       
                       html = await response.text(encoding="utf-8", errors="replace")
                       soup = BeautifulSoup(html, "html.parser")
                       
                       # Extract all links that look like news titles (length > 15, contains name or related keywords)
                       for a in soup.find_all('a'):
                           title = a.text.strip()
                           if len(title) > 15 and (name in title or "주가" in title or "실적" in title):
                               # To avoid duplicates
                               if not any(n["title"] == title for n in news_list):
                                   news_list.append({
                                       "title": title,
                                       "date": end_date.strftime("%Y.%m.%d") # Approximate date for search results
                                   })
                       page += 1
                       await asyncio.sleep(0.5)
        except Exception as e:
             logger.warning(f"Error fetching news for {name} ({ticker}): {e}")
             
        return news_list

    def analyze_sentiment(self, text: str) -> float:
        """Calculate sentiment score based on keyword matching."""
        if not text:
            return 0.0
            
        pos_count = sum(1 for keyword in self.positive_keywords if keyword in text)
        neg_count = sum(1 for keyword in self.negative_keywords if keyword in text)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
            
        return (pos_count - neg_count) / total

    async def analyze_stock_news(self, ticker: str, name: str, days: int = 1) -> dict:
        """Analyze news for a single stock asynchronously."""
        news_list = await self.get_recent_news(ticker, name, days)
        if not news_list:
             return {"name": name, "code": ticker, "news_mentions": 0, "sentiment": 0.0, "score": 0.0}
             
        sentiment_scores = [self.analyze_sentiment(n["title"]) for n in news_list]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        
        return {
            "name": name,
            "code": ticker,
            "news_mentions": len(news_list),
            "sentiment": avg_sentiment,
            "score": len(news_list) * 2 + (avg_sentiment * 10)  # simple heuristic score
        }

    async def select_stocks_by_news_and_theme(self, news_days=1, market_cap_min=100, top_n=20, markets=["KOSPI", "KOSDAQ"]):
         """Select top stocks based on News Sentiment. Currently mocks grabbing the market ticker list."""
         logger.info(f"Starting async news analysis for past {news_days} days.")
         # Since we do not have DB or real list, we mock top stocks
         mock_stocks = [
              ("005930", "삼성전자"), ("000660", "SK하이닉스"), ("035420", "NAVER"),
              ("035720", "카카오"), ("051910", "LG화학"), ("005380", "현대차"),
              ("293490", "카카오게임즈"), ("086520", "에코프로")
         ]
         
         tasks = [self.analyze_stock_news(code, name, news_days) for code, name in mock_stocks]
         results = await asyncio.gather(*tasks)
         
         # Sort by score
         valid = [r for r in results if r["score"] > 0]
         valid.sort(key=lambda x: x["score"], reverse=True)
         
         return valid[:top_n]
