import asyncio
import sys
import logging
from data.async_news_analyzer import AsyncNewsAnalyzer

logging.basicConfig(level=logging.INFO)

async def test_news():
    analyzer = AsyncNewsAnalyzer()
    print("Testing get_recent_news for Samsung Electronics (005930)...")
    news_list = await analyzer.get_recent_news("005930", "삼성전자", days=3)
    
    print(f"\nFound {len(news_list)} news items:")
    for n in news_list[:5]:
        print(f"[{n['date']}] {n['title']}")
        
    if len(news_list) > 5:
        print("...")
        
    print("\nTesting analyze_stock_news...")
    result = await analyzer.analyze_stock_news("005930", "삼성전자", days=3)
    print(f"Result: {result}")
    
    await analyzer.close()

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_news())
