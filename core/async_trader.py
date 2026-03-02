import logging
import asyncio
import os
from datetime import datetime, time

import config
from core.trader_api import AsyncKisAPI
from strategy.async_screener import AsyncStockScreener
from data.async_news_analyzer import AsyncNewsAnalyzer
from utils.notifier import AsyncTelegramNotifier
# Assuming RiskManager and TradingStrategy are either adapted or we run them in run_in_executor
# We'll use a mocked sync-wrapper for them for this refactor demo.
from risk.async_risk_manager import AsyncRiskManager as RiskManager
from strategy.async_trading_strategy import AsyncTradingStrategy as TradingStrategy

logger = logging.getLogger("auto_trade.auto_trader")

class AsyncAutoTrader:
    def __init__(self, demo_mode=True):
        self.logger = config.setup_logging()
        self.demo_mode = demo_mode
        self.api_client = AsyncKisAPI(
            app_key=config.APP_KEY,
            app_secret=config.APP_SECRET,
            account_number=config.CANO,
            demo_mode=self.demo_mode
        )
        self.screener = AsyncStockScreener(self.api_client)
        self.news_analyzer = AsyncNewsAnalyzer(self.api_client)
        self.screener.news_analyzer = self.news_analyzer  # 뉴스 분석기 주입
        self.notifier = AsyncTelegramNotifier()
        
        # Note: In a fully refactored system, RiskManager and TradingStrategy should also be async.
        # But we will use the existing sync ones by wrapping them if they aren't fully async yet.
        self.risk_manager = RiskManager(self.api_client)
        self.strategy = TradingStrategy(self.api_client, self.risk_manager)
        
        self.running = False
        self.candidate_stocks = []

    async def start(self):
        logger.info("비동기 자동 매매 엔진을 시작합니다.")
        await self.notifier.send_message("🚀 <b>자동 매매 엔진 시작</b>\n- 모드: " + ("모의투자" if self.demo_mode else "실전투자"))
        self.running = True
        
        # Connect API 
        self.api_client.connect()
        await self.api_client.init_session()
        
        # Start background tasks
        self.tasks = [
            asyncio.create_task(self._scheduled_morning_routine()),
            asyncio.create_task(self._monitor_loop())
        ]
        
        await asyncio.gather(*self.tasks)
            
    async def stop(self):
        self.running = False
        for task in getattr(self, "tasks", []):
            task.cancel()
        await self.api_client.close()
        logger.info("엔진이 중지되었습니다.")
        await self.notifier.send_message("🛑 <b>자동 매매 엔진 중지</b>")

    async def _scheduled_morning_routine(self):
        """Run daily routines at specific times."""
        while self.running:
            now = datetime.now()
            
            # Very simplistic scheduling loop. In production, use apscheduler or precise wait times.
            if now.hour == 8 and now.minute == 0:
                logger.info("Executing 08:00 AM Screening...")
                await self._run_screening()
                await asyncio.sleep(60) # Prevent multiple triggers

            elif now.hour == 9 and now.minute == 0:
                logger.info("Executing 09:00 AM Market Open Entry...")
                await self._morning_entry()
                await asyncio.sleep(60)

            elif now.hour == 15 and now.minute == 30:
                logger.info("Executing 15:30 Market Close Report...")
                # await self._generate_closing_report()
                await asyncio.sleep(60)
                
            await asyncio.sleep(10) # Check every 10 seconds

    async def _monitor_loop(self):
        """Continuous parallel monitoring for open positions and news."""
        while self.running:
            now = datetime.now()
            # Only monitor during market hours (09:00 - 15:30)
            if 9 <= now.hour < 15 or (now.hour == 15 and now.minute <= 30):
                # 1. Price/Position monitoring (Every 10 seconds)
                # In real scenario: self.strategy.check_and_execute_orders_async()
                logger.debug("Checking positions...")
                
                # 2. News monitoring (Every 30 mins)
                if now.minute in [0, 30] and now.second < 10:
                    logger.info("Background news monitoring...")
                    news_candidates = await self.screener.analyze_news_for_candidates(days=0.04)
                    if news_candidates:
                         urgent_news = [s for s in news_candidates if s["score"] > 80]
                         if urgent_news:
                              logger.warning(f"긴급 뉴스 발견: {len(urgent_news)}건")
                              # Handle urgent news
                              
            await asyncio.sleep(10)

    async def _run_screening(self):
        logger.info("종목 스크리닝 시작 (Async)")
        self.candidate_stocks = await self.screener.run_screening_async(["KOSPI", "KOSDAQ"])
        self.strategy.set_candidate_stocks(self.candidate_stocks)
        logger.info(f"선별 완료: {len(self.candidate_stocks)}종목")
        
        # 선별 결과 저장 (JSON)
        await self._save_screening_results()

    async def _save_screening_results(self):
        """스크리닝 결과를 JSON 파일로 저장합니다."""
        try:
            import json
            data = {
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "DEMO" if self.demo_mode else "REAL",
                "count": len(self.candidate_stocks),
                "candidates": self.candidate_stocks
            }
            
            with open(config.SCREENING_RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            logger.info(f"스크리닝 결과가 저장되었습니다: {config.SCREENING_RESULTS_FILE}")
        except Exception as e:
            logger.error(f"스크리닝 결과 저장 중 오류 발생: {e}")

    async def _morning_entry(self):
        # A full async wrapper to replace the _morning_entry
        logger.info("시초가 매매 로직 실행")
        if not self.candidate_stocks:
             return
             
        top = self.candidate_stocks[:3]
        for c in top:
            ticker = c["ticker"]
            # To avoid blocking, run strategy logic in thread pool if it's strictly sync
            # await asyncio.to_thread(self.strategy.entry, ticker)
            logger.info(f"매수 시도: {ticker}")
