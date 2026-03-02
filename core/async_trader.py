import logging
import asyncio
import json
import os
from datetime import datetime, time

import config
from core.trader_api import AsyncKisAPI
from strategy.async_screener import AsyncStockScreener
from data.async_news_analyzer import AsyncNewsAnalyzer
from utils.notifier import AsyncTelegramNotifier
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
        self.screener.news_analyzer = self.news_analyzer
        self.notifier = AsyncTelegramNotifier()

        self.risk_manager = RiskManager(self.api_client)
        self.strategy = TradingStrategy(self.api_client, self.risk_manager)

        self.running = False
        self.candidate_stocks = []

        # 중복 트리거 방지 — 당일 실행 완료 이벤트 기록
        self._triggered: set = set()
        self._last_trigger_date: str = ""

    # ------------------------------------------------------------------ #
    # 라이프사이클
    # ------------------------------------------------------------------ #

    async def start(self):
        logger.info("비동기 자동 매매 엔진을 시작합니다.")
        await self.notifier.send_message(
            "🚀 <b>자동 매매 엔진 시작</b>\n- 모드: " + ("모의투자" if self.demo_mode else "실전투자")
        )
        self.running = True

        self.api_client.connect()
        await self.api_client.init_session()

        # 1. 초기 시장 리스크 평가 (issue #6-C: assess_market_risk 미호출 수정)
        await self.risk_manager.assess_market_risk()

        # 2. 재시작 시 당일 스크리닝 결과 복원 (issue #7-C)
        await self._load_screening_results()

        self.tasks = [
            asyncio.create_task(self._scheduled_morning_routine()),
            asyncio.create_task(self._monitor_loop()),
        ]
        await asyncio.gather(*self.tasks)

    async def stop(self):
        self.running = False
        for task in getattr(self, "tasks", []):
            task.cancel()
        await self.api_client.close()
        logger.info("엔진이 중지되었습니다.")
        await self.notifier.send_message("🛑 <b>자동 매매 엔진 중지</b>")

    # ------------------------------------------------------------------ #
    # 중복 트리거 방지 헬퍼
    # ------------------------------------------------------------------ #

    def _should_trigger(self, label: str) -> bool:
        """당일 해당 이벤트가 아직 실행되지 않았으면 True."""
        today = datetime.now().strftime("%Y%m%d")
        if self._last_trigger_date != today:
            self._triggered = set()
            self._last_trigger_date = today
        return f"{today}_{label}" not in self._triggered

    def _mark_triggered(self, label: str):
        """이벤트를 당일 실행 완료로 기록."""
        today = datetime.now().strftime("%Y%m%d")
        self._triggered.add(f"{today}_{label}")

    # ------------------------------------------------------------------ #
    # 다단계 동적 스크리닝 스케줄러 (issue #8)
    # ------------------------------------------------------------------ #

    async def _scheduled_morning_routine(self):
        """다단계 동적 스크리닝 스케줄러.

        시간    단계                설명
        -----   ----------------    -------------------------------------------
        07:00   시장 리스크 평가    KOSPI 변동성·추세 분석
        08:00   사전 스크리닝       전날 종가 기반 후보 선정 (매수 보류)
        09:05   시초가 검증 매수    갭 필터 통과 후 진입
        10:30   장중 모멘텀 탐색    당일 거래량·모멘텀 신규 발굴 & 추가 매수
        13:30   오후 모멘텀 탐색    지속 모멘텀 종목 추가 편입
        15:10   오버나이트 진입     overnight 보너스 스크리닝 후 진입 (issue #6-A)
        15:30   장 마감 리포트      당일 성과 텔레그램 전송
        """
        while self.running:
            now = datetime.now()
            now_str = now.strftime("%H:%M")

            # 07:00 — 시장 리스크 재평가
            if now_str == "07:00" and self._should_trigger("07:00"):
                self._mark_triggered("07:00")
                logger.info("07:00 시장 리스크 평가 시작...")
                await self.risk_manager.assess_market_risk()

            # 08:00 — 사전 스크리닝 (매수 보류)
            elif now_str == "08:00" and self._should_trigger("08:00"):
                self._mark_triggered("08:00")
                logger.info("08:00 사전 스크리닝 시작...")
                await self._premarket_screening()

            # 09:05 — 시초가 갭 검증 + 첫 매수
            elif now_str == "09:05" and self._should_trigger("09:05"):
                self._mark_triggered("09:05")
                logger.info("09:05 시초가 검증 매수 시작...")
                await self._opening_validation_and_entry()

            # 10:30, 13:30 — 장중 모멘텀 스크리닝 & 추가 매수
            elif now_str in ("10:30", "13:30") and self._should_trigger(now_str):
                self._mark_triggered(now_str)
                logger.info(f"{now_str} 장중 모멘텀 스크리닝 시작...")
                await self._intraday_screening_and_entry()

            # 15:10 — 오버나이트 진입 (issue #6-A)
            elif now_str == "15:10" and self._should_trigger("15:10"):
                self._mark_triggered("15:10")
                logger.info("15:10 오버나이트 진입 시작...")
                await self._overnight_entry()

            # 15:30 — 장 마감 리포트
            elif now_str == "15:30" and self._should_trigger("15:30"):
                self._mark_triggered("15:30")
                logger.info("15:30 장 마감 리포트 생성...")
                await self._generate_closing_report()

            await asyncio.sleep(10)

    # ------------------------------------------------------------------ #
    # 모니터링 루프
    # ------------------------------------------------------------------ #

    async def _monitor_loop(self):
        """보유 포지션 청산 조건 + 뉴스 모니터링 (장 중 매 10초)."""
        while self.running:
            now = datetime.now()
            # 09:00~15:30 사이에만 실행
            if 9 <= now.hour < 15 or (now.hour == 15 and now.minute <= 30):
                # 1. 청산 조건 체크
                await self._check_exit_conditions()

                # 2. 뉴스 긴급도 모니터링 (30분마다)
                if now.minute in (0, 30) and now.second < 10:
                    logger.info("Background news monitoring...")
                    try:
                        if self.candidate_stocks and self.screener.news_analyzer:
                            urgent = []
                            for c in self.candidate_stocks[:10]:
                                ticker = c.get("ticker", "")
                                name = c.get("name", ticker)
                                if not ticker:
                                    continue
                                news_res = await self.screener.news_analyzer.analyze_stock_news(
                                    ticker, name, days=0.04
                                )
                                if news_res.get("score", 0) >= 8:
                                    urgent.append({"ticker": ticker, "news_score": news_res["score"]})
                            if urgent:
                                logger.warning(f"긴급 뉴스 발견: {len(urgent)}건 {urgent}")
                    except Exception as e:
                        logger.error(f"뉴스 모니터링 오류: {e}")

            await asyncio.sleep(10)

    # ------------------------------------------------------------------ #
    # 스크리닝 단계별 메서드
    # ------------------------------------------------------------------ #

    async def _premarket_screening(self):
        """08:00 사전 스크리닝: 전날 종가 기반 후보 선정 (매수 보류)."""
        logger.info("사전 스크리닝 시작 — 09:05 갭 검증 후 진입")
        self.candidate_stocks = await self.screener.run_screening_async(["KOSPI", "KOSDAQ"])
        self.strategy.set_candidate_stocks(self.candidate_stocks)
        logger.info(f"사전 스크리닝 완료: {len(self.candidate_stocks)}종목 후보")
        await self._save_screening_results()
        await self.notifier.send_message(
            f"📋 <b>사전 스크리닝 완료</b>\n"
            f"- 후보: {len(self.candidate_stocks)}종목\n"
            f"- 리스크: {self.risk_manager.risk_status} | 시장: {self.risk_manager.market_condition}"
        )

    async def _opening_validation_and_entry(self):
        """09:05 시초가 갭 검증 후 진입."""
        logger.info("시초가 검증 매수 로직 실행")

        if not self.candidate_stocks:
            logger.warning("사전 스크리닝 결과 없음 — 즉시 재스크리닝")
            await self._premarket_screening()

        await self.strategy.update_holdings()

        # 오버나이트 제외, 갭 필터 적용
        daytrade_candidates = [c for c in self.candidate_stocks if c.get("reason") != "Overnight"]
        validated = await self.screener.validate_opening_candidates(daytrade_candidates)
        logger.info(f"갭 필터 후 {len(validated)}종목 진입 대상")

        await self._execute_entries(validated[:config.MAX_STOCKS])

    async def _intraday_screening_and_entry(self):
        """10:30/13:30 장중 모멘텀 스크리닝 & 추가 매수.

        이 시간대에 run_screening_async()를 호출하면 OHLCV 마지막 행이
        당일 실시간 데이터를 포함해 실제 모멘텀을 반영한다.
        """
        logger.info("장중 모멘텀 스크리닝 시작")
        new_candidates = await self.screener.run_screening_async(["KOSPI", "KOSDAQ"], is_intraday=True)  # Issue #9-B

        # 기존 후보에 신규 합산 (중복 제거, 점수 내림차순)
        existing_tickers = {c["ticker"] for c in self.candidate_stocks}
        for c in new_candidates:
            if c["ticker"] not in existing_tickers:
                self.candidate_stocks.append(c)
                existing_tickers.add(c["ticker"])

        self.candidate_stocks.sort(key=lambda x: x.get("score", 0), reverse=True)
        self.strategy.set_candidate_stocks(self.candidate_stocks)
        await self._save_screening_results()
        logger.info(f"장중 스크리닝 완료: 총 {len(self.candidate_stocks)}종목")

        await self.strategy.update_holdings()
        held = set(self.strategy.holdings.keys())
        targets = [
            c for c in self.candidate_stocks
            if c.get("reason") != "Overnight" and c["ticker"] not in held
        ]
        await self._execute_entries(targets[:config.MAX_STOCKS])

    async def _overnight_entry(self):
        """15:10 오버나이트 진입.

        이 시간대(15:10~15:20)에 run_screening_async()를 실행하면
        스크리너가 is_overnight_window=True로 overnight 보너스를 적용.
        """
        logger.info("오버나이트 진입 로직 실행")
        overnight_candidates = await self.screener.run_screening_async(["KOSPI", "KOSDAQ"])
        overnight_only = [c for c in overnight_candidates if c.get("reason") == "Overnight"]

        if not overnight_only:
            logger.info("오버나이트 후보 없음")
            return

        logger.info(f"오버나이트 후보 {len(overnight_only)}종목")
        await self.strategy.update_holdings()
        held = set(self.strategy.holdings.keys())
        targets = [c for c in overnight_only if c["ticker"] not in held]
        await self._execute_entries(targets[:config.MAX_STOCKS])

    # ------------------------------------------------------------------ #
    # 공통 매수 실행 로직
    # ------------------------------------------------------------------ #

    async def _execute_entries(self, candidates: list):
        """후보 종목 매수 실행 공통 로직."""
        for c in candidates:
            ticker = c.get("ticker", "")
            reason = c.get("reason", "Momentum")
            if not ticker:
                continue
            try:
                ohlcv = await self.api_client.get_ohlcv(ticker, count=30)
                if not ohlcv.empty:
                    can_enter = await self.strategy.check_entry_condition(ticker, ohlcv)
                    if can_enter:
                        result = await self.strategy.entry(ticker, reason=reason)
                        if result and result.get("rt_cd") == "0":
                            score = c.get("score", 0)
                            gap = c.get("opening_gap", None)
                            gap_str = f" | 갭: {gap:.2%}" if gap is not None else ""
                            await self.notifier.send_message(
                                f"📈 <b>매수</b> {ticker} ({reason})\n"
                                f"점수: {score:.1f}{gap_str}"
                            )
            except Exception as e:
                logger.error(f"매수 오류 {ticker}: {e}")

    # ------------------------------------------------------------------ #
    # 포지션 청산 체크
    # ------------------------------------------------------------------ #

    async def _check_exit_conditions(self):
        """보유 포지션 청산 조건 체크 및 실행."""
        if not self.strategy.holdings:
            return
        for ticker in list(self.strategy.holdings.keys()):
            try:
                should_exit, reason = await self.strategy.check_exit_condition(ticker)
                if should_exit:
                    logger.info(f"[청산신호] {ticker}: {reason}")
                    await self.strategy.exit(ticker, reason=reason)
                    await self.notifier.send_message(
                        f"📤 <b>청산</b> {ticker}\n사유: {reason}"
                    )
            except Exception as e:
                logger.error(f"청산 조건 체크 오류 {ticker}: {e}")

    # ------------------------------------------------------------------ #
    # 스크리닝 결과 저장/복원
    # ------------------------------------------------------------------ #

    async def _save_screening_results(self):
        """스크리닝 결과를 JSON 파일로 저장합니다."""
        try:
            data = {
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "mode": "DEMO" if self.demo_mode else "REAL",
                "count": len(self.candidate_stocks),
                "candidates": self.candidate_stocks,
            }
            with open(config.SCREENING_RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logger.info(f"스크리닝 결과 저장: {config.SCREENING_RESULTS_FILE}")
        except Exception as e:
            logger.error(f"스크리닝 결과 저장 오류: {e}")

    async def _load_screening_results(self):
        """재시작 시 당일 스크리닝 결과 복원 (issue #7-C)."""
        try:
            if not os.path.exists(config.SCREENING_RESULTS_FILE):
                return
            with open(config.SCREENING_RESULTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            saved_date = data.get("date", "")
            today = datetime.now().strftime("%Y-%m-%d")
            if saved_date == today and data.get("candidates"):
                self.candidate_stocks = data["candidates"]
                self.strategy.set_candidate_stocks(self.candidate_stocks)
                logger.info(
                    f"당일 스크리닝 결과 복원: {len(self.candidate_stocks)}종목 "
                    f"(저장: {data.get('last_updated', '?')})"
                )
        except Exception as e:
            logger.error(f"스크리닝 결과 복원 오류: {e}")

    # ------------------------------------------------------------------ #
    # 장 마감 리포트
    # ------------------------------------------------------------------ #

    async def _generate_closing_report(self):
        """15:30 당일 매매 성과 리포트 생성 및 텔레그램 전송."""
        try:
            history = self.strategy.order_history
            today = datetime.now().strftime("%Y-%m-%d")
            today_trades = [h for h in history if h.get("time", "").startswith(today)]

            buys  = [t for t in today_trades if t["action"] == "BUY"]
            sells = [t for t in today_trades if t["action"] == "SELL"]

            profits = [t.get("profit_ratio", 0) for t in sells]
            avg_profit = sum(profits) / len(profits) if profits else 0
            wins = sum(1 for p in profits if p > 0)
            win_rate = wins / len(profits) if profits else 0

            msg = (
                f"📊 <b>일일 마감 리포트 ({today})</b>\n"
                f"매수: {len(buys)}건 | 매도: {len(sells)}건\n"
                f"승률: {win_rate:.0%} | 평균수익: {avg_profit:.2%}\n"
                f"리스크: {self.risk_manager.risk_status} | 시장: {self.risk_manager.market_condition}"
            )
            await self.notifier.send_message(msg)
            logger.info("마감 리포트 전송 완료")
        except Exception as e:
            logger.error(f"마감 리포트 생성 오류: {e}")

    # ------------------------------------------------------------------ #
    # 하위 호환성 래퍼 (deprecated)
    # ------------------------------------------------------------------ #

    async def _run_screening(self):
        """하위 호환성 유지용 래퍼."""
        await self._premarket_screening()

    async def _morning_entry(self):
        """하위 호환성 유지용 래퍼 (→ _opening_validation_and_entry)."""
        await self._opening_validation_and_entry()
