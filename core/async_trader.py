import logging
import asyncio
import json
import os
from datetime import datetime, time
from typing import Dict, Optional

import config
from core.trader_api import AsyncKisAPI
from strategy.async_screener import AsyncStockScreener
from data.async_news_analyzer import AsyncNewsAnalyzer
from data.trade_db import TradeDatabase
from utils.notifier import AsyncTelegramNotifier
from risk.async_risk_manager import AsyncRiskManager as RiskManager
from strategy.async_trading_strategy import AsyncTradingStrategy as TradingStrategy
from utils import market_calendar
from utils.costs import net_pnl

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
        self.strategy.position_recoverer = self._recover_position_from_db

        # ── 멀티에이전트 시스템 (config.AGENTS_ENABLED 일 때만 구성) ──────────
        self.coordinator = None
        if config.AGENTS_ENABLED:
            from agents.coordinator import AgentCoordinator
            self.coordinator = AgentCoordinator(
                api_client=self.api_client,
                config=config,
                risk_manager=self.risk_manager,
                news_analyzer=self.news_analyzer,
                strategy=self.strategy,  # Step 12: Holdings 단일 소스 참조
            )

        self.running = False
        self.candidate_stocks = []
        self._last_heartbeat_time = datetime.now()

        # 청산 실패 연속 카운터 (ticker → 실패 횟수)
        self._exit_fail_counts: Dict[str, int] = {}
        self._open_day_query_at: Optional[datetime] = None   # 마지막 KIS 개장일 조회 시각
        self._trading_halted_on: str = ""                     # 당일 매매 중단 결정 날짜

        self.db = TradeDatabase()

        # 중복 트리거 방지 — 당일 실행 완료 이벤트 기록
        self._triggered: set = set()
        self._last_trigger_date: str = ""

        # 체제 전환 알림용 — 마지막 보고한 체제 기억
        self._last_reported_regime: str = ""

        # 태스크 재시작 알림 중복 억제
        self._restart_alerts: Dict[str, datetime] = {}

    # ------------------------------------------------------------------ #
    # 시장 체제 조회 헬퍼
    # ------------------------------------------------------------------ #

    def _get_current_market_regime(self) -> str:
        """현재 시장 체제 — RiskManager 의 판정이 단일 출처다.

        과거에는 MarketIntel 에이전트의 컨텍스트를 우선했는데, 그쪽은 데이터 부족으로
        한 번도 판정에 성공하지 못해 기본값 NORMAL 이 항상 반환됐다. 같은 기간
        RiskManager 는 BULL/VOLATILE_* 를 판정하고 있었지만 쓰이지 않았다.
        """
        return self.risk_manager.market_condition

    async def _recover_position_from_db(self, ticker: str) -> Optional[dict]:
        """잔고에 있는 종목의 매수 메타데이터를 DB 에서 복원."""
        row = await self.db.get_open_buy(ticker)
        if not row:
            return None
        return {
            "entry_time": row["executed_at"].replace(tzinfo=None),
            "reason": row.get("strategy") or "Standard",
            "buy_trade_id": row["id"],
            "name": row.get("name") or ticker,
        }

    async def _sync_holdings(self) -> bool:
        """잔고 대사 후, 출처 불명으로 편입된 종목이 있으면 알린다."""
        try:
            ok = await self.strategy.update_holdings()
        except Exception as e:
            logger.error(f"잔고 동기화 오류: {e}")
            return False
        adopted, self.strategy.adopted_unknown = self.strategy.adopted_unknown, []
        for ticker in adopted:
            info = self.strategy.holdings.get(ticker, {})
            await self.notifier.send_message(
                f"⚠️ <b>출처 불명 포지션 편입</b> {info.get('name', ticker)} ({ticker})\n"
                f"잔고에 있으나 봇의 매수 기록이 없습니다. Standard 청산 규칙을 적용합니다.\n"
                f"직접 매수한 종목이라면 봇이 청산할 수 있으니 확인하세요."
            )
        return ok

    # ------------------------------------------------------------------ #
    # 라이프사이클
    # ------------------------------------------------------------------ #

    async def start(self):
        logger.info("비동기 자동 매매 엔진을 시작합니다.")
        self.running = True

        self.api_client.connect()
        await self.api_client.init_session()

        # ── DB 연결 (실패해도 계속 진행) ─────────────────────────────────────
        await self.db.connect()

        # 계좌 정보 조회 후 상세 시작 메시지 전송
        account_info = ""
        try:
            account = await self.api_client.get_account_summary()
            total_eval = int(account.get("total_evaluated_amount", 0))
            available = int(account.get("available_amount", 0))
            positions = account.get("positions", [])
            account_info = (
                f"\n\n💰 총 평가액: {total_eval:,}원\n"
                f"💵 가용 예수금: {available:,}원\n"
                f"📦 기존 보유: {len(positions)}종목"
            )
        except Exception:
            pass

        await self.notifier.send_message(
            f"🚀 <b>자동 매매 엔진 시작</b>\n"
            f"{'─' * 22}\n"
            f"모드: {'🔵 모의투자' if self.demo_mode else '🔴 실전투자'}\n"
            f"시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            f"{account_info}"
        )

        # ── 에이전트 시스템 시작 ──────────────────────────────────────────────
        if self.coordinator:
            try:
                await self.coordinator.start()
                logger.info("✅ 멀티에이전트 시스템 시작 완료")
            except Exception as e:
                logger.warning(f"에이전트 시스템 시작 실패 (기존 로직으로 계속): {e}")
        else:
            logger.info("멀티에이전트 레이어 비활성 (config.AGENTS_ENABLED=False)")

        # 0. 보유 포지션 대사 — 재시작 직후에도 손절 감시가 바로 돌도록 가장 먼저 한다.
        if not await self._sync_holdings():
            logger.error("기동 시 잔고 동기화 실패 — 모니터 루프에서 재시도")
        logger.info(f"보유 포지션: {[(t, i.get('reason')) for t, i in self.strategy.holdings.items()]}")

        # 1. 초기 시장 리스크 평가 (issue #6-C: assess_market_risk 미호출 수정)
        await self.risk_manager.assess_market_risk()

        # 2. 재시작 시 당일 스크리닝 결과 복원 (issue #7-C)
        await self._load_screening_results()

        async def _run_with_restart(coro_fn, name: str):
            while self.running:
                try:
                    await coro_fn()
                except asyncio.CancelledError:
                    logger.info(f"[{name}] 취소됨")
                    break
                except Exception as e:
                    logger.error(f"[{name}] 예외 발생, 5초 후 재시작: {e}", exc_info=True)
                    # 같은 예외가 반복되면 5초마다 알림이 나가므로 10분에 한 번만 보낸다.
                    last = self._restart_alerts.get(name)
                    if last is None or (datetime.now() - last).total_seconds() > 600:
                        self._restart_alerts[name] = datetime.now()
                        await self.notifier.send_message(
                            f"⚠️ <b>[{name}] 태스크 재시작</b>\n오류: {e}")
                    await asyncio.sleep(5)

        self.tasks = [
            asyncio.create_task(_run_with_restart(self._scheduled_morning_routine, "morning_routine")),
            asyncio.create_task(_run_with_restart(self._monitor_loop, "monitor_loop")),
        ]
        await asyncio.gather(*self.tasks)

    async def stop(self):
        self.running = False
        # 진행 중인 주문이 있으면 끝날 때까지 기다린다 (중간에 끊으면 접수 여부를 모른다).
        try:
            await asyncio.wait_for(self.strategy._order_lock.acquire(), timeout=20)
            self.strategy._order_lock.release()
        except asyncio.TimeoutError:
            logger.error("종료 대기 중 주문이 20초 내에 끝나지 않음 — 강제 종료")
        for task in getattr(self, "tasks", []):
            task.cancel()
        if self.coordinator:
            try:
                await self.coordinator.stop()
            except Exception as e:
                logger.warning(f"에이전트 종료 오류: {e}")
        await self.api_client.close()
        await self.db.close()
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

    async def _is_open_today(self, now: datetime) -> bool:
        """오늘이 매매하는 날인지. 스케줄러와 모니터 루프가 같은 판정을 쓴다.

        KIS 휴장일조회로 확인된 날짜는 달력(market_calendar)에 등록되어 단일 출처가 된다.
        미확인이면 06시 이후 조회하고, 실패하면 5분 뒤 다시 시도한다. 그동안은 로컬
        달력으로 판단한다.
        """
        date_str = now.strftime("%Y%m%d")
        if self._trading_halted_on == date_str:
            return False

        if not market_calendar.is_confirmed(now) and now.hour >= 6:
            due = (self._open_day_query_at is None
                   or (now - self._open_day_query_at).total_seconds() >= 300)
            if due:
                self._open_day_query_at = now
                await self._refresh_open_days(now)
        return market_calendar.is_trading_day(now)

    async def _refresh_open_days(self, now: datetime) -> Optional[bool]:
        """KIS 에서 개장일을 받아 달력에 등록. 오늘의 개장 여부를 반환 (실패 시 None)."""
        date_str = now.strftime("%Y%m%d")
        local_guess = market_calendar.is_trading_day(now)
        try:
            kis_open = await self.api_client.is_open_day(date_str)
        except Exception as e:
            logger.warning(f"[개장일검증] 조회 예외: {e}")
            return None
        if kis_open is None:
            return None

        market_calendar.register_open_days(self.api_client.open_day_calendar)
        if kis_open != local_guess:
            logger.warning(f"[개장일검증] 불일치 {date_str}: 로컬={local_guess} KIS={kis_open} → KIS 기준 적용")
            await self.notifier.send_message(
                f"⚠️ <b>개장일 판정 불일치</b> {date_str}\n"
                f"로컬 달력: {'개장' if local_guess else '휴장'} / "
                f"KIS: {'개장' if kis_open else '휴장'}\n"
                f"KIS 기준으로 진행합니다."
            )
        return kis_open

    async def _scheduled_morning_routine(self):
        """다단계 동적 스크리닝 스케줄러.

        시간    단계                설명
        -----   ----------------    -------------------------------------------
        07:00   시장 리스크 평가    KOSPI 변동성·추세 분석
        08:00   사전 스크리닝       전날 종가 기반 후보 선정 (매수 보류)
        09:05   시초가 검증 매수    갭 필터 통과 후 진입
        10:30   장중 모멘텀 탐색    당일 거래량·모멘텀 신규 발굴 & 추가 매수
        12:00   점심 추가 탐색      10:30-13:30 사각지대 제거 (Issue #24-M2)
        13:30   오후 모멘텀 탐색    지속 모멘텀 종목 추가 편입
        14:50   오버나이트 사전     15:10 전 후보 미리 확보 (Issue #24-M1)
        15:10   오버나이트 진입     overnight 보너스 스크리닝 후 진입 (issue #6-A)
        15:30   장 마감 리포트      당일 성과 텔레그램 전송
        """
        while self.running:
            now = datetime.now()
            now_str = now.strftime("%H:%M")

            # 휴장일(주말/공휴일) 판단
            # 단, 토큰 갱신(07:30)은 주말에도 수행할 수 있도록 조건 분리
            market_is_open_today = await self._is_open_today(now)

            # 07:00 — 시장 리스크 재평가
            if now_str == "07:00" and self._should_trigger("07:00"):
                self._mark_triggered("07:00")
                if market_is_open_today:
                    logger.info("07:00 시장 리스크 평가 시작...")
                    await self.risk_manager.assess_market_risk()
                    if self.coordinator:
                        # 에이전트 일별 리셋 (GAP-06: 오버나이트 포지션 holdings 전달)
                        self.coordinator.reset_daily(current_holdings=self.strategy.holdings)
                    self.strategy.reset_daily()
                    self._exit_fail_counts.clear()
                else:
                    logger.info("휴장일이므로 07:00 시장 리스크 평가를 건너뜁니다.")

            # 07:30 — API 토큰 선제적 갱신 (Auto Token Renewal) - 주말에도 실행
            elif now_str == "07:30" and self._should_trigger("07:30"):
                self._mark_triggered("07:30")
                logger.info("07:30 API 토큰 선제적 갱신 시작...")
                success = await asyncio.to_thread(self.api_client._sync_init)
                if success:
                    logger.info("API 토큰 갱신 완료")
                else:
                    logger.error("API 토큰 갱신 실패")
                    await self.notifier.send_message(
                        "🚨 <b>API 토큰 갱신 실패</b>\n07:30 선제 갱신이 실패했습니다. "
                        "기존 토큰이 만료되면 주문·시세 조회가 멈춥니다."
                    )
                    
            # 휴장일이면 이 시간대 이후의 주식 매매 관련 스케줄은 검사하지 않음
            if not market_is_open_today:
                 await asyncio.sleep(10)
                 continue

            # 08:00 — 사전 스크리닝 (매수 보류)
            elif now_str == "08:00" and self._should_trigger("08:00"):
                self._mark_triggered("08:00")
                logger.info("08:00 사전 스크리닝 시작...")
                await self._premarket_screening()

            # 08:30 — 사전 스크리닝 재시도 (Issue #14: 08:00 TPS 실패 대비)
            elif now_str == "08:30" and self._should_trigger("08:30"):
                self._mark_triggered("08:30")
                if not self.candidate_stocks:
                    logger.info("08:30 사전 스크리닝 재시도 (08:00 결과 없음)...")
                    await self._premarket_screening()
                else:
                    logger.info(f"08:30 재시도 불필요 — 기존 후보 {len(self.candidate_stocks)}종목 유지")

            # 08:50 — 사전 스크리닝 최종 재시도 (Issue #14)
            elif now_str == "08:50" and self._should_trigger("08:50"):
                self._mark_triggered("08:50")
                if not self.candidate_stocks:
                    logger.info("08:50 사전 스크리닝 최종 재시도...")
                    await self._premarket_screening()
                else:
                    logger.info(f"08:50 재시도 불필요 — 기존 후보 {len(self.candidate_stocks)}종목 유지")

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

            # 11:20, 13:00 — 시장 리스크 재평가 (R2)
            elif now_str in ("11:20", "13:00") and self._should_trigger(f"market_risk_{now_str}"):
                self._mark_triggered(f"market_risk_{now_str}")
                logger.info(f"{now_str} 시장 리스크 재평가...")
                await self.risk_manager.assess_market_risk()
                logger.info(f"리스크 갱신: {self.risk_manager.risk_status} / {self.risk_manager.market_condition}")

            # 12:00 — 점심 시간 추가 스크리닝 (Issue #24-M2: 10:30-13:30 사각지대 제거)
            elif now_str == "12:00" and self._should_trigger("12:00"):
                self._mark_triggered("12:00")
                logger.info("12:00 점심 시간 추가 스크리닝 시작...")
                await self._intraday_screening_and_entry()

            # 14:50 — 오버나이트 사전 스크리닝 (Issue #24-M1: 15:10 전 후보 확보)
            elif now_str == "14:50" and self._should_trigger("14:50"):
                self._mark_triggered("14:50")
                logger.info("14:50 오버나이트 사전 스크리닝 시작...")
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
        """보유 포지션 청산 조건 + 연속 시그널 모니터링 (장 중 매 10초)."""
        while self.running:
            now = datetime.now()
            # 휴장일이면 대기. 스케줄러와 같은 판정을 쓰므로 당일 매매 중단이 결정되면
            # 이 루프도 함께 멈춘다.
            if not await self._is_open_today(now):
                 await asyncio.sleep(60)
                 continue

            # 09:00~15:30 사이에만 실행
            if 9 <= now.hour < 15 or (now.hour == 15 and now.minute <= 30):
                # 1. 청산 조건 체크
                await self._check_exit_conditions()

                # 1.1 CB Level 3/4 강제 청산 (P3: 누진적 서킷브레이커)
                if self.coordinator:
                    if self.coordinator.risk.should_close_all():
                        for t in list(self.strategy.holdings.keys()):
                            logger.critical(f"🚨 [CB4] 전 포지션 청산: {t}")
                            await self.strategy.exit(t, reason="CB_LEVEL_4_CLOSE_ALL")
                    elif self.coordinator.risk.should_force_close_losers():
                        for t, info in list(self.strategy.holdings.items()):
                            try:
                                price_info = await self.api_client.get_current_price(t)
                                cur = price_info.get("price", 0) if price_info else 0
                            except Exception:
                                cur = 0
                            if cur > 0 and cur < info.get("buy_price", 0):
                                logger.critical(f"🚨 [CB3] 손실 포지션 강제 청산: {t}")
                                await self.strategy.exit(t, reason="CB_LEVEL_3_FORCE_CLOSE")

                # 1.5. 하트비트 보고 (매 STATUS_REPORT_INTERVAL_MINUTES 분마다)
                minutes_since_last = (now - self._last_heartbeat_time).total_seconds() / 60
                if minutes_since_last >= config.STATUS_REPORT_INTERVAL_MINUTES:
                     await self._send_heartbeat()
                     self._last_heartbeat_time = now

                # 2. 연속 시그널 모니터링 (매 5분마다 진입 타점 재확인)
                if now.minute % 5 == 0 and now.second < 10:
                    if self.candidate_stocks and len(self.strategy.holdings) < self.strategy.max_stocks:
                        await self._continuous_signal_check()

                # 3. 뉴스 긴급도 모니터링 (30분마다)
                if now.minute in (0, 30) and now.second < 10:
                    logger.info("Background news monitoring...")
                    try:
                        if self.candidate_stocks and self.screener.news_analyzer:
                            watch_list = [
                                (c.get("ticker",""), c.get("name", c.get("ticker","")))
                                for c in self.candidate_stocks[:10] if c.get("ticker")
                            ]
                            async def fetch_news(t, n):
                                try:
                                    return t, await asyncio.wait_for(
                                        self.screener.news_analyzer.analyze_stock_news(t, n, days=0.04),
                                        timeout=5.0
                                    )
                                except Exception:
                                    return t, {"score": 0}
                            news_results = await asyncio.gather(*[fetch_news(t, n) for t, n in watch_list])
                            urgent = [{"ticker": t, "news_score": r.get("score",0)}
                                      for t, r in news_results if r.get("score",0) >= 8]
                            if urgent:
                                logger.warning(f"긴급 뉴스 발견: {len(urgent)}건 {urgent}")
                    except Exception as e:
                        logger.error(f"뉴스 모니터링 오류: {e}")

            await asyncio.sleep(10)

    async def _send_heartbeat(self):
        """엔진 하트비트(생존 보고) 전송."""
        try:
            account = await self.api_client.get_account_summary()
            total_eval = int(account.get("total_evaluated_amount", 0))
            available = int(account.get("available_amount", 0))
            positions = account.get("positions", [])

            holdings_lines = []
            total_pnl = 0
            for p in positions[:8]:
                pnl = int(p.get("eval_profit_loss", 0))
                total_pnl += pnl
                pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                name = p.get("name", p.get("ticker", ""))
                qty = p.get("quantity", 0)
                holdings_lines.append(f"  {pnl_emoji} {name}: {qty}주 ({pnl:+,}원)")
            if len(positions) > 8:
                holdings_lines.append(f"  ... 외 {len(positions) - 8}종목")
            holdings_str = "\n".join(holdings_lines) if holdings_lines else "  없음"
            pnl_emoji_total = "🟢" if total_pnl >= 0 else "🔴"

            msg = (
                f"💓 <b>하트비트 — 봇 정상 작동 중</b>\n"
                f"{'─' * 22}\n"
                f"⏰ {datetime.now().strftime('%H:%M')} | "
                f"시장: {self.risk_manager.market_condition} | 리스크: {self.risk_manager.risk_status}\n\n"
                f"💰 총 평가액: {total_eval:,}원\n"
                f"💵 가용 예수금: {available:,}원\n"
                f"{pnl_emoji_total} 평가손익 합계: {total_pnl:+,}원\n\n"
                f"📦 보유 종목 ({len(positions)}개)\n{holdings_str}"
            )
            logger.info("엔진 하트비트 보고 전송 완료")
            await self.notifier.send_message(msg)
        except Exception as e:
            logger.error(f"하트비트 전송 중 오류: {e}")

    # ------------------------------------------------------------------ #
    # 스크리닝 단계별 메서드
    # ------------------------------------------------------------------ #

    async def _premarket_screening(self):
        """08:00 사전 스크리닝: 전날 종가 기반 후보 선정 (매수 보류)."""
        logger.info("사전 스크리닝 시작 — 09:05 갭 검증 후 진입")
        try:
            regime = self._get_current_market_regime()

            # Step 15: 니어미스 후보 로드 — 연속 2일 후보에 -5pt 보너스 적용
            import json as _json, os as _os
            near_miss_bonus_tickers: set = set()
            nm_file = _os.path.join("data", "near_miss_candidates.json")
            try:
                if _os.path.exists(nm_file):
                    from datetime import timedelta
                    nm_data = _json.load(open(nm_file))
                    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                    day_before = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
                    yday_tickers = {e["ticker"] for e in nm_data.get(yesterday, [])}
                    dby_tickers = {e["ticker"] for e in nm_data.get(day_before, [])}
                    near_miss_bonus_tickers = yday_tickers & dby_tickers
                    if near_miss_bonus_tickers:
                        logger.info(f"[니어미스] 연속 2일 후보 {len(near_miss_bonus_tickers)}종목 → 스크리닝 임계값 -5pt 보너스")
            except Exception as _e:
                logger.debug(f"[니어미스] 로드 실패: {_e}")

            self.candidate_stocks = await self.screener.run_screening_async(
                ["KOSPI", "KOSDAQ"], market_regime=regime
            )

            # 니어미스 보너스 후처리 적용
            if near_miss_bonus_tickers:
                for c in self.candidate_stocks:
                    if c.get("ticker") in near_miss_bonus_tickers:
                        c["score"] = c.get("score", 0) + 5  # 연속 니어미스 +5pt 보너스
                        c["near_miss_bonus"] = True
            self.strategy.set_candidate_stocks(self.candidate_stocks)
            logger.info(f"사전 스크리닝 완료: {len(self.candidate_stocks)}종목 후보")
            await self._save_screening_results()
            
            # 후보군 상세 리스트 생성
            if self.candidate_stocks:
                candidate_details = ""
                for i, c in enumerate(self.candidate_stocks[:12], 1):
                    name = c.get("name", "N/A")
                    ticker = c.get("ticker", "N/A")
                    score = c.get("score", c.get("total_score", 0))
                    reason = c.get("reason", "알 수 없음")
                    tech = c.get("tech_score", 0)
                    vol = c.get("volume_score", 0)
                    sup = c.get("supply_score", 0)
                    candidate_details += (
                        f"{i}. <b>{name}</b>({ticker}) {score:.0f}점 [{reason}]\n"
                        f"   기술:{tech:.0f} 거래량:{vol:.0f} 수급:{sup:.0f}\n"
                    )
                if len(self.candidate_stocks) > 12:
                    candidate_details += f"... 외 {len(self.candidate_stocks) - 12}종목\n"
                msg = (
                    f"📋 <b>사전 스크리닝 완료</b> ({datetime.now().strftime('%H:%M')})\n"
                    f"{'─' * 22}\n"
                    f"후보: {len(self.candidate_stocks)}종목 | "
                    f"시장: {self.risk_manager.market_condition} | "
                    f"리스크: {self.risk_manager.risk_status}\n\n"
                    f"🔍 <b>후보군 리스트</b>\n{candidate_details}"
                )
            else:
                msg = (
                    f"📋 <b>사전 스크리닝 완료</b> ({datetime.now().strftime('%H:%M')})\n"
                    f"{'─' * 22}\n"
                    f"후보: 0종목 | 시장: {self.risk_manager.market_condition} | 리스크: {self.risk_manager.risk_status}\n\n"
                    f"⚠️ 현재 시장 조건에 맞는 매매 대상 종목이 없습니다."
                )

            await self.notifier.send_message(msg)
        except Exception as e:
            logger.error(f"사전 스크리닝 중 오류 통지: {e}")
            await self.notifier.send_message(f"🚨 <b>[08:00 사전 스크리닝] 매매 시스템 장애 발생</b>\n사유: {e}\n종목 정보를 분석할 수 없습니다.")


    async def _opening_validation_and_entry(self):
        """09:05 시초가 갭 검증 후 진입."""
        if not config.INTRADAY_ENTRY_ENABLED:
            logger.info("시초가 매수 건너뜀 (intraday_entry_enabled=false)")
            return
        logger.info("시초가 검증 매수 로직 실행")
        try:
            if not self.candidate_stocks:
                logger.warning("사전 스크리닝 결과 없음 — 즉시 재스크리닝")
                await self._premarket_screening()
                # _premarket_screening이 에러로 실패 처리되었을 수 있음
                if not self.candidate_stocks:
                    return

            await self._sync_holdings()

            # 오버나이트 제외, 갭 필터 적용
            daytrade_candidates = [c for c in self.candidate_stocks if c.get("reason") != "Overnight"]
            validated = await self.screener.validate_opening_candidates(daytrade_candidates)
            logger.info(f"갭 필터 후 {len(validated)}종목 진입 대상")

            await self._execute_entries(validated[:config.MAX_STOCKS], context="09:05 시초가 매수")
        except Exception as e:
            logger.error(f"시초가 매수 로직 에러: {e}")
            await self.notifier.send_message(f"🚨 <b>[09:05 시초가 매수] 매매 시스템 장애 발생</b>\n사유: {e}")



    async def _intraday_screening_and_entry(self):
        """10:30/13:30 장중 모멘텀 스크리닝 & 추가 매수."""
        if not config.INTRADAY_ENTRY_ENABLED:
            logger.info("장중 스크리닝 건너뜀 (intraday_entry_enabled=false)")
            return
        logger.info("장중 모멘텀 스크리닝 시작")
        try:
            regime = self._get_current_market_regime()
            new_candidates = await self.screener.run_screening_async(
                ["KOSPI", "KOSDAQ"], is_intraday=True, market_regime=regime
            )

            # 같은 종목은 최신 스크리닝 결과로 교체한다. 예전 점수를 남겨 두면 10:30 의
            # 점수로 14:50 에 매수하게 된다. 이번에 통과하지 못한 종목은 후보에서 뺀다.
            self.candidate_stocks = list(new_candidates)

            self.candidate_stocks.sort(key=lambda x: x.get("score", 0), reverse=True)
            self.strategy.set_candidate_stocks(self.candidate_stocks)
            await self._save_screening_results()
            logger.info(f"장중 스크리닝 완료: 총 {len(self.candidate_stocks)}종목")
            
            # 추가된 후보군 상세 알림
            if new_candidates:
                new_details = ""
                for i, c in enumerate(new_candidates, 1):
                     name = c.get('name', 'N/A')
                     ticker = c.get('ticker', 'N/A')
                     score = c.get('score', 0)
                     reason = c.get('reason', '알 수 없음')
                     new_details += f"{i}. {name}({ticker}) : {score}점 [{reason}]\n"
                
                msg = (
                    f"⏱️ <b>장중 스크리닝 완료</b>\n"
                    f"- 신규 포착: {len(new_candidates)}종목\n\n"
                    f"🔍 <b>신규 후보군 리스트</b>\n{new_details}"
                )
                await self.notifier.send_message(msg)

            await self._sync_holdings()
            held = set(self.strategy.holdings.keys())
            targets = [
                c for c in self.candidate_stocks
                if c.get("reason") != "Overnight" and c["ticker"] not in held
            ]
            await self._execute_entries(targets[:config.MAX_STOCKS], context="장중 모멘텀 매수")
        except Exception as e:
            logger.error(f"장중 모멘텀 스크리닝 에러: {e}")
            await self.notifier.send_message(f"🚨 <b>[장중 모멘텀 매수] 매매 시스템 장애 발생</b>\n사유: {e}")


    async def _overnight_entry(self):
        """15:10 오버나이트 진입."""
        logger.info("오버나이트 진입 로직 실행")
        try:
            regime = self._get_current_market_regime()
            overnight_candidates = await self.screener.run_screening_async(
                ["KOSPI", "KOSDAQ"], market_regime=regime
            )
            overnight_only = [c for c in overnight_candidates if c.get("reason") == "Overnight"]

            if not overnight_only:
                logger.info("오버나이트 후보 없음")
                await self.notifier.send_message("⚠️ <b>[오버나이트 진입]</b>\n시장 조건에 맞는 오버나이트 후보 종목이 없습니다.")
                return

            logger.info(f"오버나이트 후보 {len(overnight_only)}종목")
            
            # 오버나이트 상세 알림 추가
            overnight_details = ""
            for i, c in enumerate(overnight_only, 1):
                 name = c.get('name', 'N/A')
                 ticker = c.get('ticker', 'N/A')
                 score = c.get('score', 0)
                 overnight_details += f"{i}. {name}({ticker}) : {score}점 [오버나이트 조건 부합]\n"
            
            msg = (
                f"🌙 <b>오버나이트 스크리닝 완료</b>\n"
                f"- 후보: {len(overnight_only)}종목\n\n"
                f"🔍 <b>오버나이트 진입 리스트</b>\n{overnight_details}"
            )
            await self.notifier.send_message(msg)

            await self._sync_holdings()
            held = set(self.strategy.holdings.keys())
            targets = [c for c in overnight_only if c["ticker"] not in held]
            await self._execute_entries(targets[:config.MAX_STOCKS], context="오버나이트 진입")
        except Exception as e:
            logger.error(f"오버나이트 진입 중 오류: {e}")
            await self.notifier.send_message(f"🚨 <b>[오버나이트 진입] 매매 시스템 장애 발생</b>\n사유: {e}")


    # ------------------------------------------------------------------ #
    # 연속 시그널 모니터링 (매 5분)
    # ------------------------------------------------------------------ #

    async def _continuous_signal_check(self):
        """기존 후보 종목의 진입 타이밍을 5분마다 재확인하여 즉시 매수.
        Issue #24-C3: 매 15분마다 거래량 급증 종목 신규 탐색 추가."""
        if not config.INTRADAY_ENTRY_ENABLED:
            return
        try:
            await self._sync_holdings()
            held = set(self.strategy.holdings.keys())
            remaining_slots = self.strategy.max_stocks - len(held)
            if remaining_slots <= 0:
                return

            # Issue #24-C3: 매 15분(xx:00, xx:15, xx:30, xx:45) 거래량 급증 종목 추가 탐색
            now_min = datetime.now().minute
            if now_min % 15 == 0:
                try:
                    existing_tickers = {c["ticker"] for c in self.candidate_stocks if c.get("ticker")}
                    surge_pairs = await self.screener.get_volume_surge_stocks()  # [(ticker, market), ...]
                    new_pairs = [(t, m) for t, m in surge_pairs if t not in existing_tickers and t not in held]
                    if new_pairs:
                        logger.info(f"[연속 시그널] 거래량 급증 신규 {len(new_pairs)}종목 발견, 스코어링 중...")
                        regime = self._get_current_market_regime()
                        sem = asyncio.Semaphore(2 if self.api_client.demo_mode else 5)

                        async def _score_surge(ticker, market):
                            async with sem:
                                try:
                                    result = await self.screener._process_ticker(
                                        ticker, market, is_overnight_window=False,
                                        is_intraday=True,
                                    )
                                    if result and result.get("ticker") and result.get("score", 0) > 0:
                                        return result
                                except Exception as e:
                                    logger.warning(f"[신규 포착] {ticker} 스코어링 실패: {e}")
                                return None

                        # 2026-04-24: 기존 [:10] 캡으로 49종 발견해도 10개만 스코어링 → 대부분 기회 상실.
                        # 20개로 상향 (demo 모드에선 TPS 부담 커지지만 실전에선 semaphore=20이라 여유).
                        results = await asyncio.gather(
                            *[_score_surge(t, m) for t, m in new_pairs[:20]]
                        )
                        for result in results:
                            if result:
                                self.candidate_stocks.append(result)
                                logger.info(f"[신규 포착] {result.get('name','?')}({result['ticker']}) — {result['score']}점")
                        self.candidate_stocks.sort(key=lambda x: x.get("score", 0), reverse=True)
                        self.strategy.set_candidate_stocks(self.candidate_stocks)
                except Exception as e:
                    logger.warning(f"[연속 시그널] 거래량 급증 탐색 오류: {e}")

            # 이미 보유 중인 종목 제외, 점수 높은 순서로 최대 10개만 체크
            targets = [
                c for c in self.candidate_stocks
                if c.get("ticker") and c["ticker"] not in held
            ][:10]

            if not targets:
                return

            now_str = datetime.now().strftime("%H:%M")
            logger.info(f"[연속 시그널] {now_str} — {len(targets)}종목 진입 조건 재확인 중...")

            regime = self._get_current_market_regime()

            # 체제 전환 알림 (최초 설정 시에는 알림 없음)
            if self._last_reported_regime and regime != self._last_reported_regime:
                now_full = datetime.now().strftime("%Y-%m-%d %H:%M")
                regime_msg = (
                    f"🔄 시장 체제 전환\n"
                    f"{self._last_reported_regime} → {regime}\n"
                    f"시각: {now_full}"
                )
                await self.notifier.send_message(regime_msg)
                logger.info(f"[체제전환] {self._last_reported_regime} → {regime}")
            self._last_reported_regime = regime

            # TPS 안전을 위해 동시 5개씩 배치 처리
            approved = []
            for i in range(0, len(targets), 5):
                batch = targets[i:i+5]
                async def check_one(c, _regime=regime):
                    ticker = c.get("ticker", "")
                    if self.strategy.in_rebuy_cooldown(ticker) is not None:
                        return None
                    try:
                        ohlcv = await self.api_client.get_ohlcv(ticker, count=30)
                        if ohlcv is None or ohlcv.empty:
                            return None
                        can_enter = await self.strategy.check_entry_condition(ticker, ohlcv, market_regime=_regime)
                        return c if can_enter else None
                    except Exception as e:
                        logger.debug(f"[연속 시그널] {ticker} 체크 오류: {e}")
                        return None

                results = await asyncio.gather(*[check_one(c) for c in batch])
                approved.extend([r for r in results if r])

                if len(approved) >= remaining_slots:
                    approved = approved[:remaining_slots]
                    break

            if approved:
                logger.info(f"[연속 시그널] {now_str} — {len(approved)}종목 진입 조건 충족!")
                await self._execute_entries(approved[:remaining_slots], context=f"연속 시그널 {now_str}")
            else:
                logger.info(f"[연속 시그널] {now_str} — 현재 진입 타점 도달 종목 없음")
        except Exception as e:
            logger.error(f"[연속 시그널] 오류: {e}")

    # ------------------------------------------------------------------ #
    # 공통 매수 실행 로직
    # ------------------------------------------------------------------ #

    async def _execute_entries(self, candidates: list, context: str = ""):
        """후보 종목 매수 — 조건 체크 병렬, 주문 순차 (중복·리스크 체크 정합성)."""
        if not candidates:
            if context:
                await self.notifier.send_message(
                    f"⚠️ <b>[{context}] 진입 실패</b>\n- 조건에 부합하는 매매 대상 종목이 없습니다."
                )
            return

        regime = self._get_current_market_regime()

        async def check_one(c: dict, _regime=regime):
            ticker = c.get("ticker", "")
            if not ticker:
                return (c, "티커 없음")
            cooldown_min = self.strategy.in_rebuy_cooldown(ticker)
            if cooldown_min is not None:
                return (c, f"매도 후 쿨다운({cooldown_min}분 경과)")
            # 오버나이트 후보는 스크리닝의 종가 위치·수급 조건으로 이미 검증됨.
            if c.get("reason") == "Overnight":
                return (c, None)
            try:
                ohlcv = c.get("_ohlcv_snapshot")
                if ohlcv is None or (hasattr(ohlcv, "empty") and ohlcv.empty):
                    ohlcv = await self.api_client.get_ohlcv(ticker, count=30)
                if ohlcv.empty:
                    return (c, "데이터 없음")
                can_enter = await self.strategy.check_entry_condition(ticker, ohlcv, market_regime=_regime)
                return (c, None) if can_enter else (c, "타점 미도달")
            except Exception as e:
                logger.error(f"진입 조건 체크 오류 {ticker}: {e}")
                return (c, "조건 체크 오류")

        results = await asyncio.gather(*[check_one(c) for c in candidates])
        approved = []
        timing_rejected = []  # (name, ticker, reason)
        for c, rej_reason in results:
            if rej_reason is None:
                approved.append(c)
            else:
                timing_rejected.append((c.get("name", c.get("ticker", "")), c.get("ticker", ""), rej_reason))

        if not approved:
            if context:
                msg = (
                    f"⚠️ <b>[{context}] 진입 보류</b>\n"
                    f"후보 {len(candidates)}종목 모두 현재 진입 타점 미도달\n"
                )
                if timing_rejected:
                    msg += "\n📋 <b>종목별 사유</b>\n"
                    for name, ticker, reason in timing_rejected[:10]:
                        msg += f"  • {name}({ticker}): {reason}\n"
                await self.notifier.send_message(msg)
            return

        # ── 에이전트 필터 (활성화된 경우만) ─────────────────────────────────
        # 결과는 세 가지다: None = 의견 없음(오류 포함, 필터 미적용),
        # 빈 dict = 전원 거부, 그 외 = 승인 종목. 예전에는 빈 결과를 "의견 없음"으로
        # 취급해 서킷브레이커가 전원 거부하면 오히려 전원 매수됐다.
        agent_decisions: Optional[dict] = None
        if self.coordinator:
            try:
                decisions = await self.coordinator.generate_buy_decisions(
                    approved, self.strategy.holdings
                )
                agent_decisions = {d.ticker: d for d in decisions}
                logger.info(f"[AgentCoordinator] 후보 {len(approved)}개 중 "
                            f"{len(agent_decisions)}개 승인 {list(agent_decisions)}")
            except Exception as e:
                logger.warning(f"에이전트 결정 생성 오류 — 필터 없이 진행: {e}", exc_info=True)

        success_tickers = []   # (name, ticker)
        failed_tickers = []    # (name, ticker, reason)
        rejected_tickers = []  # (name, ticker, reason) — 리스크 한도 등 내부 사유
        agent_rejected = []

        for c in approved:
            ticker = c.get("ticker", "")
            reason = c.get("reason", "Momentum")
            name = c.get("name", ticker)

            if agent_decisions is not None and ticker not in agent_decisions:
                agent_rejected.append((name, ticker, "에이전트 필터"))
                continue

            try:
                result = await self.strategy.entry(ticker, reason=reason, name=name)
                if result.get("rt_cd") == "0":
                    holding = self.strategy.holdings.get(ticker, {})
                    score = c.get("score", 0)
                    gap = c.get("opening_gap", None)
                    gap_str = f" | 갭: {gap:+.2%}" if gap is not None else ""
                    buy_price = holding.get("buy_price", 0)
                    filled_qty = holding.get("quantity", 0)
                    invest_amt = int(float(buy_price) * filled_qty)

                    await self.notifier.send_message(
                        f"📈 <b>매수 주문 접수</b> {name} ({ticker})\n"
                        f"전략: {reason} | 점수: {score:.1f}\n"
                        f"기준가: {float(buy_price):,.0f}원 | 수량: {filled_qty}주{gap_str}\n"
                        f"투자금: {invest_amt:,.0f}원"
                    )
                    success_tickers.append((name, ticker))
                    if self.coordinator:
                        try:
                            self.coordinator.on_trade_executed(
                                ticker=ticker, action="BUY", strategy=reason,
                                price=float(buy_price), quantity=filled_qty,
                            )
                        except Exception as e:
                            logger.warning(f"에이전트 매수 피드백 오류 {ticker}: {e}")

                    trade_id = await self.db.save_trade_buy(
                        ticker=ticker,
                        name=name,
                        price=float(buy_price),
                        quantity=filled_qty,
                        strategy=reason,
                        score=float(score),
                        market_regime=regime,
                    )
                    self.strategy.set_buy_trade_id(ticker, trade_id)
                elif result.get("_rejected"):
                    rejected_tickers.append((name, ticker, result.get("msg1", "")))
                elif result.get("_unconfirmed"):
                    failed_tickers.append((name, ticker, "응답 미수신 — 접수 여부 미확인"))
                    await self.notifier.send_message(
                        f"🚨 <b>매수 주문 접수 여부 미확인</b> {name} ({ticker})\n"
                        f"주문 전송 후 응답을 받지 못했습니다. 재전송하지 않았습니다.\n"
                        f"체결됐다면 다음 잔고 동기화에서 포지션으로 잡힙니다."
                    )
                else:
                    err_msg = (result.get("msg1") or result.get("msg_cd") or "응답 없음")[:50]
                    logger.error(f"[매수실패 상세] {ticker}: rt_cd={result.get('rt_cd')} "
                                 f"msg_cd={result.get('msg_cd')} msg1={result.get('msg1')}")
                    failed_tickers.append((name, ticker, err_msg))
            except Exception as e:
                logger.error(f"매수 오류 {ticker}: {e}", exc_info=True)
                failed_tickers.append((name, ticker, str(e)[:30]))

        # ── 진입 요약 메시지 ──────────────────────────────────────────────
        if context:
            total = (len(success_tickers) + len(failed_tickers) + len(rejected_tickers)
                     + len(agent_rejected) + len(timing_rejected))
            if total == 0:
                return
            msg = f"📊 <b>[{context}] 진입 요약</b>\n{'─' * 22}\n"
            if success_tickers:
                msg += f"✅ <b>매수 접수 {len(success_tickers)}종목</b>\n"
                for sname, sticker in success_tickers:
                    msg += f"  • {sname}({sticker})\n"
            if failed_tickers:
                msg += f"\n❌ <b>주문 실패 {len(failed_tickers)}종목</b>\n"
                for fname, fticker, freason in failed_tickers[:5]:
                    msg += f"  • {fname}({fticker}): {freason}\n"
            if rejected_tickers:
                msg += f"\n🛡️ <b>리스크 한도 {len(rejected_tickers)}종목</b>\n"
                for rname, rticker, rreason in rejected_tickers[:6]:
                    msg += f"  • {rname}({rticker}): {rreason}\n"
            if agent_rejected:
                msg += f"\n🤖 <b>에이전트 필터 {len(agent_rejected)}종목</b>\n"
                for aname, aticker, _ in agent_rejected[:5]:
                    msg += f"  • {aname}({aticker})\n"
            if timing_rejected:
                msg += f"\n⏱️ <b>타점 미도달 {len(timing_rejected)}종목</b>\n"
                for tname, tticker, _ in timing_rejected[:6]:
                    msg += f"  • {tname}({tticker})\n"
            await self.notifier.send_message(msg)

    # ------------------------------------------------------------------ #
    # 포지션 청산 체크
    # ------------------------------------------------------------------ #

    async def _check_exit_conditions(self):
        """보유 포지션 청산 조건 체크 및 실행."""
        # 보유 목록이 비어 있어도 동기화는 한다. 재시작 직후나 외부 체결로 생긴 포지션을
        # 여기서 잡지 못하면 손절 감시 없이 방치된다.
        await self._sync_holdings()

        tickers = self.strategy.open_tickers()
        if not tickers:
            return

        # ── 에이전트 청산 신호 (활성화된 경우만) ─────────────────────────────
        agent_sell_set: set = set()
        if self.coordinator:
            try:
                open_holdings = {t: self.strategy.holdings[t] for t in tickers}
                for d in await self.coordinator.generate_sell_decisions(open_holdings):
                    if d.ticker:
                        agent_sell_set.add(d.ticker)
                        logger.info(f"[AgentCoordinator] 청산 신호: {d.ticker} ({d.reason})")
            except Exception as e:
                logger.warning(f"에이전트 청산 신호 오류: {e}")

        regime = self._get_current_market_regime()
        for ticker in tickers:
            try:
                should_exit, reason = await self.strategy.check_exit_condition(ticker, market_regime=regime)
                if not should_exit and ticker in agent_sell_set:
                    should_exit, reason = True, "에이전트 리스크 청산"
                if not should_exit:
                    continue

                holding = dict(self.strategy.holdings.get(ticker, {}))
                logger.info(f"[청산신호] {ticker}: {reason}")
                result = await self.strategy.exit(ticker, reason=reason)

                if result is None:
                    continue  # 매도 불가 시간·미체결 주문 대기 등 — 다음 틱에 다시 판단
                if result.get("rt_cd") == "0":
                    self._exit_fail_counts.pop(ticker, None)
                    await self._on_sell_accepted(ticker, holding, reason, regime)
                elif result.get("_unconfirmed"):
                    await self.notifier.send_message(
                        f"🚨 <b>매도 주문 접수 여부 미확인</b> {holding.get('name', ticker)} ({ticker})\n"
                        f"주문 전송 후 응답을 받지 못했습니다. 재전송하지 않았습니다.\n"
                        f"잔고에 매도가능수량이 남아 있으면 30초 뒤 다시 시도합니다."
                    )
                else:
                    halted = await self._on_sell_failed(ticker, holding, result)
                    if halted:
                        return
            except Exception as e:
                logger.error(f"청산 조건 체크 오류 {ticker}: {e}", exc_info=True)

    async def _on_sell_accepted(self, ticker: str, holding: dict, reason: str, regime: str):
        """매도 주문 접수 후 기록·알림. 가격은 접수 시점 시세 기준의 추정치다."""
        buy_price = holding.get("buy_price", 0)
        sold = next((h for h in reversed(self.strategy.order_history)
                     if h["action"] == "SELL" and h["ticker"] == ticker), {})
        sell_price = sold.get("price") or holding.get("current_price", 0)
        qty = sold.get("quantity") or holding.get("quantity", 0)
        # 수수료·거래세를 뺀 값. 승패 판정과 일일 손실 한도가 이 값을 쓴다.
        pnl_amount, profit_pct = net_pnl(buy_price, sell_price, qty)
        stock_name = holding.get("name", ticker)
        strategy_name = holding.get("reason", "Standard")

        hold_mins = int((datetime.now() - holding["entry_time"]).total_seconds() / 60) \
            if holding.get("entry_time") else 0
        hold_str = f"{hold_mins // 60}h{hold_mins % 60}m" if hold_mins >= 60 else f"{hold_mins}분"

        # 일일 손실 한도가 이 값을 읽는다.
        self.risk_manager.record_trade_pnl(pnl_amount)

        await self.notifier.send_message(
            f"📤 <b>청산 주문 접수</b> {stock_name} ({ticker})\n"
            f"{'🟢' if profit_pct >= 0 else '🔴'} 수익률: <b>{profit_pct:+.2%}</b> | {qty}주 | 보유: {hold_str}\n"
            f"손익(비용 차감, 추정): {pnl_amount:+,.0f}원\n"
            f"매수가: {buy_price:,.0f}원 → 기준가: {sell_price:,.0f}원\n"
            f"전략: {strategy_name} | 사유: {reason}"
        )
        if self.coordinator:
            try:
                self.coordinator.on_trade_executed(
                    ticker=ticker, action="SELL", strategy=strategy_name,
                    price=float(sell_price), quantity=qty,
                    pnl_ratio=profit_pct, pnl_amount=pnl_amount,
                )
            except Exception as e:
                logger.warning(f"에이전트 청산 피드백 오류 {ticker}: {e}")

        await self.db.save_trade_sell(
            ticker=ticker,
            name=stock_name,
            price=float(sell_price),
            quantity=qty,
            buy_price=float(buy_price),
            pnl_amount=float(pnl_amount),
            pnl_ratio=float(profit_pct),
            reason=reason,
            buy_trade_id=holding.get("buy_trade_id"),
            market_regime=regime,
            strategy=strategy_name,
        )

    async def _on_sell_failed(self, ticker: str, holding: dict, result: dict) -> bool:
        """매도 실패 처리. 당일 매매를 중단했으면 True."""
        err_msg = result.get("msg1", "unknown")
        err_code = result.get("msg_cd", "")
        stock_name = holding.get("name", ticker)
        logger.error(f"[청산실패] {ticker}: {err_code} {err_msg}")

        # APBK0919 = KIS 가 본 장운영일자가 주문일과 다르다 → 휴장일 의심.
        # 주문 오류만으로 달력을 덮지 않고 휴장일조회로 확인한 뒤에만 중단한다.
        # 개장일로 확인되면 일시적 오류로 보고 일반 실패로 처리한다 (손절 감시 유지).
        if err_code == "APBK0919":
            now = datetime.now()
            kis_open = await self._refresh_open_days(now)
            if kis_open is not True:
                self._trading_halted_on = now.strftime("%Y%m%d")
                confirmed = "휴장일로 확인됨" if kis_open is False else "휴장일조회 실패로 확인 불가"
                logger.error(f"[개장일오류] APBK0919 수신, {confirmed} → 당일 매매 중단")
                await self.notifier.send_message(
                    f"🚨 <b>당일 매매 중단</b>\n"
                    f"{stock_name}({ticker}) 주문이 APBK0919(장운영일자 상이)로 거부됐고 "
                    f"{confirmed}."
                )
                return True

        fail_cnt = self._exit_fail_counts.get(ticker, 0) + 1
        self._exit_fail_counts[ticker] = fail_cnt
        if fail_cnt >= 3:
            # 포지션을 로컬에서 지우지 않는다. 지워도 다음 동기화가 잔고에서 다시 넣고,
            # 그때 메타데이터(전략·매수 시각)만 잃는다. 당일 재시도만 막고 알린다.
            self.strategy._unsellable_tickers.add(ticker)
            self._exit_fail_counts.pop(ticker, None)
            logger.warning(f"[청산실패] {ticker} {fail_cnt}회 연속 실패 → 당일 매도 재시도 중단")
            await self.notifier.send_message(
                f"🚨 <b>청산 실패</b> {stock_name} ({ticker})\n"
                f"{fail_cnt}회 연속 실패로 오늘은 재시도하지 않습니다. 포지션은 그대로 남아 있습니다.\n"
                f"오류: {err_code} {err_msg}\n"
                f"직접 확인이 필요합니다. 내일 장 시작 전 차단이 풀립니다."
            )
        return False

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
                "candidates": [
                    {k: v for k, v in c.items() if k != "_ohlcv_snapshot"}
                    for c in self.candidate_stocks
                ],
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
                content = f.read().strip()
            if not content:
                return
            data = json.loads(content)
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
            gross_pnl = sum(t.get("pnl_amount", 0) for t in sells)
            avg_profit = sum(profits) / len(profits) if profits else 0
            wins = sum(1 for p in profits if p > 0)
            losses = len(profits) - wins
            win_rate = wins / len(profits) if profits else 0

            # 베스트/워스트 거래
            best_str = worst_str = ""
            if sells:
                best = max(sells, key=lambda t: t.get("profit_ratio", 0))
                worst = min(sells, key=lambda t: t.get("profit_ratio", 0))
                best_str = f"\n🥇 최고: {best.get('name', best.get('ticker',''))} {best.get('profit_ratio', 0):+.2%}"
                worst_str = f"\n🥉 최저: {worst.get('name', worst.get('ticker',''))} {worst.get('profit_ratio', 0):+.2%}"

            # 전략별 성과
            strategy_stats: dict = {}
            for t in sells:
                strat = t.get("strategy", t.get("reason", "Unknown"))
                if strat not in strategy_stats:
                    strategy_stats[strat] = {"count": 0, "wins": 0, "pnl": 0}
                strategy_stats[strat]["count"] += 1
                if t.get("profit_ratio", 0) > 0:
                    strategy_stats[strat]["wins"] += 1
                strategy_stats[strat]["pnl"] += t.get("pnl_amount", 0)
            strategy_str = ""
            if strategy_stats:
                strategy_str = "\n\n📈 <b>전략별 성과</b>\n"
                for strat, stat in sorted(strategy_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
                    wr = stat["wins"] / stat["count"] if stat["count"] > 0 else 0
                    strategy_str += f"  • {strat}: {stat['count']}건 승률{wr:.0%} {stat['pnl']:+,.0f}원\n"

            # 오늘 청산 내역
            sells_list_str = ""
            if sells:
                sells_list_str = "\n\n📋 <b>오늘 청산 내역</b>\n"
                for t in sells[:8]:
                    tname = t.get("name", t.get("ticker", ""))
                    pct = t.get("profit_ratio", 0)
                    pnl = t.get("pnl_amount", 0)
                    emoji = "🟢" if pct >= 0 else "🔴"
                    sells_list_str += f"  {emoji} {tname}: {pct:+.2%} ({pnl:+,.0f}원)\n"
                if len(sells) > 8:
                    sells_list_str += f"  ... 외 {len(sells) - 8}건\n"

            # 에이전트 일별 성과 취합
            agent_report = ""
            try:
                if not self.coordinator:
                    raise LookupError("agents disabled")
                daily = self.coordinator.daily_report()
                alpha_stats = daily.get("alpha_strategies", {})
                best_strategy = max(
                    alpha_stats.items(),
                    key=lambda x: x[1].get("win_rate", 0),
                    default=("N/A", {}),
                )
                ctx = self.coordinator.get_market_context()
                agent_report = (
                    f"\n\n🤖 <b>에이전트 분석</b>\n"
                    f"시장체제: {ctx.regime} | 시장폭: {ctx.breadth_score:.0f}\n"
                    f"최고전략: {best_strategy[0]} (승률 {best_strategy[1].get('win_rate', 0):.0%})\n"
                    f"리스크: {daily.get('risk_status', {}).get('heat_level', 'N/A')}"
                )
            except LookupError:
                pass
            except Exception as e:
                logger.warning(f"에이전트 리포트 취합 오류: {e}")

            pnl_emoji = "🟢" if gross_pnl >= 0 else "🔴"
            msg = (
                f"📊 <b>일일 마감 리포트 ({today})</b>\n"
                f"{'─' * 22}\n"
                f"매수: {len(buys)}건 | 매도: {len(sells)}건\n"
                f"승률: {win_rate:.0%} ({wins}승 {losses}패) | 평균: {avg_profit:+.2%}\n"
                f"{pnl_emoji} 총 손익: <b>{gross_pnl:+,.0f}원</b>"
                f"{best_str}{worst_str}\n"
                f"리스크: {self.risk_manager.risk_status} | 시장: {self.risk_manager.market_condition}"
                f"{strategy_str}{sells_list_str}{agent_report}"
            )
            await self.notifier.send_message(msg)
            logger.info("마감 리포트 전송 완료")

            # DB 일별 요약 저장
            try:
                await self.db.save_daily_summary(
                    total_trades=len(today_trades),
                    buy_trades=len(buys),
                    sell_trades=len(sells),
                    win_trades=wins,
                    loss_trades=len(sells) - wins,
                    gross_pnl=float(gross_pnl),
                    market_regime=self._get_current_market_regime(),
                    risk_status=getattr(self.risk_manager, "risk_status", ""),
                    screened_count=len(self.candidate_stocks),
                )
            except Exception as db_err:
                logger.debug(f"DB 일별 요약 저장 오류 (무시): {db_err}")
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
