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
from agents.coordinator import AgentCoordinator

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

        # ── 멀티에이전트 시스템 ──────────────────────────────────────────────
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

        # DB (ticker → buy_trade_id 매핑: 매수 저장 ID를 매도 시 연결)
        self.db = TradeDatabase()
        self._buy_trade_ids: Dict[str, Optional[int]] = {}

        # 중복 트리거 방지 — 당일 실행 완료 이벤트 기록
        self._triggered: set = set()
        self._last_trigger_date: str = ""

        # 체제 전환 알림용 — 마지막 보고한 체제 기억
        self._last_reported_regime: str = ""

        # 자동 최적화 — 거래 누적 카운터 + 마지막 최적화 일자
        self._trades_since_last_optimize: int = 0
        self._last_optimize_date: str = ""
        self._AUTO_OPTIMIZE_TRADE_THRESHOLD = 50  # 50거래마다 자동 최적화
        self._AUTO_OPTIMIZE_DAY = 5               # 토요일(weekday=5)에 주간 자동 최적화

    # ------------------------------------------------------------------ #
    # 시장 체제 조회 헬퍼
    # ------------------------------------------------------------------ #

    def _get_current_market_regime(self) -> str:
        """현재 시장 체제 문자열 반환.

        우선순위:
          1. AgentCoordinator의 MarketIntel 컨텍스트 (가장 정확)
          2. RiskManager의 market_condition (fallback)
        """
        try:
            if hasattr(self, "coordinator") and self.coordinator:
                ctx = self.coordinator.get_market_context()
                if ctx and ctx.regime:
                    regime = ctx.regime
                    return regime.value if hasattr(regime, "value") else str(regime)
        except Exception:
            pass
        return self.risk_manager.market_condition  # "BULL" / "BEAR" / "NORMAL"

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
        try:
            await self.coordinator.start()
            logger.info("✅ 멀티에이전트 시스템 시작 완료")
        except Exception as e:
            logger.warning(f"에이전트 시스템 시작 실패 (기존 로직으로 계속): {e}")

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
                    await self.notifier.send_message(f"⚠️ <b>[{name}] 태스크 재시작</b>\n오류: {e}")
                    await asyncio.sleep(5)

        self.tasks = [
            asyncio.create_task(_run_with_restart(self._scheduled_morning_routine, "morning_routine")),
            asyncio.create_task(_run_with_restart(self._monitor_loop, "monitor_loop")),
        ]
        await asyncio.gather(*self.tasks)

    async def stop(self):
        self.running = False
        for task in getattr(self, "tasks", []):
            task.cancel()
        # 에이전트 종료
        try:
            await self.coordinator.stop()
        except Exception as e:
            logger.debug(f"에이전트 종료 오류: {e}")
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
            from utils.utils import is_market_open
            market_is_open_today = is_market_open()

            # 06:00 — 자동 파라미터 최적화 (토요일 or 50거래 누적)
            if now_str == "06:00" and self._should_trigger("06:00_optimize"):
                self._mark_triggered("06:00_optimize")
                await self._run_auto_optimize()

            # 07:00 — 시장 리스크 재평가
            if now_str == "07:00" and self._should_trigger("07:00"):
                self._mark_triggered("07:00")
                if market_is_open_today:
                    logger.info("07:00 시장 리스크 평가 시작...")
                    await self.risk_manager.assess_market_risk()
                    # 에이전트 일별 리셋 (GAP-06: 오버나이트 포지션 holdings 전달)
                    self.coordinator.reset_daily(current_holdings=self.strategy.holdings)
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
            from utils.utils import is_market_open
            
            # 주말 및 공휴일일 때는 아무것도 하지 않고 대기
            if not is_market_open():
                 await asyncio.sleep(60)
                 continue
                 
            now = datetime.now()
            # 09:00~15:30 사이에만 실행
            if 9 <= now.hour < 15 or (now.hour == 15 and now.minute <= 30):
                # 1. 청산 조건 체크
                await self._check_exit_conditions()

                # 1.1 CB Level 3/4 강제 청산 (P3: 누진적 서킷브레이커)
                if hasattr(self, 'coordinator') and self.coordinator:
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
        logger.info("시초가 검증 매수 로직 실행")
        try:
            if not self.candidate_stocks:
                logger.warning("사전 스크리닝 결과 없음 — 즉시 재스크리닝")
                await self._premarket_screening()
                # _premarket_screening이 에러로 실패 처리되었을 수 있음
                if not self.candidate_stocks:
                    return

            await self.strategy.update_holdings()

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
        logger.info("장중 모멘텀 스크리닝 시작")
        try:
            regime = self._get_current_market_regime()
            new_candidates = await self.screener.run_screening_async(
                ["KOSPI", "KOSDAQ"], is_intraday=True, market_regime=regime
            )

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

            await self.strategy.update_holdings()
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

            await self.strategy.update_holdings()
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
        try:
            await self.strategy.update_holdings()
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
                                        is_intraday=True, market_regime=regime
                                    )
                                    if result and result.get("ticker") and result.get("score", 0) > 0:
                                        return result
                                except Exception as e:
                                    logger.debug(f"[신규 포착] {ticker} 스코어링 실패: {e}")
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
                    import time as _time
                    ticker = c.get("ticker", "")
                    # 30분 재매수 쿨다운 체크
                    recently_sold = getattr(self.strategy, "_recently_sold", {})
                    if ticker in recently_sold:
                        elapsed = _time.time() - recently_sold[ticker]
                        if elapsed < 1800:
                            logger.debug(f"[연속 시그널] {ticker}: 매도 후 재매수 쿨다운 ({int(elapsed/60)}분 경과, 30분 필요)")
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
            import time as _time
            ticker = c.get("ticker", "")
            if not ticker:
                return (c, "티커 없음")
            # 30분 재매수 쿨다운 체크
            recently_sold = getattr(self.strategy, "_recently_sold", {})
            if ticker in recently_sold:
                elapsed = _time.time() - recently_sold[ticker]
                if elapsed < 1800:
                    return (c, f"매도 후 쿨다운({int(elapsed/60)}분 경과)")
            # 오버나이트 후보: 스크리닝(close_position≥0.7)에서 이미 검증 완료
            # Strategy A~E(모멘텀/눌림)는 인트라데이 조건 → 오버나이트에 부적합
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

        # ── 에이전트 시스템: 추가 신뢰도 필터 ─────────────────────────────
        agent_decisions = {}
        try:
            decisions = await self.coordinator.generate_buy_decisions(
                approved, self.strategy.holdings
            )
            for d in decisions:
                agent_decisions[d.ticker] = d
            if decisions:
                logger.info(
                    f"[AgentCoordinator] {len(decisions)}개 매수 결정 "
                    f"({[d.ticker for d in decisions]})"
                )
            else:
                # Bug #6 visibility (2026-04-24): 코디네이터가 후보 전체를 거부한 경우 명시
                logger.info(
                    f"[AgentCoordinator] 후보 {len(approved)}개 전원 거부 "
                    f"(alpha_conf/score/sentiment/risk 7단계 필터)"
                )
        except Exception as e:
            # 기존: logger.debug — 코디네이터 크래시가 조용히 숨겨짐.
            # Bug #6 visibility: 경고 레벨 + 스택트레이스로 승격
            logger.warning(
                f"에이전트 결정 생성 오류 (fallback to legacy logic): {e}",
                exc_info=True,
            )

        # 에이전트 거부 종목 집계
        agent_rejected = []
        if agent_decisions:
            for c in approved:
                ticker = c.get("ticker", "")
                if ticker and ticker not in agent_decisions:
                    agent_rejected.append((c.get("name", ticker), ticker, "에이전트 리스크 필터"))
                    # Bug #6 visibility: debug → info (Intraday 사일런트 차단 추적)
                    logger.info(
                        f"[AgentCoordinator] {ticker} 에이전트 거부 "
                        f"(reason={c.get('reason','?')} score={c.get('score',0)})"
                    )

        success_tickers = []   # (name, ticker)
        failed_tickers = []    # (name, ticker, reason)

        for c in approved:
            # 포트폴리오 슬롯 체크: max_stocks 도달 시 중단
            if len(self.strategy.holdings) >= config.MAX_STOCK_COUNT:
                logger.info(f"[매수중단] 최대 보유 종목 수 도달 ({config.MAX_STOCK_COUNT})")
                break

            ticker = c.get("ticker", "")
            reason = c.get("reason", "Momentum")
            name = c.get("name", ticker)
            agent_dec = agent_decisions.get(ticker)

            # 에이전트가 이 종목을 거부했으면 건너뜀 (상세 로그는 위에서 이미 기록됨)
            if agent_decisions and ticker not in agent_decisions:
                continue

            try:
                # G5: 에이전트가 Kelly 기반 수량을 계산했으면 전달
                agent_qty = agent_dec.quantity if agent_dec and agent_dec.quantity > 0 else 0
                result = await self.strategy.entry(ticker, quantity=agent_qty, reason=reason)
                if result and result.get("rt_cd") == "0":
                    score = c.get("score", 0)
                    gap = c.get("opening_gap", None)
                    gap_str = f" | 갭: {gap:+.2%}" if gap is not None else ""
                    # Bug #4 fix (2026-04-24): 실제 체결가는 strategy.holdings 에 저장되어 있음.
                    # 기존에는 screener 후보 dict의 current_price/close 를 썼는데 대부분 0이었음 →
                    # DB BUY 레코드 price 컬럼이 전량 0으로 오염됨.
                    filled_price = self.strategy.holdings.get(ticker, {}).get("buy_price", 0)
                    buy_price = filled_price or c.get("current_price", 0) or c.get("close", 0)
                    filled_qty = (
                        self.strategy.holdings.get(ticker, {}).get("quantity", 0)
                        or agent_qty
                    )
                    invest_amt = int(float(buy_price) * filled_qty) if buy_price and filled_qty else 0
                    invest_str = f"\n투자금: {invest_amt:,.0f}원" if invest_amt > 0 else ""
                    conf_str = f" | 신뢰도: {agent_dec.combined_confidence:.0%}" if agent_dec else ""
                    strategy_str = agent_dec.strategy if agent_dec else reason

                    await self.notifier.send_message(
                        f"📈 <b>매수 체결</b> {name} ({ticker})\n"
                        f"전략: {strategy_str} | 점수: {score:.1f}{conf_str}\n"
                        f"체결가: {float(buy_price):,.0f}원 | 수량: {filled_qty}주{gap_str}"
                        f"{invest_str}"
                    )
                    success_tickers.append((name, ticker))
                    try:
                        self.coordinator.on_trade_executed(
                            ticker=ticker,
                            action="BUY",
                            strategy=strategy_str,
                            price=float(buy_price),
                            quantity=filled_qty,
                        )
                    except Exception:
                        pass

                    # DB 저장 (비동기, 실패해도 매매 계속)
                    trade_id = await self.db.save_trade_buy(
                        ticker=ticker,
                        name=name,
                        price=float(buy_price),
                        quantity=filled_qty,
                        strategy=strategy_str,
                        score=float(score),
                        market_regime=self._get_current_market_regime(),
                        agent_confidence=agent_dec.combined_confidence if agent_dec else 0.0,
                    )
                    self._buy_trade_ids[ticker] = trade_id
                else:
                    r = result or {}
                    err_msg = r.get("msg1") or r.get("msg_cd") or r.get("message") or "응답 없음"
                    err_msg = err_msg[:50]
                    logger.error(f"[매수실패 상세] {ticker}: rt_cd={r.get('rt_cd')} msg_cd={r.get('msg_cd')} msg1={r.get('msg1')}")
                    failed_tickers.append((name, ticker, err_msg))
            except Exception as e:
                logger.error(f"매수 오류 {ticker}: {e}")
                failed_tickers.append((name, ticker, str(e)[:30]))

        # ── 진입 요약 메시지 ──────────────────────────────────────────────
        if context:
            total = len(success_tickers) + len(failed_tickers) + len(agent_rejected) + len(timing_rejected)
            if total == 0:
                return
            msg = f"📊 <b>[{context}] 진입 요약</b>\n{'─' * 22}\n"
            if success_tickers:
                msg += f"✅ <b>매수 성공 {len(success_tickers)}종목</b>\n"
                for sname, sticker in success_tickers:
                    msg += f"  • {sname}({sticker})\n"
            if failed_tickers:
                msg += f"\n❌ <b>주문 실패 {len(failed_tickers)}종목</b>\n"
                for fname, fticker, freason in failed_tickers[:5]:
                    msg += f"  • {fname}({fticker}): {freason}\n"
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
        if not self.strategy.holdings:
            return

        # 청산 전 실제 보유 잔고 동기화 (stale holdings 방지 → "청산 실패" 감소)
        try:
            await self.strategy.update_holdings()
        except Exception:
            pass  # 동기화 실패해도 로컬 holdings로 진행

        # ── 에이전트 청산 신호 수집 ──────────────────────────────────────────
        agent_sell_set: set = set()
        try:
            sell_decisions = await self.coordinator.generate_sell_decisions(self.strategy.holdings)
            for d in sell_decisions:
                if d.ticker:
                    agent_sell_set.add(d.ticker)
                    logger.info(f"[AgentCoordinator] 청산 신호: {d.ticker} ({d.reason})")
        except Exception as e:
            logger.debug(f"에이전트 청산 신호 오류: {e}")

        regime = self._get_current_market_regime()
        for ticker in list(self.strategy.holdings.keys()):
            try:
                # 기존 전략 청산 조건 체크 (체제별 손익비 적용)
                should_exit, reason = await self.strategy.check_exit_condition(ticker, market_regime=regime)

                # 에이전트가 추가 청산 신호를 내렸으면 보강
                if not should_exit and ticker in agent_sell_set:
                    should_exit = True
                    reason = "에이전트 리스크 청산"

                if should_exit:
                    holding = self.strategy.holdings.get(ticker, {})
                    buy_price = holding.get("buy_price", 0)
                    current_price = holding.get("current_price", 0)
                    profit_pct = (current_price / buy_price - 1) if buy_price > 0 and current_price > 0 else 0
                    profit_emoji = "🟢" if profit_pct >= 0 else "🔴"
                    stock_name = holding.get("name", ticker)

                    logger.info(f"[청산신호] {ticker}: {reason}")
                    result = await self.strategy.exit(ticker, reason=reason)
                    if result and result.get("rt_cd") == "0":
                        # 성공 — 수익/손실 금액 표시
                        qty = holding.get("quantity", 0)
                        pnl_amount = (current_price - buy_price) * qty if buy_price else 0
                        strategy_name = holding.get("reason", holding.get("strategy", "Momentum"))
                        buy_time_raw = holding.get("entry_time", "")
                        hold_str = ""
                        if buy_time_raw:
                            try:
                                if isinstance(buy_time_raw, datetime):
                                    bt = buy_time_raw
                                else:
                                    bt = datetime.strptime(str(buy_time_raw)[:19], "%Y-%m-%d %H:%M:%S")
                                hold_mins = int((datetime.now() - bt).total_seconds() / 60)
                                hold_str = (
                                    f" | 보유: {hold_mins // 60}h{hold_mins % 60}m"
                                    if hold_mins >= 60 else f" | 보유: {hold_mins}분"
                                )
                            except Exception:
                                pass
                        await self.notifier.send_message(
                            f"📤 <b>청산 체결</b> {stock_name} ({ticker})\n"
                            f"{profit_emoji} 수익률: <b>{profit_pct:+.2%}</b> | {qty}주{hold_str}\n"
                            f"손익: {pnl_amount:+,.0f}원\n"
                            f"매수가: {buy_price:,.0f}원 → 청산가: {current_price:,.0f}원\n"
                            f"전략: {strategy_name} | 사유: {reason}"
                        )
                        # 실패 카운터 초기화 + 자동 최적화 카운터 증가
                        self._exit_fail_counts.pop(ticker, None)
                        self._increment_trade_counter()
                        # 에이전트에 청산 결과 피드백
                        try:
                            self.coordinator.on_trade_executed(
                                ticker=ticker,
                                action="SELL",
                                strategy=strategy_name,
                                price=float(current_price),
                                quantity=holding.get("quantity", 0),
                                pnl_ratio=profit_pct,
                                pnl_amount=pnl_amount,
                            )
                        except Exception:
                            pass
                        # DB 저장
                        await self.db.save_trade_sell(
                            ticker=ticker,
                            name=stock_name,
                            price=float(current_price),
                            quantity=holding.get("quantity", 0),
                            buy_price=float(buy_price),
                            pnl_amount=float(pnl_amount),
                            pnl_ratio=float(profit_pct),
                            reason=reason,
                            buy_trade_id=self._buy_trade_ids.pop(ticker, None),
                            market_regime=self._get_current_market_regime(),
                            strategy=strategy_name,
                        )
                    elif result is None:
                        # can_trade() 거부 — 시장 시간 외 또는 리스크 차단 (정상)
                        logger.debug(f"[청산거부] {ticker}: 리스크 매니져 또는 시장 시간 외")
                    else:
                        # API 오류 — 연속 실패 추적
                        err_msg = result.get("msg1", "unknown")
                        err_code = result.get("msg_cd", "")
                        logger.error(f"[청산실패] {ticker}: {err_code} {err_msg}")

                        self._exit_fail_counts[ticker] = self._exit_fail_counts.get(ticker, 0) + 1
                        fail_cnt = self._exit_fail_counts[ticker]

                        if fail_cnt >= 3:
                            # 실제로 이미 매도됐거나 잔고 불일치 가능성 → 로컬 제거
                            logger.warning(
                                f"[청산실패] {ticker} {fail_cnt}회 연속 실패 "
                                f"→ 로컬 holdings 강제 제거 (잔고 확인 필요)"
                            )
                            self.strategy.holdings.pop(ticker, None)
                            self._exit_fail_counts.pop(ticker, None)
                            await self.notifier.send_message(
                                f"🚨 <b>청산 이상</b> {stock_name} ({ticker})\n"
                                f"{fail_cnt}회 연속 API 오류 → 잔고 확인 요망\n"
                                f"오류: {err_code} {err_msg}"
                            )
                        # 1~2회 실패: 텔레그램 없이 재시도 대기
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
    # 자동 파라미터 최적화 (데이터 누적 시 자동 실행)
    # ------------------------------------------------------------------ #

    def _increment_trade_counter(self):
        """매매 완료 시 호출. 임계값 도달 시 자동 최적화 예약."""
        self._trades_since_last_optimize += 1
        if self._trades_since_last_optimize >= self._AUTO_OPTIMIZE_TRADE_THRESHOLD:
            logger.info(f"[AutoOptimize] {self._trades_since_last_optimize}건 누적 → 자동 최적화 예약")

    async def _run_auto_optimize(self):
        """OHLCV 캐시 데이터로 베이지안 최적화 실행 후 config.yaml 자동 업데이트."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_optimize_date == today:
            return  # 하루 1회 제한

        # 조건 체크: 50거래 누적 OR 토요일
        is_saturday = datetime.now().weekday() == self._AUTO_OPTIMIZE_DAY
        enough_trades = self._trades_since_last_optimize >= self._AUTO_OPTIMIZE_TRADE_THRESHOLD

        if not (is_saturday or enough_trades):
            return

        logger.info(f"[AutoOptimize] 시작 (사유: {'주간 정기' if is_saturday else f'{self._trades_since_last_optimize}건 누적'})")
        await self.notifier.send_message(
            f"🔧 <b>자동 파라미터 최적화 시작</b>\n"
            f"사유: {'주간 정기 (토요일)' if is_saturday else f'{self._trades_since_last_optimize}건 거래 누적'}"
        )

        try:
            # 1. OHLCV 캐시에서 데이터 로드
            import glob
            import pandas as pd
            cache_dir = os.path.join("data", "backtest_cache")
            ohlcv_data = {}
            for csv_file in glob.glob(os.path.join(cache_dir, "*.csv")):
                ticker = os.path.basename(csv_file).replace(".csv", "")
                try:
                    df = pd.read_csv(csv_file, parse_dates=["date"])
                    if len(df) >= 60:
                        ohlcv_data[ticker] = df
                except Exception:
                    continue

            if len(ohlcv_data) < 5:
                logger.warning(f"[AutoOptimize] 캐시 종목 부족 ({len(ohlcv_data)}개). 최소 5개 필요.")
                return

            # 2. 현재 체제 기반 최적화
            regime = self._get_current_market_regime()
            from backtest.optimizer import RegimeOptimizer
            optimizer = RegimeOptimizer(
                ohlcv_data=ohlcv_data,
                regime=regime if regime != "NORMAL" else None,
                n_trials=50,
            )

            # 3. 동기 최적화를 스레드에서 실행 (이벤트 루프 블록 방지)
            best_params = await asyncio.to_thread(optimizer.optimize)

            if not best_params:
                logger.warning("[AutoOptimize] 최적 파라미터 없음")
                return

            # 4. config.yaml에 저장
            await asyncio.to_thread(optimizer.save_best_params)

            # 5. 카운터 리셋
            self._trades_since_last_optimize = 0
            self._last_optimize_date = today

            # 6. 결과 알림
            result_msg = "\n".join(f"  {k}: {v}" for k, v in best_params.items())
            await self.notifier.send_message(
                f"✅ <b>자동 최적화 완료</b>\n"
                f"체제: {regime}\n"
                f"종목: {len(ohlcv_data)}개\n"
                f"최적 파라미터:\n<code>{result_msg}</code>\n\n"
                f"⚠️ 다음 거래 세션부터 적용됩니다."
            )
            logger.info(f"[AutoOptimize] 완료: {best_params}")

        except ImportError:
            logger.warning("[AutoOptimize] optuna 미설치. pip install optuna 필요.")
        except Exception as e:
            logger.error(f"[AutoOptimize] 실패: {e}", exc_info=True)
            await self.notifier.send_message(f"⚠️ <b>자동 최적화 실패</b>\n오류: {e}")

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
            except Exception as e:
                logger.debug(f"에이전트 리포트 취합 오류: {e}")

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
