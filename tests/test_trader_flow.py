"""AsyncAutoTrader — 청산/진입 흐름과 개장일 판정."""
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import config
from core.async_trader import AsyncAutoTrader
from strategy.async_trading_strategy import rejection
from tests.helpers import IsolatedStateTestCase, run
from utils import market_calendar
from utils.costs import net_pnl

ENTRY = datetime(2026, 9, 14, 15, 10)


def holding(**kw):
    base = {"ticker": "005930", "name": "삼성전자", "quantity": 2, "sellable_quantity": 2,
            "buy_price": 70_000, "current_price": 63_000, "high_price": 71_000,
            "entry_time": ENTRY, "reason": "Overnight", "buy_trade_id": 6, "pending_sell": None}
    base.update(kw)
    return base


def make_trader(holdings=None, coordinator=None):
    t = AsyncAutoTrader.__new__(AsyncAutoTrader)
    t.running = True
    t.demo_mode = False
    t.coordinator = coordinator
    t.candidate_stocks = []
    t._exit_fail_counts = {}
    t._exit_retry_at = {}
    t._open_day_query_at = None
    t._trading_halted_on = ""
    t._restart_alerts = {}
    t.notifier = MagicMock(); t.notifier.send_message = AsyncMock()
    t.db = MagicMock()
    t.db.save_trade_sell = AsyncMock(return_value=1)
    t.db.save_trade_buy = AsyncMock(return_value=77)
    t.api_client = MagicMock()
    t.api_client.open_day_calendar = {}
    t.api_client.is_open_day = AsyncMock(return_value=True)
    t.risk_manager = MagicMock()
    t.risk_manager.market_condition = "NORMAL"

    st = MagicMock()
    st.holdings = dict(holdings or {})
    st._unsellable_tickers = set()
    st.adopted_unknown = []
    st.closed_positions = []
    st.opened_positions = []
    st.unfilled_alerts = []
    st.order_history = []
    st.update_holdings = AsyncMock(return_value=True)
    st.open_tickers = lambda: [k for k, v in st.holdings.items()
                               if not v.get("pending_sell") and k not in st._unsellable_tickers]
    st.check_exit_condition = AsyncMock(return_value=(True, "Overnight Hard Stop -10.00%"))
    st.in_rebuy_cooldown = MagicMock(return_value=None)
    st.check_entry_condition = AsyncMock(return_value=True)
    t.strategy = st
    return t, st


def accept_sell(st, price=63_000, qty=2):
    async def _exit(ticker, reason=""):
        st.holdings[ticker]["pending_sell"] = {
            "at": datetime.now().isoformat(), "reason": reason, "quantity": qty, "price": price}
        return {"rt_cd": "0"}
    st.exit = AsyncMock(side_effect=_exit)


def broker_confirms_close(st, ticker):
    """다음 잔고 동기화에서 포지션이 사라진 것이 확인된 상황."""
    st.closed_positions.append(st.holdings.pop(ticker))


class ExitFlowTest(IsolatedStateTestCase):
    def test_holdings_are_synced_even_when_empty(self):
        # 재시작 직후: 로컬 보유 목록이 비어 있어도 잔고를 봐야 손절 감시가 시작된다.
        t, st = make_trader()
        run(t._check_exit_conditions())
        st.update_holdings.assert_awaited_once()
        st.check_exit_condition.assert_not_awaited()

    def test_accepted_sell_records_nothing_until_broker_confirms(self):
        t, st = make_trader({"005930": holding()})
        accept_sell(st)
        run(t._check_exit_conditions())
        t.risk_manager.record_trade_pnl.assert_not_called()
        t.db.save_trade_sell.assert_not_awaited()
        self.assertIn("청산 주문 접수", t.notifier.send_message.await_args.args[0])

    def test_confirmed_close_is_recorded_once_with_pnl_and_buy_link(self):
        t, st = make_trader({"005930": holding()})
        accept_sell(st)
        run(t._check_exit_conditions())
        broker_confirms_close(st, "005930")
        run(t._check_exit_conditions())
        run(t._check_exit_conditions())

        amount, ratio = net_pnl(70_000, 63_000, 2)
        self.assertLess(amount, -14_000)   # 수수료·거래세만큼 더 손실
        t.risk_manager.record_trade_pnl.assert_called_once_with(amount)
        t.db.save_trade_sell.assert_awaited_once()
        kw = t.db.save_trade_sell.await_args.kwargs
        self.assertEqual((kw["buy_trade_id"], kw["strategy"], kw["quantity"]), (6, "Overnight", 2))
        self.assertAlmostEqual(kw["pnl_ratio"], ratio)
        self.assertIn("Hard Stop", kw["reason"])
        self.assertEqual(st.order_history[-1]["pnl_amount"], amount)

    def test_released_order_then_resell_is_recorded_once(self):
        # 접수된 주문이 체결 없이 풀렸다가 다시 매도되는 경우 손익이 두 번 잡히면 안 된다.
        t, st = make_trader({"005930": holding()})
        accept_sell(st)
        run(t._check_exit_conditions())
        st.holdings["005930"]["pending_sell"] = None      # 동기화가 주문이 풀린 것을 확인
        run(t._check_exit_conditions())                   # 재매도 접수
        broker_confirms_close(st, "005930")
        run(t._check_exit_conditions())
        self.assertEqual(st.exit.await_count, 2)
        t.risk_manager.record_trade_pnl.assert_called_once()
        t.db.save_trade_sell.assert_awaited_once()

    def test_pending_sell_is_not_signalled_again(self):
        t, st = make_trader({"005930": holding(pending_sell={"at": "x"})})
        st.exit = AsyncMock()
        run(t._check_exit_conditions())
        st.check_exit_condition.assert_not_awaited()
        st.exit.assert_not_awaited()

    def test_unknown_adopted_position_is_announced(self):
        t, st = make_trader({"000660": holding(ticker="000660", name="SK하이닉스", reason="Standard")})
        st.adopted_unknown = ["000660"]
        st.check_exit_condition = AsyncMock(return_value=(False, "Hold"))
        run(t._check_exit_conditions())
        self.assertIn("출처 불명", t.notifier.send_message.await_args.args[0])
        self.assertEqual(st.adopted_unknown, [])

    def test_rejected_sell_keeps_retrying_with_growing_interval(self):
        t, st = make_trader({"005930": holding()})
        st.exit = AsyncMock(return_value={"rt_cd": "1", "msg_cd": "APBK0400", "msg1": "수량 초과"})
        run(t._check_exit_conditions())
        self.assertEqual(st.exit.await_count, 1)
        run(t._check_exit_conditions())                   # 재시도 시각 전 → 건너뜀
        self.assertEqual(st.exit.await_count, 1)

        for expected in (2, 3, 4):
            t._exit_retry_at["005930"] = datetime.now() - timedelta(seconds=1)
            run(t._check_exit_conditions())
            self.assertEqual(st.exit.await_count, expected)
        self.assertIn("005930", st.holdings)
        self.assertNotIn("005930", st._unsellable_tickers)   # 손절을 포기하지 않는다
        delay = (t._exit_retry_at["005930"] - datetime.now()).total_seconds()
        self.assertTrue(70 < delay <= 80, delay)              # 10 → 20 → 40 → 80초
        alerts = [c.args[0] for c in t.notifier.send_message.await_args_list]
        self.assertEqual(sum("청산 실패 3회" in a for a in alerts), 1)

    def test_transport_failure_is_not_counted_as_rejection(self):
        # 서킷브레이커가 열린 60초 동안의 실패는 주문이 KIS 에 닿지도 않은 것이다.
        t, st = make_trader({"005930": holding()})
        st.exit = AsyncMock(return_value={"rt_cd": "-1", "msg1": "Circuit breaker open",
                                          "_transport_failure": True})
        for _ in range(6):
            run(t._check_exit_conditions())
        self.assertEqual(st.exit.await_count, 6)
        self.assertEqual(t._exit_fail_counts, {})
        self.assertEqual(st.open_tickers(), ["005930"])

        accept_sell(st)                                   # 서킷이 닫히면 바로 나간다
        run(t._check_exit_conditions())
        self.assertIsNotNone(st.holdings["005930"]["pending_sell"])

    def test_trading_halt_reported_by_broker_is_announced(self):
        t, st = make_trader({"005930": holding()})
        st.exit = AsyncMock(return_value={"rt_cd": "1", "msg_cd": "APBK0066", "msg1": "거래정지",
                                          "_unsellable": True})
        run(t._check_exit_conditions())
        self.assertIn("매도 불가", t.notifier.send_message.await_args.args[0])
        self.assertEqual(t._exit_fail_counts, {})

    def test_unconfirmed_sell_raises_alert(self):
        t, st = make_trader({"005930": holding()})
        st.exit = AsyncMock(return_value={"rt_cd": "-1", "_unconfirmed": True,
                                          "_transport_failure": True})
        run(t._check_exit_conditions())
        self.assertIn("접수 여부 미확인", t.notifier.send_message.await_args.args[0])
        t.db.save_trade_sell.assert_not_awaited()


class MarketClosedRejectionTest(IsolatedStateTestCase):
    def _reject(self, kis_answer):
        t, st = make_trader({"005930": holding()})
        st.exit = AsyncMock(return_value={"rt_cd": "1", "msg_cd": "APBK0919", "msg1": "장운영일자 상이"})
        t.api_client.is_open_day = AsyncMock(return_value=kis_answer)
        if kis_answer is not None:
            key = datetime.now().strftime("%Y%m%d")
            t.api_client.open_day_calendar = {key: kis_answer}
        run(t._check_exit_conditions())
        return t, st

    def test_confirmed_holiday_halts_the_day(self):
        t, _ = self._reject(False)
        self.assertEqual(t._trading_halted_on, datetime.now().strftime("%Y%m%d"))
        self.assertFalse(run(t._is_open_today(datetime.now())))

    def test_unverifiable_does_not_halt(self):
        # 휴장일조회가 실패했다는 이유만으로 손절 감시를 끄지 않는다.
        t, st = self._reject(None)
        self.assertEqual(t._trading_halted_on, "")
        self.assertEqual(t._exit_fail_counts["005930"], 1)

    def test_query_failure_falls_back_to_morning_confirmation(self):
        market_calendar.register_open_days({datetime.now().strftime("%Y%m%d"): True})
        t, st = self._reject(None)
        self.assertEqual(t._trading_halted_on, "")
        self.assertTrue(run(t._is_open_today(datetime.now().replace(hour=13))))

    def test_confirmed_open_day_keeps_monitoring(self):
        # 개장일에 난 APBK0919 하나로 그날 손절 감시까지 멈추면 안 된다.
        t, st = self._reject(True)
        self.assertEqual(t._trading_halted_on, "")
        self.assertEqual(t._exit_fail_counts["005930"], 1)

    def test_halt_alert_is_sent_once(self):
        t, st = self._reject(False)
        for _ in range(5):
            if run(t._is_open_today(datetime.now())):
                run(t._check_exit_conditions())
        self.assertEqual(st.exit.await_count, 1)
        self.assertEqual(sum("당일 매매 중단" in c.args[0]
                             for c in t.notifier.send_message.await_args_list), 1)

    def test_halt_is_lifted_the_next_day(self):
        t, st = self._reject(False)
        tomorrow = datetime.now() + timedelta(days=1)
        market_calendar.register_open_days({tomorrow.strftime("%Y%m%d"): True})
        self.assertTrue(run(t._is_open_today(tomorrow.replace(hour=9))))


class OpenDayTest(IsolatedStateTestCase):
    MON = datetime(2026, 9, 28, 7, 0)

    def test_kis_answer_is_registered_and_wins_over_local_calendar(self):
        t, _ = make_trader()
        t.api_client.is_open_day = AsyncMock(return_value=False)
        t.api_client.open_day_calendar = {"20260928": False}
        self.assertFalse(run(t._is_open_today(self.MON)))
        self.assertTrue(market_calendar.is_confirmed(self.MON))
        self.assertIn("불일치", t.notifier.send_message.await_args.args[0])

    def test_confirmed_day_is_not_queried_again(self):
        t, _ = make_trader()
        t.api_client.open_day_calendar = {"20260928": True}
        for _ in range(3):
            self.assertTrue(run(t._is_open_today(self.MON)))
        self.assertEqual(t.api_client.is_open_day.await_count, 1)

    def test_failed_query_is_retried_after_five_minutes(self):
        t, _ = make_trader()
        t.api_client.is_open_day = AsyncMock(return_value=None)
        self.assertTrue(run(t._is_open_today(self.MON)))                       # 로컬 달력 폴백
        run(t._is_open_today(self.MON + timedelta(minutes=1)))
        self.assertEqual(t.api_client.is_open_day.await_count, 1)
        run(t._is_open_today(self.MON + timedelta(minutes=6)))
        self.assertEqual(t.api_client.is_open_day.await_count, 2)

    def test_no_query_before_six(self):
        t, _ = make_trader()
        run(t._is_open_today(datetime(2026, 9, 28, 3, 0)))
        t.api_client.is_open_day.assert_not_awaited()

    def test_regime_comes_from_risk_manager(self):
        t, _ = make_trader()
        t.risk_manager.market_condition = "VOLATILE_DOWN"
        self.assertEqual(t._get_current_market_regime(), "VOLATILE_DOWN")


class EntryFlowTest(IsolatedStateTestCase):
    CANDS = [{"ticker": "000001", "name": "가", "reason": "Overnight", "score": 80},
             {"ticker": "000002", "name": "나", "reason": "Overnight", "score": 70}]

    def _enter(self, t, st, results):
        async def _entry(ticker, reason="", name="", score=0.0):
            r = results[ticker]
            if r.get("rt_cd") == "0":
                st.holdings[ticker] = holding(ticker=ticker, name=name, quantity=3, buy_price=10_000)
            return r
        st.entry = AsyncMock(side_effect=_entry)
        st.set_buy_trade_id = MagicMock()
        run(t._execute_entries(list(self.CANDS), context="오버나이트 진입"))

    def test_risk_rejection_is_reported_without_error_log(self):
        t, st = make_trader()
        with self.assertNoLogs("auto_trade.auto_trader", level="ERROR"):
            self._enter(t, st, {"000001": {"rt_cd": "0"}, "000002": rejection("주가가 예산 초과")})
        summary = t.notifier.send_message.await_args.args[0]
        self.assertIn("리스크 한도 1종목", summary)
        self.assertIn("주가가 예산 초과", summary)

    def test_buy_is_recorded_only_after_broker_shows_the_position(self):
        # 상한가 종목의 시장가 매수는 접수(rt_cd=0) 후 거부된다. 접수만으로 기록하면
        # 존재하지 않는 매수가 DB 와 알림에 남는다 (실거래 5건).
        t, st = make_trader()
        self._enter(t, st, {"000001": {"rt_cd": "0"}, "000002": rejection("x")})
        t.db.save_trade_buy.assert_not_awaited()
        self.assertFalse(any("체결 확인" in c.args[0] for c in t.notifier.send_message.await_args_list))

        st.opened_positions = [holding(ticker="000001", name="가", quantity=3, buy_price=10_050.0,
                                       reason="Overnight", score=80.0)]
        st.check_exit_condition = AsyncMock(return_value=(False, "Hold"))
        run(t._check_exit_conditions())
        kw = t.db.save_trade_buy.await_args.kwargs
        self.assertEqual((kw["ticker"], kw["price"], kw["quantity"], kw["score"]), ("000001", 10_050.0, 3, 80.0))
        st.set_buy_trade_id.assert_called_once_with("000001", 77)
        self.assertTrue(any("매수 체결 확인" in c.args[0] for c in t.notifier.send_message.await_args_list))

    def test_unfilled_buy_is_announced(self):
        t, st = make_trader()
        st.unfilled_alerts = [holding(ticker="030530", name="원익홀딩스")]
        run(t._check_exit_conditions())
        self.assertIn("매수 미체결", t.notifier.send_message.await_args.args[0])
        self.assertEqual(st.unfilled_alerts, [])

    def test_unconfirmed_buy_raises_alert(self):
        t, st = make_trader()
        self._enter(t, st, {"000001": {"rt_cd": "-1", "_unconfirmed": True},
                            "000002": rejection("x")})
        self.assertTrue(any("접수 여부 미확인" in c.args[0]
                            for c in t.notifier.send_message.await_args_list))
        t.db.save_trade_buy.assert_not_awaited()

    def test_agent_rejecting_everything_blocks_everything(self):
        # 과거: 빈 결정 목록은 "의견 없음"으로 취급돼 전원 매수됐다.
        coord = MagicMock()
        coord.generate_buy_decisions = AsyncMock(return_value=[])
        t, st = make_trader(coordinator=coord)
        self._enter(t, st, {})
        st.entry.assert_not_awaited()

    def test_agent_crash_falls_back_to_no_filter(self):
        coord = MagicMock()
        coord.generate_buy_decisions = AsyncMock(side_effect=ValueError("boom"))
        t, st = make_trader(coordinator=coord)
        self._enter(t, st, {"000001": {"rt_cd": "0"}, "000002": {"rt_cd": "0"}})
        self.assertEqual(st.entry.await_count, 2)

    def test_intraday_paths_are_skipped_when_disabled(self):
        t, st = make_trader()
        t.screener = MagicMock(); t.screener.run_screening_async = AsyncMock()
        t.candidate_stocks = list(self.CANDS)
        with patch.object(config, "INTRADAY_ENTRY_ENABLED", False):
            run(t._opening_validation_and_entry())
            run(t._intraday_screening_and_entry())
            run(t._continuous_signal_check())
        t.screener.run_screening_async.assert_not_awaited()
        st.update_holdings.assert_not_awaited()


if __name__ == "__main__":
    import unittest
    unittest.main()
