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
        st.order_history.append({"action": "SELL", "ticker": ticker, "price": price, "quantity": qty})
        st.holdings[ticker]["pending_sell"] = {"at": datetime.now().isoformat()}
        return {"rt_cd": "0"}
    st.exit = AsyncMock(side_effect=_exit)


class ExitFlowTest(IsolatedStateTestCase):
    def test_holdings_are_synced_even_when_empty(self):
        # 재시작 직후: 로컬 보유 목록이 비어 있어도 잔고를 봐야 손절 감시가 시작된다.
        t, st = make_trader()
        run(t._check_exit_conditions())
        st.update_holdings.assert_awaited_once()
        st.check_exit_condition.assert_not_awaited()

    def test_accepted_sell_is_recorded_with_pnl_and_buy_link(self):
        t, st = make_trader({"005930": holding()})
        accept_sell(st)
        run(t._check_exit_conditions())
        amount, ratio = net_pnl(70_000, 63_000, 2)
        self.assertLess(amount, -14_000)   # 수수료·거래세만큼 더 손실
        t.risk_manager.record_trade_pnl.assert_called_once_with(amount)
        kw = t.db.save_trade_sell.await_args.kwargs
        self.assertEqual((kw["buy_trade_id"], kw["strategy"], kw["quantity"]), (6, "Overnight", 2))
        self.assertAlmostEqual(kw["pnl_ratio"], ratio)

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

    def test_three_failures_block_retries_but_keep_the_position(self):
        t, st = make_trader({"005930": holding()})
        st.exit = AsyncMock(return_value={"rt_cd": "1", "msg_cd": "APBK0400", "msg1": "수량 초과"})
        for _ in range(3):
            run(t._check_exit_conditions())
        self.assertIn("005930", st.holdings)             # 로컬에서 지우지 않는다
        self.assertIn("005930", st._unsellable_tickers)
        run(t._check_exit_conditions())
        self.assertEqual(st.exit.await_count, 3)
        alerts = [c.args[0] for c in t.notifier.send_message.await_args_list]
        self.assertEqual(sum("청산 실패" in a for a in alerts), 1)

    def test_unconfirmed_sell_raises_alert(self):
        t, st = make_trader({"005930": holding()})
        st.exit = AsyncMock(return_value={"rt_cd": "-1", "_unconfirmed": True})
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

    def test_unverifiable_halts_the_day(self):
        t, _ = self._reject(None)
        self.assertTrue(t._trading_halted_on)

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
        async def _entry(ticker, reason="", name=""):
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
        st.set_buy_trade_id.assert_called_once_with("000001", 77)

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
