"""청산 규칙 — Overnight B 로직, 거래일 기준 보유일, 고정 손절가."""
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from strategy.async_trading_strategy import AsyncTradingStrategy
from tests.helpers import IsolatedStateTestCase, freeze_time, run
from utils import market_calendar

MODULE = "strategy.async_trading_strategy"


def make_strategy():
    api = MagicMock()
    rm = MagicMock()
    rm.calculate_dynamic_stoploss = AsyncMock(return_value=9_800.0)
    return AsyncTradingStrategy(api, rm), api, rm


def hold(s, entry: datetime, buy=10_000, reason="Overnight", **extra):
    s.holdings["T"] = {"ticker": "T", "name": "T", "quantity": 1, "sellable_quantity": 1,
                       "buy_price": buy, "current_price": buy, "high_price": buy,
                       "entry_time": entry, "reason": reason, **extra}


def decide(s, api, now: datetime, price: int, status="REGULAR"):
    api.get_current_price = AsyncMock(return_value={"price": price})
    with freeze_time(MODULE, now), patch(f"{MODULE}.get_trading_time_status", return_value=status):
        return run(s.check_exit_condition("T"))


MON_BUY = datetime(2026, 9, 14, 15, 10)   # 월요일 15:10 매수
FRI_BUY = datetime(2026, 9, 18, 15, 10)   # 금요일 15:10 매수


class OvernightExitTest(IsolatedStateTestCase):
    def test_d0_right_after_buy_is_held(self):
        # 과거 버그: '15:10' >= '09:05' 문자열 비교로 매수 직후 즉시 청산
        s, api, _ = make_strategy(); hold(s, MON_BUY)
        self.assertEqual(decide(s, api, datetime(2026, 9, 14, 15, 10, 30), 10_000),
                         (False, "Hold Overnight (D+0)"))

    def test_d0_hard_stop(self):
        s, api, _ = make_strategy(); hold(s, MON_BUY)
        exit_, reason = decide(s, api, datetime(2026, 9, 14, 15, 25), 9_550)
        self.assertTrue(exit_); self.assertIn("Hard Stop", reason)

    def test_d1_below_take_profit_is_held(self):
        s, api, _ = make_strategy(); hold(s, MON_BUY)
        self.assertFalse(decide(s, api, datetime(2026, 9, 15, 10, 0), 10_300)[0])

    def test_d1_take_profit(self):
        s, api, _ = make_strategy(); hold(s, MON_BUY)
        exit_, reason = decide(s, api, datetime(2026, 9, 15, 13, 0), 10_500)
        self.assertTrue(exit_); self.assertIn("D+1 TP", reason)

    def test_d2_morning_window_forces_exit(self):
        s, api, _ = make_strategy(); hold(s, MON_BUY)
        exit_, reason = decide(s, api, datetime(2026, 9, 16, 9, 10), 10_100)
        self.assertTrue(exit_); self.assertIn("D+2 Morning", reason)

    def test_d2_between_windows_is_held_then_closed_in_afternoon(self):
        s, api, _ = make_strategy(); hold(s, MON_BUY)
        self.assertFalse(decide(s, api, datetime(2026, 9, 16, 9, 45), 10_100)[0])
        exit_, reason = decide(s, api, datetime(2026, 9, 16, 14, 30), 10_100)
        self.assertTrue(exit_); self.assertIn("Close Exit", reason)

    def test_friday_buy_gets_its_d1_window_on_monday(self):
        # 달력일로 세면 월요일이 D+3 이라 09:05 에 바로 강제 청산됐다.
        s, api, _ = make_strategy(); hold(s, FRI_BUY)
        exit_, reason = decide(s, api, datetime(2026, 9, 21, 9, 10), 10_100)
        self.assertFalse(exit_); self.assertIn("D+1", reason)
        exit_, reason = decide(s, api, datetime(2026, 9, 22, 9, 10), 10_100)
        self.assertTrue(exit_); self.assertIn("D+2 Morning", reason)

    def test_holidays_do_not_count_as_holding_days(self):
        market_calendar.register_open_days({"20260924": False, "20260925": False})
        s, api, _ = make_strategy(); hold(s, datetime(2026, 9, 23, 15, 10))
        exit_, reason = decide(s, api, datetime(2026, 9, 28, 9, 10), 10_100)
        self.assertFalse(exit_); self.assertIn("D+1", reason)


class PriceLookupFailureTest(IsolatedStateTestCase):
    def test_stop_still_fires_on_last_synced_price_when_quote_fails(self):
        # 급락 중 시세 조회가 실패해도 손절 판단을 멈추지 않는다.
        s, api, _ = make_strategy(); hold(s, MON_BUY)
        s.holdings["T"]["current_price"] = 9_000   # 마지막 잔고 동기화 값
        api.get_current_price = AsyncMock(return_value=None)
        with freeze_time(MODULE, datetime(2026, 9, 15, 10, 0)), \
                patch(f"{MODULE}.get_trading_time_status", return_value="REGULAR"):
            exit_, reason = run(s.check_exit_condition("T"))
        self.assertTrue(exit_); self.assertIn("Hard Stop", reason)

    def test_pending_sell_is_not_re_evaluated(self):
        s, api, _ = make_strategy(); hold(s, MON_BUY, pending_sell={"at": "2026-09-15T09:05:00"})
        self.assertEqual(decide(s, api, datetime(2026, 9, 15, 10, 0), 9_000), (False, "Sell order pending"))


class StandardExitTest(IsolatedStateTestCase):
    def test_stop_price_fixed_at_entry_is_not_recomputed(self):
        s, api, rm = make_strategy(); hold(s, MON_BUY, reason="Momentum", stop_price=9_850.0)
        self.assertFalse(decide(s, api, datetime(2026, 9, 14, 15, 15), 9_900)[0])
        rm.calculate_dynamic_stoploss.assert_not_awaited()
        exit_, reason = decide(s, api, datetime(2026, 9, 14, 15, 16), 9_840)
        self.assertTrue(exit_); self.assertIn("9850", reason)

    def test_missing_stop_is_computed_once_and_stored(self):
        s, api, rm = make_strategy(); hold(s, MON_BUY, reason="Standard")
        decide(s, api, datetime(2026, 9, 14, 15, 15), 9_950)
        decide(s, api, datetime(2026, 9, 14, 15, 16), 9_950)
        self.assertEqual(rm.calculate_dynamic_stoploss.await_count, 1)
        self.assertEqual(s.holdings["T"]["stop_price"], 9_800.0)

    def test_short_term_positions_close_in_closing_auction(self):
        s, api, _ = make_strategy(); hold(s, MON_BUY.replace(hour=10), reason="Intraday", stop_price=9_000.0)
        exit_, reason = decide(s, api, datetime(2026, 9, 14, 15, 21), 10_050, status="CLOSING_AUCTION")
        self.assertTrue(exit_); self.assertIn("동시호가", reason)

    def test_short_term_take_profit(self):
        s, api, _ = make_strategy(); hold(s, MON_BUY.replace(hour=10), reason="Momentum", stop_price=9_000.0)
        exit_, reason = decide(s, api, datetime(2026, 9, 14, 11, 0), 10_300)
        self.assertTrue(exit_); self.assertIn("목표 수익", reason)


if __name__ == "__main__":
    import unittest
    unittest.main()
