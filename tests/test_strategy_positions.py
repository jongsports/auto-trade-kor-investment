"""포지션 상태 무결성 — 영속화, 잔고 대사, 미체결 매도, 주문 직렬화."""
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from risk.async_risk_manager import BuyPlan
from strategy.async_trading_strategy import AsyncTradingStrategy
from tests.helpers import IsolatedStateTestCase, run


def broker_pos(ticker="005930", qty=1, sellable=None, buy=70_000.0, cur=71_000, name="삼성전자"):
    return {"ticker": ticker, "name": name, "quantity": qty,
            "sellable_quantity": qty if sellable is None else sellable,
            "buy_price": buy, "current_price": cur, "eval_profit_loss": int((cur - buy) * qty)}


def account(*positions, equity=1_300_000, cash=1_000_000):
    return {"total_evaluated_amount": equity, "available_amount": cash, "positions": list(positions)}


def make_strategy(acct=None):
    api = MagicMock()
    api.get_account_summary = AsyncMock(return_value=account() if acct is None else acct)
    api.get_current_price = AsyncMock(return_value={"price": 71_000})
    api.market_buy = AsyncMock(return_value={"rt_cd": "0", "output": {"ODNO": "1"}})
    api.market_sell = AsyncMock(return_value={"rt_cd": "0", "output": {"ODNO": "2"}})
    rm = MagicMock()
    rm.can_trade = AsyncMock(return_value=(True, "OK"))
    rm.plan_buy = AsyncMock(return_value=BuyPlan(3))
    rm.calculate_dynamic_stoploss = AsyncMock(return_value=68_600.0)
    return AsyncTradingStrategy(api, rm), api, rm


class PersistenceTest(IsolatedStateTestCase):
    def test_strategy_tag_and_entry_time_survive_restart(self):
        s, api, _ = make_strategy()
        run(s.entry("005930", reason="Overnight", name="삼성전자"))
        entry_time = s.holdings["005930"]["entry_time"]
        s.set_buy_trade_id("005930", 42)

        restarted, _, _ = make_strategy()   # 새 프로세스
        info = restarted.holdings["005930"]
        self.assertEqual(info["reason"], "Overnight")
        self.assertEqual(info["entry_time"], entry_time)
        self.assertEqual(info["buy_trade_id"], 42)

    def test_restart_then_sync_keeps_metadata_and_takes_broker_numbers(self):
        s, _, _ = make_strategy()
        run(s.entry("005930", reason="Overnight"))
        restarted, _, _ = make_strategy(account(broker_pos(qty=3, buy=70_150.0, cur=72_000)))
        self.assertTrue(run(restarted.update_holdings()))
        info = restarted.holdings["005930"]
        self.assertEqual(info["reason"], "Overnight")
        self.assertEqual(info["buy_price"], 70_150.0)   # 추정가 → 실제 평단
        self.assertEqual(info["high_price"], 72_000)

    def test_corrupt_state_entry_is_skipped_not_fatal(self):
        from utils.state_store import save_state
        save_state("positions", {"BAD": {"reason": "Overnight"}})   # entry_time 없음
        s, _, _ = make_strategy()
        self.assertEqual(s.holdings, {})


class ReconcileTest(IsolatedStateTestCase):
    def test_lookup_failure_keeps_local_state(self):
        s, api, _ = make_strategy()
        run(s.entry("005930", reason="Overnight"))
        api.get_account_summary = AsyncMock(return_value={})
        self.assertFalse(run(s.update_holdings()))
        self.assertIn("005930", s.holdings)

    def test_unknown_broker_position_is_adopted_and_flagged(self):
        s, _, _ = make_strategy(account(broker_pos("000660", name="SK하이닉스")))
        run(s.update_holdings())
        self.assertEqual(s.holdings["000660"]["reason"], "Standard")
        self.assertEqual(s.holdings["000660"]["origin"], "unknown")
        self.assertEqual(s.adopted_unknown, ["000660"])

    def test_unknown_position_is_recovered_from_db_when_possible(self):
        s, _, _ = make_strategy(account(broker_pos()))
        bought = datetime(2026, 6, 1, 15, 10, 18)
        s.position_recoverer = AsyncMock(return_value={
            "entry_time": bought, "reason": "Overnight", "buy_trade_id": 6})
        run(s.update_holdings())
        info = s.holdings["005930"]
        self.assertEqual((info["reason"], info["entry_time"], info["buy_trade_id"]),
                         ("Overnight", bought, 6))
        self.assertEqual(s.adopted_unknown, [])

    def test_fresh_buy_not_yet_at_broker_is_kept(self):
        s, _, _ = make_strategy(account())
        run(s.entry("005930", reason="Overnight"))
        run(s.update_holdings())   # 체결이 아직 잔고에 안 잡힘
        self.assertIn("005930", s.holdings)

    def test_position_sold_outside_the_bot_is_dropped(self):
        s, api, _ = make_strategy(account(broker_pos()))
        run(s.update_holdings())
        s.holdings["005930"]["entry_time"] = datetime.now() - timedelta(days=3)
        api.get_account_summary = AsyncMock(return_value=account())
        run(s.update_holdings())
        self.assertNotIn("005930", s.holdings)
        self.assertEqual(s.closed_positions, [])          # 봇이 판 것이 아니므로 손익 기록 없음

    def test_buy_never_seen_at_broker_is_remembered_and_blocks_rebuy(self):
        s, api, _ = make_strategy(account())
        run(s.entry("005930", reason="Overnight"))
        s.holdings["005930"]["entry_time"] = datetime.now() - timedelta(minutes=5)
        run(s.update_holdings())
        self.assertNotIn("005930", s.holdings)
        self.assertTrue(run(s.entry("005930", reason="Overnight"))["_rejected"])
        self.assertEqual(api.market_buy.await_count, 1)

        api.get_account_summary = AsyncMock(return_value=account(broker_pos(qty=3)))
        run(s.update_holdings())                          # 늦게 체결
        self.assertEqual(s.holdings["005930"]["reason"], "Overnight")
        self.assertEqual(s.adopted_unknown, [])

    def test_unfilled_entry_survives_restart_and_clears_next_day(self):
        s, _, _ = make_strategy(account())
        run(s.entry("005930", reason="Overnight"))
        s.holdings["005930"]["entry_time"] = datetime.now() - timedelta(minutes=5)
        run(s.update_holdings())
        restarted, _, _ = make_strategy(account())
        self.assertTrue(run(restarted.entry("005930"))["_rejected"])
        restarted.reset_daily()
        self.assertFalse(run(restarted.entry("005930")).get("_rejected", False))

    def test_numpy_values_do_not_break_persistence(self):
        import numpy as np
        s, _, _ = make_strategy(account(broker_pos(buy=np.float64(70_150.0), cur=np.int64(72_000))))
        self.assertTrue(run(s.update_holdings()))
        restarted, _, _ = make_strategy()
        self.assertEqual(restarted.holdings["005930"]["buy_price"], 70_150.0)


class SellLifecycleTest(IsolatedStateTestCase):
    def _held(self, **pos):
        s, api, rm = make_strategy(account(broker_pos(**pos)))
        run(s.update_holdings())
        return s, api, rm

    def test_accepted_sell_keeps_position_until_broker_confirms(self):
        # 15:20 동시호가 매도: 접수는 됐지만 체결은 15:30. 잔고에는 그대로 있다.
        s, api, _ = self._held()
        run(s.exit("005930", reason="x"))
        self.assertIn("005930", s.holdings)
        self.assertIsNotNone(s.holdings["005930"]["pending_sell"])

        api.get_account_summary = AsyncMock(return_value=account(broker_pos(sellable=0)))
        run(s.update_holdings())
        self.assertIsNone(run(s.exit("005930", reason="again")))
        self.assertEqual(api.market_sell.await_count, 1)   # 재매도 없음
        self.assertEqual(s.open_tickers(), [])

    def test_position_is_closed_once_broker_no_longer_holds_it(self):
        s, api, _ = self._held()
        run(s.exit("005930", reason="Hard Stop"))
        self.assertEqual(s.closed_positions, [])          # 접수만으로는 청산이 아니다
        api.get_account_summary = AsyncMock(return_value=account())
        run(s.update_holdings())
        self.assertNotIn("005930", s.holdings)
        closed = s.closed_positions[0]
        self.assertEqual(closed["pending_sell"]["reason"], "Hard Stop")
        self.assertEqual(closed["pending_sell"]["price"], 71_000)
        self.assertEqual(s.in_rebuy_cooldown("005930"), 0)

    def test_unfilled_order_released_by_broker_allows_retry(self):
        s, api, _ = self._held()
        run(s.exit("005930", reason="x"))
        s.holdings["005930"]["pending_sell"]["at"] = (datetime.now() - timedelta(minutes=2)).isoformat()
        api.get_account_summary = AsyncMock(return_value=account(broker_pos(sellable=1)))
        run(s.update_holdings())
        self.assertIsNone(s.holdings["005930"]["pending_sell"])
        self.assertEqual(s.open_tickers(), ["005930"])

    def test_sell_quantity_is_limited_to_sellable(self):
        # 과거 APBK0400(주문 가능 수량 초과): 로컬 17주, 실제 매도가능 0주
        s, api, _ = self._held(qty=17, sellable=0)
        self.assertIsNone(run(s.exit("005930", reason="x")))
        api.market_sell.assert_not_awaited()

        s, api, _ = self._held(qty=17, sellable=5)
        run(s.exit("005930", reason="x"))
        api.market_sell.assert_awaited_once_with("005930", 5)

    def test_unconfirmed_sell_is_not_resent_immediately(self):
        s, api, _ = self._held()
        api.market_sell = AsyncMock(return_value={"rt_cd": "-1", "_unconfirmed": True})
        run(s.exit("005930", reason="x"))
        self.assertIsNone(run(s.exit("005930", reason="x")))
        self.assertEqual(api.market_sell.await_count, 1)

    def test_halted_ticker_is_blocked_until_daily_reset(self):
        s, api, _ = self._held()
        api.market_sell = AsyncMock(return_value={"rt_cd": "1", "msg_cd": "APBK0066", "_unsellable": True})
        run(s.exit("005930", reason="x"))
        self.assertEqual(s.open_tickers(), [])
        s.reset_daily()
        self.assertEqual(s.open_tickers(), ["005930"])

    def test_pending_sell_survives_restart(self):
        s, api, _ = self._held()
        run(s.exit("005930", reason="x"))
        restarted, api2, _ = make_strategy(account(broker_pos(sellable=0)))
        run(restarted.update_holdings())
        self.assertIsNone(run(restarted.exit("005930", reason="x")))
        api2.market_sell.assert_not_awaited()


class EntryTest(IsolatedStateTestCase):
    def test_risk_rejection_is_distinguishable_from_api_failure(self):
        s, api, rm = make_strategy()
        rm.plan_buy = AsyncMock(return_value=BuyPlan(0, "주가가 예산 초과"))
        res = run(s.entry("005930"))
        self.assertTrue(res["_rejected"])
        self.assertEqual(res["msg1"], "주가가 예산 초과")
        api.market_buy.assert_not_awaited()
        self.assertEqual(s.holdings, {})

    def test_quantity_comes_from_risk_plan(self):
        s, api, _ = make_strategy()
        run(s.entry("005930", reason="Overnight"))
        api.market_buy.assert_awaited_once_with("005930", 3)

    def test_risk_plan_is_told_about_positions_the_bot_already_knows(self):
        s, api, rm = make_strategy()
        run(s.entry("005930", reason="Overnight"))
        run(s.entry("000660", reason="Overnight"))
        local = rm.plan_buy.await_args.args[2]
        self.assertEqual(local, {"005930": 71_000 * 3})

    def test_requested_quantity_is_only_an_upper_bound(self):
        s, api, _ = make_strategy()
        run(s.entry("005930", quantity=100))
        api.market_buy.assert_awaited_once_with("005930", 3)

    def test_stop_price_is_fixed_at_entry_for_non_overnight(self):
        s, _, rm = make_strategy()
        run(s.entry("005930", reason="Momentum"))
        self.assertEqual(s.holdings["005930"]["stop_price"], 68_600.0)
        run(s.entry("000660", reason="Overnight"))
        self.assertIsNone(s.holdings["000660"]["stop_price"])

    def test_unconfirmed_buy_is_tracked_provisionally(self):
        s, api, _ = make_strategy()
        api.market_buy = AsyncMock(return_value={"rt_cd": "-1", "_unconfirmed": True})
        run(s.entry("005930", reason="Overnight"))
        self.assertTrue(s.holdings["005930"]["unconfirmed"])
        self.assertEqual(s.opened_positions, [])

    def test_buy_is_confirmed_once_when_first_seen_at_broker(self):
        s, api, _ = make_strategy(account())
        run(s.entry("005930", reason="Overnight", score=82))
        run(s.update_holdings())
        self.assertEqual(s.opened_positions, [])                  # 아직 잔고에 없음
        api.get_account_summary = AsyncMock(return_value=account(broker_pos(qty=3, buy=71_150.0)))
        run(s.update_holdings())
        run(s.update_holdings())
        self.assertEqual(len(s.opened_positions), 1)
        opened = s.opened_positions[0]
        self.assertEqual((opened["buy_price"], opened["quantity"], opened["score"]), (71_150.0, 3, 82.0))

    def test_adopted_positions_are_not_reported_as_bot_buys(self):
        s, _, _ = make_strategy(account(broker_pos("000660")))
        run(s.update_holdings())
        self.assertEqual(s.opened_positions, [])

    def test_rejected_order_raises_unfilled_alert_once(self):
        s, _, _ = make_strategy(account())
        run(s.entry("005930", reason="Overnight"))
        s.holdings["005930"]["entry_time"] = datetime.now() - timedelta(minutes=5)
        run(s.update_holdings())
        run(s.update_holdings())
        self.assertEqual([a["ticker"] for a in s.unfilled_alerts], ["005930"])
        self.assertEqual(s.opened_positions, [])

    def test_limit_up_price_blocks_market_buy(self):
        s, api, _ = make_strategy()
        api.get_current_price = AsyncMock(return_value={"price": 19_860, "upper_limit": 19_860})
        res = run(s.entry("030530", reason="Overnight"))
        self.assertTrue(res["_rejected"])
        self.assertIn("상한가", res["msg1"])
        api.market_buy.assert_not_awaited()

    def test_concurrent_entries_for_same_ticker_buy_once(self):
        # 스케줄러와 모니터 루프가 같은 분에 같은 종목으로 진입하는 경우
        s, api, _ = make_strategy()

        async def slow_buy(ticker, qty):
            await asyncio.sleep(0.01)
            return {"rt_cd": "0"}
        api.market_buy = AsyncMock(side_effect=slow_buy)

        async def both():
            return await asyncio.gather(s.entry("005930"), s.entry("005930"))
        results = run(both())
        self.assertEqual(api.market_buy.await_count, 1)
        self.assertEqual(sum(1 for r in results if r.get("_rejected")), 1)

    def test_sync_cannot_erase_a_position_being_bought(self):
        s, api, _ = make_strategy(account())

        async def slow_buy(ticker, qty):
            await asyncio.sleep(0.02)
            return {"rt_cd": "0"}
        api.market_buy = AsyncMock(side_effect=slow_buy)

        async def race():
            await asyncio.gather(s.entry("005930", reason="Overnight"), s.update_holdings())
        run(race())
        self.assertEqual(s.holdings["005930"]["reason"], "Overnight")


class DailyResetTest(IsolatedStateTestCase):
    def test_rebuy_cooldown_entries_are_pruned(self):
        s, _, _ = make_strategy()
        now = datetime.now().timestamp()
        s._recently_sold = {"OLD": now - 3600, "NEW": now - 60}
        s.reset_daily()
        self.assertEqual(list(s._recently_sold), ["NEW"])
        self.assertIsNone(s.in_rebuy_cooldown("OLD"))
        self.assertEqual(s.in_rebuy_cooldown("NEW"), 1)


if __name__ == "__main__":
    import unittest
    unittest.main()
