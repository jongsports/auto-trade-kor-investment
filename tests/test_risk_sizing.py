import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

import config
from risk.async_risk_manager import AsyncRiskManager, plan_buy_quantity
from tests.helpers import IsolatedStateTestCase, freeze_time, run
from datetime import datetime

LIMITS = dict(max_investment_ratio=0.2, max_stock_count=3)


def plan(**kw):
    base = dict(price=10_000, target_amount=130_000, equity=1_300_000, cash=1_000_000,
                invested=0, position_count=0, **LIMITS)
    base.update(kw)
    return plan_buy_quantity(**base)


class PlanBuyQuantityTest(unittest.TestCase):
    def test_buys_floor_of_budget(self):
        self.assertEqual(plan().quantity, 13)

    def test_price_above_budget_means_no_order(self):
        # 과거에는 max(1, ...) 로 삼성전자 1주(계좌의 27%)를 샀다.
        p = plan(price=352_500)
        self.assertEqual(p.quantity, 0)
        self.assertIn("종목당 예산", p.reason)

    def test_total_investment_cap_shrinks_order(self):
        # 한도 26만원 중 20만원 사용 → 남은 6만원만큼만
        self.assertEqual(plan(invested=200_000).quantity, 6)

    def test_over_cap_blocks_even_one_share(self):
        p = plan(invested=285_500)
        self.assertEqual(p.quantity, 0)
        self.assertIn("투자 비율 한도 도달", p.reason)

    def test_cash_reserves_market_order_margin(self):
        # 시장가 매수는 상한가(+30%) 기준 증거금 → 현금 5만원으로는 3주(39,000원)까지
        self.assertEqual(plan(cash=50_000).quantity, 3)

    def test_insufficient_cash(self):
        p = plan(cash=12_000)
        self.assertEqual(p.quantity, 0)
        self.assertIn("현금", p.reason)

    def test_position_count_cap(self):
        self.assertEqual(plan(position_count=3).quantity, 0)

    def test_unknown_equity_blocks(self):
        self.assertEqual(plan(equity=0).quantity, 0)

    def test_order_never_exceeds_any_limit(self):
        for price in (1_000, 9_999, 39_000, 130_000, 131_000, 900_000):
            for invested in (0, 100_000, 259_000, 300_000):
                p = plan(price=price, invested=invested)
                amount = p.quantity * price
                self.assertLessEqual(amount, 130_000)
                if p.quantity:
                    self.assertLessEqual(invested + amount, 1_300_000 * 0.2 + 1e-6)


def make_rm(account):
    api = MagicMock()
    api.get_account_summary = AsyncMock(return_value=account)
    closes = pd.Series([100 + (i % 5) for i in range(20)], dtype=float)
    api.get_ohlcv = AsyncMock(return_value=pd.DataFrame({
        "close": closes, "high": closes + 1, "low": closes - 1}))
    rm = AsyncRiskManager(api)
    return rm, api


class PlanBuyTest(IsolatedStateTestCase):
    def test_account_lookup_failure_blocks_buy(self):
        rm, _ = make_rm({})
        p = run(rm.plan_buy("005930", 10_000))
        self.assertEqual(p.quantity, 0)
        self.assertIn("잔고 조회 실패", p.reason)

    def test_missing_volatility_data_blocks_buy(self):
        rm, api = make_rm({"total_evaluated_amount": 1_300_000, "available_amount": 1_000_000, "positions": []})
        api.get_ohlcv = AsyncMock(return_value=pd.DataFrame())
        self.assertEqual(run(rm.plan_buy("005930", 10_000)).quantity, 0)

    def test_sizing_uses_equity_and_existing_positions(self):
        rm, _ = make_rm({"total_evaluated_amount": 1_300_000, "available_amount": 1_000_000,
                         "positions": [{"current_price": 100_000, "quantity": 2}]})
        p = run(rm.plan_buy("000001", 10_000))
        self.assertGreater(p.quantity, 0)
        self.assertLessEqual(200_000 + p.quantity * 10_000, 260_000)


    def test_orders_not_yet_at_broker_consume_limits(self):
        acct = {"total_evaluated_amount": 1_300_000, "available_amount": 1_300_000, "positions": []}
        rm, _ = make_rm(acct)
        free = run(rm.plan_buy("C", 10_000)).quantity
        self.assertGreater(free, 0)
        # 방금 접수한 두 주문(합계 25만원)이 잔고에 아직 없다 → 한도 26만원 중 1만원 남음
        p = run(rm.plan_buy("C", 10_000, {"A": 125_000.0, "B": 125_000.0}))
        self.assertEqual(p.quantity, 1)
        p = run(rm.plan_buy("D", 10_000, {"A": 80_000.0, "B": 80_000.0, "C": 80_000.0}))
        self.assertEqual(p.quantity, 0)                  # 종목 수 한도

    def test_positions_already_at_broker_are_not_double_counted(self):
        acct = {"total_evaluated_amount": 1_300_000, "available_amount": 1_000_000,
                "positions": [{"ticker": "A", "current_price": 100_000, "quantity": 1}]}
        rm, _ = make_rm(acct)
        with_local = run(rm.plan_buy("C", 10_000, {"A": 100_000.0})).quantity
        without = run(rm.plan_buy("C", 10_000)).quantity
        self.assertEqual(with_local, without)

    def test_unreadable_market_data_downgrades_risk_state(self):
        rm, api = make_rm({})
        rm.risk_status, rm.market_condition = "NORMAL", "BULL"
        api.get_ohlcv = AsyncMock(return_value=pd.DataFrame())
        run(rm.assess_market_risk())
        self.assertEqual((rm.risk_status, rm.market_condition), ("CAUTION", "NORMAL"))


class CanTradeTest(IsolatedStateTestCase):
    def _rm(self, account, realized=0.0):
        rm, api = make_rm(account)
        if realized:
            rm.record_trade_pnl(realized)
        return rm

    def test_daily_loss_limit_blocks_buy(self):
        rm = self._rm({"total_evaluated_amount": 1_000_000}, realized=-25_000)
        with patch("risk.async_risk_manager.is_trading_time", return_value=True):
            ok, msg = run(rm.can_trade("X", "buy"))
        self.assertFalse(ok)
        self.assertIn("일일 최대 손실", msg)

    def test_small_loss_does_not_block(self):
        rm = self._rm({"total_evaluated_amount": 1_000_000}, realized=-5_000)
        with patch("risk.async_risk_manager.is_trading_time", return_value=True):
            self.assertTrue(run(rm.can_trade("X", "buy"))[0])

    def test_loss_check_fails_closed_when_account_unknown(self):
        rm = self._rm({}, realized=-5_000)
        with patch("risk.async_risk_manager.is_trading_time", return_value=True):
            self.assertFalse(run(rm.can_trade("X", "buy"))[0])

    def test_daily_pnl_survives_restart_and_resets_next_day(self):
        rm = self._rm({"total_evaluated_amount": 1_000_000}, realized=-25_000)
        rm2, _ = make_rm({"total_evaluated_amount": 1_000_000})
        self.assertEqual(rm2.daily_realized_pnl(), -25_000)
        tomorrow = datetime.now().replace(year=datetime.now().year + 1)
        with freeze_time("risk.async_risk_manager", tomorrow):
            self.assertEqual(rm2.daily_realized_pnl(), 0.0)

    def test_sell_is_allowed_in_closing_auction_only_during_session(self):
        rm = self._rm({})
        with patch("risk.async_risk_manager.get_trading_time_status", return_value="CLOSING_AUCTION"):
            self.assertTrue(run(rm.can_trade("X", "sell"))[0])
        with patch("risk.async_risk_manager.get_trading_time_status", return_value="POST_MARKET"):
            self.assertFalse(run(rm.can_trade("X", "sell"))[0])


class StopLossTest(IsolatedStateTestCase):
    def test_stop_is_never_looser_than_loss_cut_and_never_above_entry(self):
        rm, _ = make_rm({})
        stop = run(rm.calculate_dynamic_stoploss("X", 100_000))
        self.assertGreaterEqual(stop, 100_000 * (1 - config.LOSS_CUT_RATIO) - 1e-6)
        self.assertLessEqual(stop, 100_000)

    def test_missing_data_falls_back_to_loss_cut(self):
        rm, api = make_rm({})
        api.get_ohlcv = AsyncMock(return_value=pd.DataFrame())
        self.assertAlmostEqual(run(rm.calculate_dynamic_stoploss("X", 100_000)),
                               100_000 * (1 - config.LOSS_CUT_RATIO))


if __name__ == "__main__":
    unittest.main()
