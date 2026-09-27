"""스크리너 — 당일 봉 처리, 점수 척도, 거래량 환산, 수급 결측."""
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock

import pandas as pd

from strategy.async_screener import AsyncStockScreener
from tests.helpers import IsolatedStateTestCase, freeze_time, run

TODAY = date(2026, 9, 28)


def bars(n=100, end="2026-09-25", close=10_000.0, volume=100_000):
    dates = pd.bdate_range(end=end, periods=n)
    closes = [close + i * 10 for i in range(n)]
    return pd.DataFrame({
        "date": dates, "open": closes, "high": [c + 100 for c in closes],
        "low": [c - 100 for c in closes], "close": closes,
        "volume": [volume] * n, "amount": [c * volume for c in closes],
    })


def quote(price=11_200, open_=11_000, high=11_250, low=10_950, volume=150_000):
    return {"price": price, "open": open_, "high": high, "low": low,
            "volume": volume, "amount": price * volume}


def make_screener():
    api = MagicMock()
    api.demo_mode = False
    api.stock_names = {"000001": "테스트"}
    api.get_account_summary = AsyncMock(return_value={"total_evaluated_amount": 1_300_000})
    api.get_ohlcv = AsyncMock(return_value=bars())
    api.get_current_price = AsyncMock(return_value=quote())
    api.get_investor_trend = AsyncMock(return_value={"foreign_net_buy": 1, "institution_net_buy": 1})
    api.get_top_market_stocks = AsyncMock(return_value=["000001"])
    return AsyncStockScreener(api), api


class TodayBarTest(IsolatedStateTestCase):
    def test_today_bar_is_added_once_when_daily_data_lacks_it(self):
        df = AsyncStockScreener._with_today_bar(bars(), quote(), today=TODAY)
        self.assertEqual(len(df), 101)
        self.assertEqual(df["close"].iloc[-1], 11_200)

    def test_today_bar_replaces_existing_one(self):
        # KIS 일봉이 당일 봉을 포함해 내려오는 경우 — 중복되면 안 된다.
        with_today = bars(end="2026-09-28")
        df = AsyncStockScreener._with_today_bar(with_today, quote(), today=TODAY)
        self.assertEqual(len(df), 100)
        self.assertEqual((pd.to_datetime(df["date"]).dt.date == TODAY).sum(), 1)
        self.assertEqual(df["close"].iloc[-1], 11_200)

    def test_premarket_uses_completed_bars_only(self):
        df = AsyncStockScreener._completed_bars(bars(end="2026-09-28"), today=TODAY)
        self.assertEqual(pd.to_datetime(df["date"]).dt.date.max(), date(2026, 9, 25))


class ElapsedFractionTest(IsolatedStateTestCase):
    def test_fraction_over_session(self):
        f = AsyncStockScreener._session_elapsed_fraction
        self.assertEqual(f(datetime(2026, 9, 28, 8, 0)), 0.0)
        self.assertAlmostEqual(f(datetime(2026, 9, 28, 12, 15)), 0.5)
        self.assertEqual(f(datetime(2026, 9, 28, 16, 0)), 1.0)


class VolumeSurgeTest(IsolatedStateTestCase):
    def _df(self, today_volume):
        s, _ = make_screener()
        df = AsyncStockScreener._with_today_bar(bars(), quote(volume=today_volume), today=TODAY)
        return s, s.calculate_technical_indicators(df)

    def test_partial_day_volume_is_projected(self):
        # 10:30 (경과 23%) 에 하루 평균의 60% — 하루치로 환산하면 2.6배
        s, df = self._df(60_000)
        self.assertFalse(s.check_volume_surge(df, elapsed_fraction=1.0))
        self.assertTrue(s.check_volume_surge(df, elapsed_fraction=90 / 390))

    def test_projection_is_floored_early_in_session(self):
        # 09:05 (경과 1%) 에 평균의 10% — 하한 20% 로 환산하면 0.5배
        s, df = self._df(10_000)
        self.assertFalse(s.check_volume_surge(df, elapsed_fraction=0.01))


class ScoreScaleTest(IsolatedStateTestCase):
    def test_intraday_score_is_not_multiplied(self):
        s, _ = make_screener()
        df = AsyncStockScreener._with_today_bar(bars(), quote(), today=TODAY)
        trend = {"foreign_net_buy": 1, "institution_net_buy": 1}
        sc = s.calculate_stock_score("X", df, trend, is_intraday=True,
                                     intraday_data=quote(), elapsed_fraction=0.5)
        parts = sc["technical"] + sc["volume"] + sc["order_flow"] + sc["news"] + sc["intraday_bonus"]
        self.assertEqual(sc["total"], min(100, parts))

    def test_scoring_does_not_mutate_or_duplicate_input(self):
        s, _ = make_screener()
        df = AsyncStockScreener._with_today_bar(bars(), quote(), today=TODAY)
        before = len(df)
        s.calculate_stock_score("X", df, {}, is_intraday=True, intraday_data=quote())
        self.assertEqual(len(df), before)
        self.assertNotIn("ma5", df.columns)


class OvernightRulesTest(IsolatedStateTestCase):
    def _score(self, q, trend):
        s, _ = make_screener()
        df = AsyncStockScreener._with_today_bar(bars(), q, today=TODAY)
        return s.calculate_stock_score("X", df, trend, is_overnight_window=True, elapsed_fraction=0.95)

    def test_close_near_low_is_vetoed(self):
        sc = self._score(quote(price=10_960, high=11_250, low=10_950),
                         {"foreign_net_buy": 1, "institution_net_buy": 1})
        self.assertEqual((sc["technical"], sc["volume"], sc["overnight_bonus"]), (0, 0, 0))

    def test_one_sided_flow_is_vetoed(self):
        sc = self._score(quote(price=11_240, high=11_250, low=10_950),
                         {"foreign_net_buy": 1, "institution_net_buy": -1})
        self.assertEqual(sc["overnight_bonus"], 0)

    def test_flat_candle_is_vetoed(self):
        sc = self._score(quote(price=11_000, high=11_000, low=11_000),
                         {"foreign_net_buy": 1, "institution_net_buy": 1})
        self.assertEqual((sc["technical"], sc["volume"]), (0, 0))


class ProcessTickerTest(IsolatedStateTestCase):
    def test_missing_flow_data_blocks_candidate_in_every_mode(self):
        s, api = make_screener()
        api.get_investor_trend = AsyncMock(return_value={
            "foreign_net_buy": 0, "institution_net_buy": 0, "data_available": False})
        for kw in ({}, {"is_intraday": True}, {"is_overnight_window": True}):
            self.assertEqual(run(s._process_ticker("000001", "KOSPI", **kw)), {})

    def test_yesterdays_flow_does_not_qualify_intraday_or_overnight(self):
        s, api = make_screener()
        s.get_entry_threshold = lambda *_: 0
        api.get_investor_trend = AsyncMock(return_value={
            "foreign_net_buy": 5000, "institution_net_buy": 5000,
            "source": "confirmed", "as_of": "20260925"})
        with freeze_time("strategy.async_screener", datetime(2026, 9, 28, 15, 10)):
            self.assertEqual(run(s._process_ticker("000001", "KOSPI", is_overnight_window=True)), {})
            self.assertEqual(run(s._process_ticker("000001", "KOSPI", is_intraday=True)), {})
        # 프리마켓은 전 거래일 확정치를 쓰는 것이 맞다
        with freeze_time("strategy.async_screener", datetime(2026, 9, 28, 8, 0)):
            self.assertEqual(run(s._process_ticker("000001", "KOSPI"))["ticker"], "000001")

    def test_quote_failure_blocks_intraday_candidate(self):
        s, api = make_screener()
        api.get_current_price = AsyncMock(return_value=None)
        self.assertEqual(run(s._process_ticker("000001", "KOSPI", is_intraday=True)), {})

    def test_candidate_carries_name_and_today_bar(self):
        s, api = make_screener()
        s.get_entry_threshold = lambda *_: 0
        with freeze_time("strategy.async_screener", datetime(2026, 9, 28, 15, 10)):
            c = run(s._process_ticker("000001", "KOSPI", is_overnight_window=True))
        self.assertEqual(c["name"], "테스트")
        snap = c["_ohlcv_snapshot"]
        self.assertEqual(pd.to_datetime(snap["date"]).dt.date.iloc[-1], TODAY)
        self.assertEqual(len(snap), 101)


class UniverseTest(IsolatedStateTestCase):
    def test_universe_is_limited_to_affordable_prices(self):
        s, api = make_screener()
        run(s.run_screening_async(["KOSPI"]))
        kwargs = api.get_top_market_stocks.await_args.kwargs
        self.assertEqual(kwargs["max_price"], int(1_300_000 * 0.1 * 1.5))
        self.assertGreater(kwargs["min_price"], 0)


class OpeningGapTest(IsolatedStateTestCase):
    def _validate(self, open_price, daily):
        s, api = make_screener()
        api.get_ohlcv = AsyncMock(return_value=daily)
        api.get_current_price = AsyncMock(return_value=quote(open_=open_price))
        with freeze_time("strategy.async_screener", datetime(2026, 9, 28, 9, 5)):
            return run(s.validate_opening_candidates([{"ticker": "000001"}]))

    def test_gap_uses_todays_open_against_last_completed_close(self):
        daily = bars(5, close=10_000.0)           # 마지막 종가 10,040
        out = self._validate(10_140, daily)       # +1.0%
        self.assertAlmostEqual(out[0]["opening_gap"], 0.01, places=3)

    def test_same_result_when_daily_data_already_has_today(self):
        daily = bars(6, end="2026-09-28", close=9_990.0)   # 전일 종가 10,030
        out = self._validate(10_130, daily)
        self.assertAlmostEqual(out[0]["opening_gap"], 0.01, places=3)

    def test_large_gaps_are_excluded(self):
        self.assertEqual(self._validate(10_700, bars(5)), [])   # +6.6%
        self.assertEqual(self._validate(9_600, bars(5)), [])    # -4.4%

    def test_unverifiable_gap_is_excluded(self):
        s, api = make_screener()
        api.get_current_price = AsyncMock(return_value=None)
        self.assertEqual(run(s.validate_opening_candidates([{"ticker": "000001"}])), [])


if __name__ == "__main__":
    import unittest
    unittest.main()
