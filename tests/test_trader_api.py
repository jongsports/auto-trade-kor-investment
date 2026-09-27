"""AsyncKisAPI 오프라인 테스트 — 네트워크 호출 없음."""
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

from core.trader_api import AsyncKisAPI


def make_api() -> AsyncKisAPI:
    api = AsyncKisAPI("key", "secret", "12345678", demo_mode=False)
    api.is_connected = True
    api.access_token = "tok-1"
    api.init_session = AsyncMock()
    api._wait_rate_limit = AsyncMock()
    return api


class FakeResponse:
    def __init__(self, payload, status=200):
        self.status = status
        self._payload = payload

    async def json(self):
        return self._payload

    async def text(self):
        return str(self._payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    """side_effects 의 각 항목이 예외면 raise, 아니면 응답으로 돌려준다."""

    def __init__(self, side_effects):
        self.side_effects = list(side_effects)
        self.calls = 0
        self.closed = False

    def request(self, method, url, **kwargs):
        self.calls += 1
        effect = self.side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return FakeResponse(effect)


def run(coro):
    return asyncio.run(coro)


class OrderResendTest(unittest.TestCase):
    def test_order_is_not_resent_after_network_error(self):
        api = make_api()
        api.session = FakeSession([asyncio.TimeoutError(), {"rt_cd": "0"}])
        res = run(api.market_buy("005930", 1))
        self.assertEqual(api.session.calls, 1)
        self.assertTrue(res.get("_unconfirmed"))
        self.assertNotEqual(res.get("rt_cd"), "0")

    def test_sell_is_not_resent_after_network_error(self):
        api = make_api()
        api.session = FakeSession([ConnectionError("boom"), {"rt_cd": "0"}])
        res = run(api.market_sell("005930", 1))
        self.assertEqual(api.session.calls, 1)
        self.assertTrue(res.get("_unconfirmed"))

    def test_query_is_retried_after_network_error(self):
        api = make_api()
        api.session = FakeSession([asyncio.TimeoutError(), {"rt_cd": "0", "output": {}}])
        with patch("asyncio.sleep", new=AsyncMock()):
            res = run(api._fetch("GET", "/x", "TR"))
        self.assertEqual(api.session.calls, 2)
        self.assertEqual(res.get("rt_cd"), "0")

    def test_order_is_resent_after_explicit_tps_rejection(self):
        api = make_api()
        api.session = FakeSession([{"msg_cd": "EGW00201"}, {"rt_cd": "0", "output": {"ODNO": "0001"}}])
        with patch("asyncio.sleep", new=AsyncMock()):
            res = run(api.market_buy("005930", 1))
        self.assertEqual(api.session.calls, 2)
        self.assertEqual(res.get("rt_cd"), "0")

    def test_exception_log_omits_account_number(self):
        api = make_api()
        api.session = FakeSession([ConnectionError("https://host/x?CANO=12345678")])
        with self.assertLogs("auto_trade.api", level="ERROR") as cm:
            run(api.market_sell("005930", 1))
        self.assertNotIn("12345678", "\n".join(cm.output))


class TokenRefreshTest(unittest.TestCase):
    def test_rejected_current_token_is_reissued_even_if_not_locally_expired(self):
        from datetime import datetime, timedelta
        api = make_api()
        api.token_expire_time = datetime.now() + timedelta(hours=12)

        def reissue():
            api.access_token = "tok-2"
            return True

        api._sync_init = MagicMock(side_effect=reissue)
        api.session = FakeSession([{"msg_cd": "EGW00123"}, {"rt_cd": "0"}])
        res = run(api._fetch("GET", "/x", "TR"))
        api._sync_init.assert_called_once()
        self.assertEqual(res.get("rt_cd"), "0")

    def test_refresh_failure_is_reported_not_looped(self):
        api = make_api()
        api._sync_init = MagicMock(return_value=False)
        api.session = FakeSession([{"msg_cd": "EGW00123"}] * 5)
        res = run(api._fetch("GET", "/x", "TR"))
        self.assertEqual(res.get("msg_cd"), "TOKEN_REFRESH_FAILED")
        self.assertEqual(api._sync_init.call_count, 1)


class OrderNumberTest(unittest.TestCase):
    def test_uppercase_order_number_is_read(self):
        self.assertEqual(AsyncKisAPI._order_no({"output": {"ODNO": "0000123"}}), "0000123")
        self.assertEqual(AsyncKisAPI._order_no({"output": {"odno": "77"}}), "77")
        self.assertEqual(AsyncKisAPI._order_no({}), "")


class UnsellableClassificationTest(unittest.TestCase):
    def _sell(self, payload):
        api = make_api()
        api._fetch = AsyncMock(return_value=payload)
        return run(api.market_sell("005930", 1))

    def test_market_closed_error_is_not_a_ticker_block(self):
        res = self._sell({"rt_cd": "1", "msg_cd": "APBK0919", "msg1": "장운영일자가 주문일과 상이합니다"})
        self.assertIsNone(res.get("_unsellable"))

    def test_trading_halt_is_a_ticker_block(self):
        self.assertTrue(self._sell({"rt_cd": "1", "msg_cd": "APBK0066", "msg1": "x"}).get("_unsellable"))
        self.assertTrue(self._sell({"rt_cd": "1", "msg_cd": "Z", "msg1": "거래정지 종목"}).get("_unsellable"))


class InvestorTrendTest(unittest.TestCase):
    def test_estimate_uses_latest_cumulative_row_only(self):
        api = make_api()
        api._fetch = AsyncMock(return_value={"rt_cd": "0", "output2": [
            {"bsop_hour_gb": "5", "frgn_fake_ntby_qty": "000000000001926000", "orgn_fake_ntby_qty": "000000000000790000"},
            {"bsop_hour_gb": "4", "frgn_fake_ntby_qty": "000000000001963000", "orgn_fake_ntby_qty": "000000000000599000"},
            {"bsop_hour_gb": "1", "frgn_fake_ntby_qty": "000000000000729000", "orgn_fake_ntby_qty": "0"},
        ]})
        res = run(api.get_investor_trend("005930"))
        self.assertEqual(res["foreign_net_buy"], 1_926_000)
        self.assertEqual(res["institution_net_buy"], 790_000)
        self.assertEqual(res["source"], "estimate")

    def test_fallback_skips_blank_intraday_row(self):
        api = make_api()
        api._fetch = AsyncMock(side_effect=[
            {"rt_cd": "0", "output2": [{"bsop_hour_gb": "1", "frgn_fake_ntby_qty": "0", "orgn_fake_ntby_qty": "0"}]},
            {"rt_cd": "0", "output": [
                {"stck_bsop_date": "20260928", "frgn_ntby_qty": "", "orgn_ntby_qty": ""},
                {"stck_bsop_date": "20260923", "frgn_ntby_qty": "4513767", "orgn_ntby_qty": "-100"},
            ]},
        ])
        res = run(api.get_investor_trend("005930"))
        self.assertEqual(res["foreign_net_buy"], 4_513_767)
        self.assertEqual(res["institution_net_buy"], -100)
        self.assertEqual(res["as_of"], "20260923")

    def test_all_sources_failing_marks_data_unavailable(self):
        api = make_api()
        api._fetch = AsyncMock(return_value={"rt_cd": "-1"})
        res = run(api.get_investor_trend("005930"))
        self.assertIs(res.get("data_available"), False)


class AccountSummaryTest(unittest.TestCase):
    def _summary(self, output1, output2):
        api = make_api()
        api._fetch = AsyncMock(return_value={"rt_cd": "0", "output1": output1, "output2": [output2]})
        return run(api.get_account_summary())

    def test_positions_expose_sellable_quantity(self):
        s = self._summary(
            [{"pdno": "005930", "prdt_name": "삼성전자", "hldg_qty": "3", "ord_psbl_qty": "0",
              "pchs_avg_pric": "353000.0000", "prpr": "285500", "evlu_pfls_amt": "-67500"}],
            {"dnca_tot_amt": "1012920", "prvs_rcdl_excc_amt": "1012920", "cma_evlu_amt": "0", "tot_evlu_amt": "1298420"},
        )
        pos = s["positions"][0]
        self.assertEqual(pos["quantity"], 3)
        self.assertEqual(pos["sellable_quantity"], 0)
        self.assertEqual(pos["buy_price"], 353000.0)

    def test_available_cash_reflects_same_day_purchases(self):
        s = self._summary([], {"dnca_tot_amt": "1000000", "prvs_rcdl_excc_amt": "600000",
                               "cma_evlu_amt": "0", "tot_evlu_amt": "1000000"})
        self.assertEqual(s["available_amount"], 600000)

    def test_cma_account_falls_back_to_cma_balance(self):
        s = self._summary([], {"dnca_tot_amt": "0", "prvs_rcdl_excc_amt": "0",
                               "cma_evlu_amt": "500000", "tot_evlu_amt": "500000"})
        self.assertEqual(s["available_amount"], 500000)

    def test_blank_fields_do_not_raise(self):
        s = self._summary([{"pdno": "1", "hldg_qty": "1", "ord_psbl_qty": "", "pchs_avg_pric": "", "prpr": ""}],
                          {"dnca_tot_amt": "", "prvs_rcdl_excc_amt": "", "cma_evlu_amt": "", "tot_evlu_amt": ""})
        self.assertEqual(s["positions"][0]["sellable_quantity"], 1)
        self.assertEqual(s["available_amount"], 0)

    def test_failure_returns_empty_dict(self):
        api = make_api()
        api._fetch = AsyncMock(return_value={"rt_cd": "-1", "msg1": "x"})
        self.assertEqual(run(api.get_account_summary()), {})


class OhlcvTest(unittest.TestCase):
    def test_long_window_uses_range_endpoint(self):
        api = make_api()
        rows = [{"stck_bsop_date": d.strftime("%Y%m%d"), "stck_oprc": "10", "stck_hgpr": "11",
                 "stck_lwpr": "9", "stck_clpr": "10", "acml_vol": "100", "acml_tr_pbmn": "1000"}
                for d in pd.bdate_range("2026-05-01", periods=100)]
        api._fetch = AsyncMock(return_value={"rt_cd": "0", "output2": rows})
        df = run(api.get_ohlcv("005930", count=100))
        self.assertEqual(len(df), 100)
        self.assertEqual(api._fetch.await_args_list[0].args[2], "FHKST03010100")
        self.assertTrue(df["date"].is_monotonic_increasing)

    def test_short_window_keeps_single_cheap_call(self):
        api = make_api()
        rows = [{"stck_bsop_date": d.strftime("%Y%m%d"), "stck_oprc": "10", "stck_hgpr": "11",
                 "stck_lwpr": "9", "stck_clpr": "10", "acml_vol": "100"}
                for d in pd.bdate_range("2026-08-01", periods=30)]
        api._fetch = AsyncMock(return_value={"rt_cd": "0", "output": rows})
        df = run(api.get_ohlcv("005930", count=20))
        self.assertEqual(len(df), 20)
        self.assertEqual(api._fetch.await_count, 1)
        self.assertEqual(api._fetch.await_args.args[2], "FHKST01010400")

    def test_cache_is_bounded(self):
        from datetime import datetime, timedelta
        api = make_api()
        old = datetime.now() - timedelta(hours=2)
        cache = {f"k{i}": (None, old) for i in range(api._CACHE_MAX_ENTRIES)}
        api._cache_put(cache, "new", 1)
        self.assertEqual(list(cache), ["new"])


class OpenDayTest(unittest.TestCase):
    def test_calendar_is_filled_from_one_response(self):
        api = make_api()
        api._fetch = AsyncMock(return_value={"rt_cd": "0", "output": [
            {"bass_dt": "20260927", "opnd_yn": "N"},
            {"bass_dt": "20260928", "opnd_yn": "Y"},
            {"bass_dt": "20261005", "opnd_yn": "N"},
        ]})
        self.assertIs(run(api.is_open_day("20260927")), False)
        self.assertEqual(api.open_day_calendar, {"20260927": False, "20260928": True, "20261005": False})

    def test_single_dict_response(self):
        api = make_api()
        api._fetch = AsyncMock(return_value={"rt_cd": "0", "output": {"bass_dt": "20260928", "opnd_yn": "Y"}})
        self.assertIs(run(api.is_open_day("20260928")), True)

    def test_failure_returns_none(self):
        api = make_api()
        api._fetch = AsyncMock(return_value={"rt_cd": "1"})
        self.assertIsNone(run(api.is_open_day("20260928")))


class UniverseTest(unittest.TestCase):
    def test_ranking_is_by_trading_value_with_exclusions_and_price_band(self):
        api = make_api()
        api._fetch = AsyncMock(return_value={"rt_cd": "0", "output": [{"mksc_shrn_iscd": "000001"}]})
        tickers = run(api.get_top_market_stocks("0001", min_price=2000, max_price=130000))
        params = api._fetch.await_args.kwargs["params"]
        self.assertEqual(tickers, ["000001"])
        self.assertEqual(params["FID_BLNG_CLS_CODE"], "3")
        self.assertEqual(len(params["FID_TRGT_EXLS_CLS_CODE"]), 10)
        self.assertEqual(params["FID_INPUT_PRICE_1"], "2000")
        self.assertEqual(params["FID_INPUT_PRICE_2"], "130000")


if __name__ == "__main__":
    unittest.main()
