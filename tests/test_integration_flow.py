"""통합 흐름 — 가짜 KIS 서버 위에서 실제 클래스들을 끝까지 돌린다.

AsyncKisAPI._fetch 만 가짜로 바꾸고 나머지(API 파싱, 스크리너, 전략, 리스크,
트레이더)는 실제 코드를 쓴다. mock 으로는 보이지 않는 클래스 간 시그니처 불일치를
잡는 것이 목적이다.
"""
from contextlib import ExitStack
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pandas as pd

import config
from core.async_trader import AsyncAutoTrader
from core.trader_api import AsyncKisAPI
from data.async_news_analyzer import AsyncNewsAnalyzer
from risk.async_risk_manager import AsyncRiskManager
from strategy.async_screener import AsyncStockScreener
from strategy.async_trading_strategy import AsyncTradingStrategy
from tests.helpers import IsolatedStateTestCase, freeze_time, run

CLOCK_MODULES = ("utils.utils", "utils.market_calendar", "strategy.async_screener",
                 "strategy.async_trading_strategy", "core.async_trader",
                 "core.trader_api", "risk.async_risk_manager")


class FakeBroker(AsyncKisAPI):
    """잔고와 시세를 가진 가짜 KIS. 시장가 주문은 현재가에 즉시 체결된다."""

    def __init__(self, cash=1_300_000, fill_orders=True):
        super().__init__("k", "s", "12345678", demo_mode=False)
        self.is_connected = True
        self.access_token = "t"
        self.cash = cash
        self.positions = {}          # ticker -> {"qty", "avg", "working_sell"}
        self.prices = {}             # ticker -> 현재가 dict
        self.daily = {}              # ticker -> 완성된 일봉 list (오래된 → 최신)
        self.universe = []           # [(ticker, name)]
        self.orders = []
        self.fill_orders = fill_orders
        self.fail_balance = False

    # -- 시나리오 구성 -----------------------------------------------------
    def add_stock(self, ticker, name, last_close, *, today, end="2026-09-25"):
        dates = pd.bdate_range(end=end, periods=100)
        rows, close = [], last_close * 0.75
        step = (last_close / close) ** (1 / 99)
        for i, d in enumerate(dates):
            c = round(last_close * 0.75 * step ** i)
            rows.append({"stck_bsop_date": d.strftime("%Y%m%d"), "stck_oprc": str(round(c * 0.995)),
                         "stck_hgpr": str(round(c * 1.012)), "stck_lwpr": str(round(c * 0.988)),
                         "stck_clpr": str(c), "acml_vol": "100000", "acml_tr_pbmn": str(c * 100000)})
        self.daily[ticker] = rows
        self.prices[ticker] = today
        self.universe.append((ticker, name))

    def set_price(self, ticker, price):
        p = self.prices[ticker]
        p.update(price=price, high=max(p["high"], price), low=min(p["low"], price))

    # -- 가짜 서버 ---------------------------------------------------------
    async def _fetch(self, method, path, tr_id, *, resend_on_error=True, **kw):
        params = kw.get("params") or {}
        body = kw.get("json") or {}
        if tr_id == "FHPST01710000":
            lo = int(params["FID_INPUT_PRICE_1"] or 0)
            hi = int(params["FID_INPUT_PRICE_2"] or 0) or 10 ** 12
            out = [{"mksc_shrn_iscd": t, "hts_kor_isnm": n} for t, n in self.universe
                   if params["FID_INPUT_ISCD"] == "0001" and lo <= self.prices[t]["price"] <= hi]
            return {"rt_cd": "0", "output": out}
        if tr_id == "FHKST03010100":
            rows = self.daily[params["FID_INPUT_ISCD"]]
            return {"rt_cd": "0", "output2": list(reversed(rows[-100:]))}
        if tr_id == "FHKST01010400":
            return {"rt_cd": "0", "output": list(reversed(self.daily[params["FID_INPUT_ISCD"]][-30:]))}
        if tr_id == "FHKST01010100":
            p = self.prices[params["FID_INPUT_ISCD"]]
            return {"rt_cd": "0", "output": {
                "stck_prpr": str(p["price"]), "stck_oprc": str(p["open"]), "stck_hgpr": str(p["high"]),
                "stck_lwpr": str(p["low"]), "acml_vol": str(p["volume"]),
                "acml_tr_pbmn": str(p["price"] * p["volume"]), "prdy_ctrt": "1.0"}}
        if tr_id == "HHPTJ04160200":
            return {"rt_cd": "0", "output2": [
                {"bsop_hour_gb": "4", "frgn_fake_ntby_qty": "5000", "orgn_fake_ntby_qty": "3000"},
                {"bsop_hour_gb": "1", "frgn_fake_ntby_qty": "1000", "orgn_fake_ntby_qty": "0"}]}
        if tr_id == "FHKST01011800":
            return {"rt_cd": "0", "output": []}
        if tr_id == "CTCA0903R":
            return {"rt_cd": "0", "output": [{"bass_dt": params["BASS_DT"], "opnd_yn": "Y"}]}
        if tr_id == "TTTC8434R":
            if self.fail_balance:
                return {"rt_cd": "-1", "msg1": "timeout"}
            out1 = [{"pdno": t, "prdt_name": dict(self.universe).get(t, t), "hldg_qty": str(p["qty"]),
                     "ord_psbl_qty": str(p["qty"] - p["working_sell"]),
                     "pchs_avg_pric": f"{p['avg']:.4f}", "prpr": str(self.prices[t]["price"]),
                     "evlu_pfls_amt": str(int((self.prices[t]["price"] - p["avg"]) * p["qty"]))}
                    for t, p in self.positions.items()]
            stock = sum(self.prices[t]["price"] * p["qty"] for t, p in self.positions.items())
            return {"rt_cd": "0", "output1": out1, "output2": [{
                "dnca_tot_amt": str(self.cash), "prvs_rcdl_excc_amt": str(self.cash),
                "cma_evlu_amt": "0", "tot_evlu_amt": str(self.cash + stock)}]}
        if tr_id in ("TTTC0012U", "TTTC0011U"):
            assert resend_on_error is False, "주문 TR 은 재전송 금지여야 한다"
            ticker, qty = body["PDNO"], int(body["ORD_QTY"])
            price = self.prices[ticker]["price"]
            side = "BUY" if tr_id == "TTTC0012U" else "SELL"
            self.orders.append((side, ticker, qty))
            if side == "BUY":
                if price * qty > self.cash:
                    return {"rt_cd": "1", "msg_cd": "APBK0952", "msg1": "주문가능금액을 초과"}
                self.cash -= price * qty
                pos = self.positions.setdefault(ticker, {"qty": 0, "avg": 0.0, "working_sell": 0})
                pos["avg"] = (pos["avg"] * pos["qty"] + price * qty) / (pos["qty"] + qty)
                pos["qty"] += qty
            else:
                pos = self.positions.get(ticker)
                if not pos or qty > pos["qty"] - pos["working_sell"]:
                    return {"rt_cd": "1", "msg_cd": "APBK0400", "msg1": "주문 가능한 수량을 초과했습니다."}
                if self.fill_orders:
                    self.cash += price * qty
                    pos["qty"] -= qty
                    if pos["qty"] == 0:
                        del self.positions[ticker]
                else:
                    pos["working_sell"] += qty      # 동시호가: 접수만 되고 미체결
            return {"rt_cd": "0", "output": {"ODNO": f"{len(self.orders):010d}"}}
        raise AssertionError(f"처리하지 않은 TR: {tr_id}")

    def fill_working_sells(self):
        for t in list(self.positions):
            pos = self.positions[t]
            if pos["working_sell"]:
                self.cash += self.prices[t]["price"] * pos["working_sell"]
                pos["qty"] -= pos["working_sell"]
                pos["working_sell"] = 0
                if pos["qty"] == 0:
                    del self.positions[t]


def strong_close(price, volume=260_000):
    """고가 부근에서 마감 중인 당일 봉."""
    return {"price": price, "open": round(price * 0.97), "high": round(price * 1.002),
            "low": round(price * 0.965), "volume": volume}


def build_trader(broker):
    t = AsyncAutoTrader.__new__(AsyncAutoTrader)
    t.running = True
    t.demo_mode = False
    t.api_client = broker
    t.screener = AsyncStockScreener(broker)
    t.news_analyzer = AsyncNewsAnalyzer(broker)
    t.screener.news_analyzer = t.news_analyzer
    t.risk_manager = AsyncRiskManager(broker)
    t.strategy = AsyncTradingStrategy(broker, t.risk_manager)
    t.coordinator = None
    t.candidate_stocks = []
    t._exit_fail_counts = {}
    t._open_day_query_at = None
    t._trading_halted_on = ""
    t._restart_alerts = {}
    t._last_reported_regime = ""
    t._last_heartbeat_time = datetime.now()
    t.notifier = MagicMock(); t.notifier.send_message = AsyncMock(return_value=True)
    t.db = MagicMock()
    t.db.save_trade_buy = AsyncMock(side_effect=iter(range(100, 200)))
    t.db.save_trade_sell = AsyncMock(return_value=1)
    t.db.save_daily_summary = AsyncMock()
    t.db.get_open_buy = AsyncMock(return_value=None)
    t.strategy.position_recoverer = t._recover_position_from_db
    t._save_screening_results = AsyncMock()
    return t


class at:
    """여러 모듈의 시계를 한꺼번에 고정."""
    def __init__(self, now): self.now, self.stack = now, ExitStack()
    def __enter__(self):
        for m in CLOCK_MODULES:
            self.stack.enter_context(freeze_time(m, self.now))
        return self
    def __exit__(self, *a): self.stack.close()


MON_1510 = datetime(2026, 9, 28, 15, 10, 5)


def alerts(trader):
    return [c.args[0] for c in trader.notifier.send_message.await_args_list]


class OvernightRoundTripTest(IsolatedStateTestCase):
    def _broker(self, **kw):
        b = FakeBroker(**kw)
        b.add_stock("000100", "알파전자", 20_000, today=strong_close(20_600))
        b.add_stock("000200", "베타화학", 45_000, today=strong_close(46_400))
        b.add_stock("000300", "감마중공업", 900_000, today=strong_close(930_000))   # 예산 초과
        return b

    def test_entry_restart_and_take_profit(self):
        broker = self._broker()
        trader = build_trader(broker)

        with at(MON_1510):
            run(trader._overnight_entry())

        bought = {t for side, t, _ in broker.orders if side == "BUY"}
        self.assertTrue(bought, f"매수가 없음. 알림: {alerts(trader)}")
        self.assertNotIn("000300", bought)                       # 계좌로 살 수 없는 종목
        equity = 1_300_000
        for side, ticker, qty in broker.orders:
            amount = qty * broker.prices[ticker]["price"]
            self.assertLessEqual(amount, equity * config.MAX_STOCK_RATIO * 1.5 + 1)
        invested = sum(broker.prices[t]["price"] * p["qty"] for t, p in broker.positions.items())
        self.assertLessEqual(invested, equity * config.MAX_INVESTMENT_RATIO)
        self.assertLessEqual(len(broker.positions), config.MAX_STOCK_COUNT)

        ticker = sorted(bought)[0]
        info = trader.strategy.holdings[ticker]
        self.assertEqual(info["reason"], "Overnight")
        self.assertIsNotNone(info["buy_trade_id"])
        first_buy = trader.db.save_trade_buy.await_args_list[0].kwargs
        self.assertEqual(first_buy["name"], dict(broker.universe)[first_buy["ticker"]])

        # ── 재배포: 새 프로세스가 같은 상태 디렉터리에서 기동 ──────────────
        restarted = build_trader(broker)
        self.assertEqual(restarted.strategy.holdings[ticker]["reason"], "Overnight")
        self.assertEqual(restarted.strategy.holdings[ticker]["entry_time"], info["entry_time"])

        # ── 당일 15:25: D+0 은 보유 ────────────────────────────────────────
        with at(datetime(2026, 9, 28, 15, 25)):
            run(restarted._check_exit_conditions())
        self.assertEqual([o for o in broker.orders if o[0] == "SELL"], [])

        # ── 화요일 10:00: +6% → D+1 익절 ─────────────────────────────────
        buy_price = broker.positions[ticker]["avg"]
        broker.set_price(ticker, round(buy_price * 1.06))
        with at(datetime(2026, 9, 29, 10, 0)):
            run(restarted._check_exit_conditions())
            run(restarted._check_exit_conditions())     # 다음 틱
            realized = restarted.risk_manager.daily_realized_pnl()
        sells = [o for o in broker.orders if o[0] == "SELL" and o[1] == ticker]
        self.assertEqual(len(sells), 1)
        self.assertNotIn(ticker, restarted.strategy.holdings)
        kw = restarted.db.save_trade_sell.await_args.kwargs
        self.assertEqual(kw["buy_trade_id"], info["buy_trade_id"])
        self.assertEqual(kw["strategy"], "Overnight")
        self.assertIn("D+1 TP", kw["reason"])
        self.assertGreater(realized, 0)

    def test_hard_stop_fires_after_restart_without_any_candidates(self):
        # 재시작 직후 후보가 없는 날에도 손절 감시가 돌아야 한다.
        broker = self._broker()
        trader = build_trader(broker)
        with at(MON_1510):
            run(trader._overnight_entry())
        ticker = next(iter(broker.positions))

        restarted = build_trader(broker)
        restarted.strategy.holdings.clear()              # 상태 파일까지 잃은 최악의 경우
        restarted.db.get_open_buy = AsyncMock(return_value={
            "id": 100, "executed_at": MON_1510, "strategy": "Overnight", "name": "x", "price": 1})
        broker.set_price(ticker, round(broker.positions[ticker]["avg"] * 0.95))
        with at(datetime(2026, 9, 29, 9, 30)):
            run(restarted._check_exit_conditions())
            realized = restarted.risk_manager.daily_realized_pnl()
        kw = restarted.db.save_trade_sell.await_args.kwargs
        self.assertIn("Hard Stop", kw["reason"])
        self.assertEqual(kw["buy_trade_id"], 100)
        self.assertLess(realized, 0)

    def test_closing_auction_sell_is_not_repeated_while_unfilled(self):
        broker = self._broker(fill_orders=False)
        trader = build_trader(broker)
        broker.positions["000100"] = {"qty": 5, "avg": 20_000.0, "working_sell": 0}
        with at(datetime(2026, 9, 28, 15, 21)):
            for _ in range(6):                           # 15:21 ~ 15:22, 10초 틱
                run(trader._check_exit_conditions())
        sells = [o for o in broker.orders if o[0] == "SELL"]
        self.assertEqual(sells, [("SELL", "000100", 5)])
        self.assertFalse(any("청산 실패" in a or "청산 이상" in a for a in alerts(trader)))

        broker.fill_working_sells()                      # 15:30 체결
        with at(datetime(2026, 9, 28, 15, 30, 10)):
            run(trader._check_exit_conditions())
        self.assertEqual(trader.strategy.holdings, {})

    def test_balance_outage_blocks_buys_and_keeps_positions(self):
        broker = self._broker()
        trader = build_trader(broker)
        broker.positions["000100"] = {"qty": 5, "avg": 20_000.0, "working_sell": 0}
        with at(datetime(2026, 9, 28, 10, 0)):
            run(trader._check_exit_conditions())
        self.assertIn("000100", trader.strategy.holdings)

        broker.fail_balance = True
        with at(MON_1510):
            run(trader._overnight_entry())
            run(trader._check_exit_conditions())
        self.assertEqual([o for o in broker.orders if o[0] == "BUY"], [])
        self.assertIn("000100", trader.strategy.holdings)

    def test_over_invested_account_buys_nothing(self):
        # 현재 실계좌 상태: 삼성전자 1주가 계좌의 22% → 한도 20% 초과
        broker = self._broker(cash=1_012_920)
        broker.add_stock("005930", "삼성전자", 286_500, today=strong_close(285_500))
        broker.positions["005930"] = {"qty": 1, "avg": 353_000.0, "working_sell": 0}
        trader = build_trader(broker)
        with at(MON_1510):
            run(trader._overnight_entry())
        self.assertEqual([o for o in broker.orders if o[0] == "BUY"], [])
        self.assertTrue(any("투자 비율 한도" in a for a in alerts(trader)), alerts(trader))

    def test_reports_do_not_crash(self):
        broker = self._broker()
        trader = build_trader(broker)
        with at(MON_1510):
            run(trader._overnight_entry())
            run(trader._send_heartbeat())
            run(trader._generate_closing_report())
        self.assertTrue(any("하트비트" in a for a in alerts(trader)))
        self.assertTrue(any("마감 리포트" in a for a in alerts(trader)))


class MorningFlowTest(IsolatedStateTestCase):
    def test_premarket_then_opening_entry_runs_end_to_end(self):
        broker = FakeBroker()
        broker.add_stock("000100", "알파전자", 20_000, today=strong_close(20_300))
        trader = build_trader(broker)
        with at(datetime(2026, 9, 28, 8, 0)):
            run(trader._premarket_screening())
        with at(datetime(2026, 9, 28, 9, 5, 10)):
            run(trader._opening_validation_and_entry())
        with at(datetime(2026, 9, 28, 10, 30, 5)):
            run(trader._intraday_screening_and_entry())
            run(trader._continuous_signal_check())
        self.assertFalse(any("장애 발생" in a for a in alerts(trader)), alerts(trader))


if __name__ == "__main__":
    import unittest
    unittest.main()
