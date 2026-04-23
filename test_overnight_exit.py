"""Overnight exit 로직 B 패치 단위 테스트 (읽기 전용, API 호출 없음)."""
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from strategy.async_trading_strategy import AsyncTradingStrategy


def make_strategy():
    api = MagicMock()
    rm = MagicMock()
    rm.calculate_dynamic_stoploss = AsyncMock(return_value=0)
    return AsyncTradingStrategy(api, rm)


def setup_holding(strategy, ticker, entry_date, buy_price, current_price):
    strategy.holdings[ticker] = {
        "ticker": ticker, "name": ticker,
        "quantity": 1, "buy_price": buy_price,
        "current_price": current_price, "high_price": current_price,
        "entry_time": datetime.combine(entry_date, datetime.min.time()).replace(hour=15, minute=10),
        "reason": "Overnight",
    }


async def run_case(desc, entry_days_ago, buy, current, fake_now, expect_exit, expect_reason_hint):
    s = make_strategy()
    entry_date = (datetime.now() - timedelta(days=entry_days_ago)).date()
    setup_holding(s, "TEST", entry_date, buy, current)
    s.api_client.get_current_price = AsyncMock(return_value={"price": current})

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fake_now

    with patch("strategy.async_trading_strategy.datetime", FakeDatetime):
        exit_flag, reason = await s.check_exit_condition("TEST")

    ok = (exit_flag == expect_exit) and (expect_reason_hint in reason)
    mark = "✓" if ok else "✗"
    print(f"  {mark} {desc}")
    print(f"     exit={exit_flag} reason={reason!r}")
    return ok


async def main():
    print("Overnight Exit v2 — 분기 테스트\n")
    results = []
    base_entry = datetime.now().replace(hour=15, minute=10, second=0, microsecond=0)

    # D+0 매수 직후 — 버그 재현 시나리오
    results.append(await run_case(
        "D+0 15:10 매수 직후, 현재가=매수가 (버그 재현 테스트)",
        0, buy=10000, current=10000,
        fake_now=base_entry.replace(hour=15, minute=10, second=30),
        expect_exit=False, expect_reason_hint="D+0"))

    # D+1 +3% (TP 미달)
    results.append(await run_case(
        "D+1 10:00 +3% (TP 미달)",
        1, buy=10000, current=10300,
        fake_now=datetime.now().replace(hour=10, minute=0, second=0, microsecond=0),
        expect_exit=False, expect_reason_hint="D+1"))

    # D+1 +5% TP 달성
    results.append(await run_case(
        "D+1 13:00 +5% (TP 달성)",
        1, buy=10000, current=10500,
        fake_now=datetime.now().replace(hour=13, minute=0, second=0, microsecond=0),
        expect_exit=True, expect_reason_hint="TP"))

    # D+2 09:10 오전 강제 청산
    results.append(await run_case(
        "D+2 09:10 오전 시가권 강제 청산",
        2, buy=10000, current=10200,
        fake_now=datetime.now().replace(hour=9, minute=10, second=0, microsecond=0),
        expect_exit=True, expect_reason_hint="Morning Exit"))

    # D+2 09:45 윈도우 밖 → pre-window 보류
    results.append(await run_case(
        "D+2 09:45 (오전 윈도우 밖, 종가까지 홀딩)",
        2, buy=10000, current=10100,
        fake_now=datetime.now().replace(hour=9, minute=45, second=0, microsecond=0),
        expect_exit=False, expect_reason_hint="pre-window"))

    # D+2 14:30 종가권 청산
    results.append(await run_case(
        "D+2 14:30 종가권 청산",
        2, buy=10000, current=10100,
        fake_now=datetime.now().replace(hour=14, minute=30, second=0, microsecond=0),
        expect_exit=True, expect_reason_hint="Close Exit"))

    # 하드스탑 -4.5%
    results.append(await run_case(
        "D+0 16:00 -4.5% (하드 스탑)",
        0, buy=10000, current=9550,
        fake_now=base_entry.replace(hour=16, minute=0, second=0, microsecond=0),
        expect_exit=True, expect_reason_hint="Hard Stop"))

    # ─── Shadow C 로직 테스트 ──────────────────────────────────────────
    print("\n--- Shadow C (트레일링) 분기 테스트 ---\n")
    s = make_strategy()
    c_tests = [
        # (desc, holding, profit_ratio, days_held, now_str, expect_exit, hint)
        ("D+0 어떤 상황이든 보유",
         {"buy_price": 10000, "current_price": 10300, "high_price": 10400},
         0.03, 0, "15:30", False, "D+0"),
        ("D+1 고점+2% 후 현재+1.8% (트레일링 3% 미달)",
         {"buy_price": 10000, "current_price": 10180, "high_price": 10200},
         0.018, 1, "11:00", False, "peak"),
        ("D+1 고점+5% 후 현재+1.5% (트레일링 3.33% 발동)",
         {"buy_price": 10000, "current_price": 10150, "high_price": 10500},
         0.015, 1, "11:00", True, "Trailing"),
        ("D+1 +10% Runner TP",
         {"buy_price": 10000, "current_price": 11000, "high_price": 11000},
         0.10, 1, "13:00", True, "Runner"),
        ("D+1 고점+1% (활성화 기준 미달이라 트레일링 무시)",
         {"buy_price": 10000, "current_price": 9800, "high_price": 10100},
         -0.02, 1, "13:00", False, "Hold"),
        ("D+2 09:10 강제 청산",
         {"buy_price": 10000, "current_price": 10300, "high_price": 10400},
         0.03, 2, "09:10", True, "Morning"),
        ("하드 스탑 -4.5%",
         {"buy_price": 10000, "current_price": 9550, "high_price": 10050},
         -0.045, 1, "10:00", True, "Hard Stop"),
    ]
    c_pass = 0
    for desc, h, pnl, dh, now, want_exit, hint in c_tests:
        got_exit, got_reason = s._shadow_c_overnight_decision(h, pnl, dh, now)
        ok = got_exit == want_exit and hint in got_reason
        mark = "✓" if ok else "✗"
        print(f"  {mark} {desc}")
        print(f"     exit={got_exit} reason={got_reason!r}")
        if ok:
            c_pass += 1

    b_passed = sum(results)
    print(f"\nB logic    : {b_passed}/{len(results)} 통과")
    print(f"Shadow C   : {c_pass}/{len(c_tests)} 통과")
    total_ok = (b_passed == len(results)) and (c_pass == len(c_tests))
    exit(0 if total_ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
