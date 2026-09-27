import unittest
from datetime import date, datetime

from tests.helpers import IsolatedStateTestCase
from utils import market_calendar as cal
from utils import utils


class TradingDayTest(IsolatedStateTestCase):
    holidays = frozenset({"20260817"})

    def test_weekend_is_closed(self):
        self.assertFalse(cal.is_trading_day(date(2026, 9, 26)))
        self.assertFalse(cal.is_trading_day(date(2026, 9, 27)))

    def test_local_holiday_is_closed(self):
        self.assertFalse(cal.is_trading_day(date(2026, 8, 17)))

    def test_weekday_is_open_regardless_of_clock(self):
        # 과거 버그: 07:00 에는 is_market_open() 이 False 라 개장일도 휴장 분기로 빠졌다.
        self.assertTrue(cal.is_trading_day(datetime(2026, 9, 28, 7, 0)))

    def test_kis_confirmation_overrides_local_calendar(self):
        # 로컬 달력에는 없는 추석 연휴
        self.assertTrue(cal.is_trading_day(date(2026, 9, 24)))
        cal.register_open_days({"20260924": False, "20260925": False})
        self.assertFalse(cal.is_trading_day(date(2026, 9, 24)))
        self.assertTrue(cal.is_confirmed(date(2026, 9, 24)))

    def test_confirmed_days_survive_restart(self):
        cal.register_open_days({"20260924": False})
        cal._reset_for_tests()   # 프로세스 재시작을 흉내
        self.assertFalse(cal.is_trading_day(date(2026, 9, 24)))

    def test_utils_helpers_use_the_same_calendar(self):
        cal.register_open_days({"20260928": False})
        self.assertFalse(utils.is_trading_day(datetime(2026, 9, 28, 10, 0)))


class TradingDaysBetweenTest(IsolatedStateTestCase):
    def test_same_day_is_zero(self):
        self.assertEqual(cal.trading_days_between(date(2026, 9, 14), date(2026, 9, 14)), 0)

    def test_next_weekday_is_one(self):
        self.assertEqual(cal.trading_days_between(date(2026, 9, 14), date(2026, 9, 15)), 1)

    def test_friday_to_monday_is_one(self):
        self.assertEqual(cal.trading_days_between(date(2026, 9, 18), date(2026, 9, 21)), 1)

    def test_holidays_are_skipped(self):
        cal.register_open_days({"20260924": False, "20260925": False})
        # 9/23(수) 매수 → 9/28(월): 목·금 추석, 토·일 주말 → 1거래일
        self.assertEqual(cal.trading_days_between(date(2026, 9, 23), date(2026, 9, 28)), 1)


if __name__ == "__main__":
    unittest.main()
