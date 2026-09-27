"""테스트 공용 도구 — 상태 파일을 임시 디렉터리로 격리한다."""
import asyncio
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import config
from utils import market_calendar


def run(coro):
    return asyncio.run(coro)


class IsolatedStateTestCase(unittest.TestCase):
    """data/state 를 건드리지 않도록 STATE_DIR 을 임시 디렉터리로 바꾼다.

    달력은 기본적으로 주말만 휴장으로 본다 (holidays.json 을 읽지 않음).
    """

    holidays: frozenset = frozenset()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._patches = [
            patch.object(config, "STATE_DIR", Path(self._tmp.name)),
            patch.object(market_calendar, "_local_holidays", lambda: self.holidays),
        ]
        for p in self._patches:
            p.start()
        market_calendar._reset_for_tests()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        market_calendar._reset_for_tests()
        self._tmp.cleanup()


def freeze_time(module: str, now: datetime):
    """module 안의 datetime.now() 를 고정한다."""

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    return patch(f"{module}.datetime", Frozen)
