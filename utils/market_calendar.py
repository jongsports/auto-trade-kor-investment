"""거래일 달력 — 개장일 판정의 단일 출처.

우선순위:
  1. KIS 휴장일조회(CTCA0903R)로 확인되어 등록된 날짜
  2. 로컬 holidays.json (KIS 조회가 닿지 않은 날짜의 폴백)

holidays.json 은 대체공휴일·음력 공휴일이 빠지기 쉽다. 배포 서버에서는 기본
8일짜리 목록만 들고 있어 석가탄신일·광복절 대체공휴일과 추석 연휴에 봇이
정상 거래일처럼 돌았다. KIS 로 확인된 날짜는 파일에 누적 저장해 재시작 후에도
과거 보유일 계산에 쓸 수 있게 한다.
"""
import logging
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Union

from utils.state_store import load_state, save_state

logger = logging.getLogger("auto_trade.calendar")

_STATE_NAME = "open_days"
_confirmed: Optional[Dict[str, bool]] = None   # {YYYYMMDD: 개장 여부}
_holiday_cache: Optional[tuple] = None         # (로드한 날짜, frozenset)

DateLike = Union[date, datetime]


def _confirmed_days() -> Dict[str, bool]:
    global _confirmed
    if _confirmed is None:
        loaded = load_state(_STATE_NAME, {})
        _confirmed = {k: bool(v) for k, v in loaded.items()} if isinstance(loaded, dict) else {}
    return _confirmed


def register_open_days(days: Dict[str, bool]) -> None:
    """KIS 로 확인된 개장 여부를 등록하고 저장한다."""
    if not days:
        return
    known = _confirmed_days()
    changed = {k: v for k, v in days.items() if known.get(k) != v}
    if not changed:
        return
    known.update(changed)
    save_state(_STATE_NAME, dict(sorted(known.items())))


def _local_holidays() -> frozenset:
    """holidays.json 을 하루 한 번만 읽는다 (10초 틱마다 디스크를 읽지 않도록)."""
    global _holiday_cache
    today = datetime.now().date()
    if _holiday_cache is None or _holiday_cache[0] != today:
        from utils.utils import load_holidays
        _holiday_cache = (today, frozenset(str(h) for h in (load_holidays() or [])))
    return _holiday_cache[1]


def _as_date(d: Optional[DateLike]) -> date:
    if d is None:
        return datetime.now().date()
    return d.date() if isinstance(d, datetime) else d


def is_confirmed(d: Optional[DateLike] = None) -> bool:
    """해당 날짜의 개장 여부가 KIS 로 확인됐는지."""
    return _as_date(d).strftime("%Y%m%d") in _confirmed_days()


def is_trading_day(d: Optional[DateLike] = None) -> bool:
    """개장일 여부 — 시각은 보지 않는다."""
    day = _as_date(d)
    key = day.strftime("%Y%m%d")
    confirmed = _confirmed_days()
    if key in confirmed:
        return confirmed[key]
    if day.weekday() >= 5:
        return False
    return key not in _local_holidays()


def trading_days_between(start: DateLike, end: DateLike) -> int:
    """start 다음 날부터 end 까지(포함)의 거래일 수.

    금요일 15:10 매수 → 월요일이면 1 (달력일로는 3). Overnight 의 D+1/D+2 는
    거래일 기준이어야 주말을 낀 포지션이 D+1 익절 구간을 건너뛰지 않는다.
    """
    s, e = _as_date(start), _as_date(end)
    count = 0
    cur = s
    while cur < e:
        cur += timedelta(days=1)
        if is_trading_day(cur):
            count += 1
    return count


def _reset_for_tests() -> None:
    global _confirmed, _holiday_cache
    _confirmed = None
    _holiday_cache = None
