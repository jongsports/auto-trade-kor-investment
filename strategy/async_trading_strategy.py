import logging
from datetime import datetime
from typing import Awaitable, Callable, Dict, List, Optional

import pandas as pd
import asyncio

import config
from core.trader_api import AsyncKisAPI
from risk.async_risk_manager import AsyncRiskManager
from utils import market_calendar
from utils.state_store import load_state, save_state
from utils.utils import get_trading_time_status

logger = logging.getLogger("auto_trade.trading_strategy")

# 주문 접수 후 잔고에 반영되기까지 기다리는 시간
ENTRY_FILL_GRACE_SECONDS = 60
PENDING_SELL_GRACE_SECONDS = 30
REBUY_COOLDOWN_SECONDS = 1800

# 재시작 후에도 유지해야 하는 포지션 필드
_PERSISTED_FIELDS = (
    "ticker", "name", "quantity", "buy_price", "high_price", "entry_time",
    "reason", "stop_price", "buy_trade_id", "pending_sell", "origin", "seen_at_broker", "score",
)

PositionRecoverer = Callable[[str], Awaitable[Optional[dict]]]


def rejection(reason: str) -> dict:
    """내부 사유로 주문을 내지 않았음을 나타내는 결과. API 실패와 구분된다."""
    return {"rt_cd": "-1", "msg_cd": "REJECTED", "msg1": reason, "_rejected": True}


class AsyncTradingStrategy:
    def __init__(self, api_client: AsyncKisAPI, risk_manager: AsyncRiskManager, candidate_stocks=None):
        self.api_client = api_client
        self.risk_manager = risk_manager
        self.candidate_stocks = candidate_stocks or []

        self.holdings: Dict[str, dict] = self._load_positions()
        self.order_history = []
        self._recently_sold: dict = {}        # Issue #21: 최근 매도 {ticker: unix_ts} (30분 쿨다운)
        self._unsellable_tickers: set = set() # 거래정지/매매불가 종목 (당일 재시도 차단)
        # 주문과 잔고 동기화를 직렬화한다. 스케줄러와 모니터 루프가 같은 분에 돌면서
        # 같은 종목을 이중 매수하거나, 동기화가 방금 넣은 포지션을 지우는 것을 막는다.
        self._order_lock = asyncio.Lock()

        # 잔고에는 있는데 메타데이터가 없는 포지션을 복원하는 훅 (DB 조회). trader 가 주입.
        self.position_recoverer: Optional[PositionRecoverer] = None
        # 출처를 알 수 없어 Standard 규칙으로 편입한 종목 — trader 가 알림 후 비운다.
        self.adopted_unknown: List[str] = []
        # 잔고에서 사라진 것이 확인된 청산 — trader 가 손익 기록·알림 후 비운다.
        self.closed_positions: List[dict] = []
        # 잔고에 처음 잡힌(=체결이 확인된) 매수 — trader 가 DB 기록·알림 후 비운다.
        self.opened_positions: List[dict] = []
        # 접수됐지만 체결되지 않은 것으로 보이는 매수 — trader 가 알림 후 비운다.
        self.unfilled_alerts: List[dict] = []
        # 매수 접수 후 유예 시간 안에 잔고에 잡히지 않은 주문. 늦게 체결되면 여기서
        # 메타데이터를 되찾고, 그동안 같은 종목을 다시 사지 않는다.
        self._unfilled_entries: Dict[str, dict] = self._load_entries("unfilled_entries")

        self.take_profit_ratio = config.TAKE_PROFIT_RATIO
        self.stop_loss_ratio = config.STOP_LOSS_RATIO
        self.trailing_stop = config.TRAILING_STOP
        self.max_stocks = config.MAX_STOCKS

    # ------------------------------------------------------------------ #
    # 포지션 영속화
    # ------------------------------------------------------------------ #

    @classmethod
    def _load_positions(cls) -> Dict[str, dict]:
        """저장된 포지션 메타데이터 복원.

        메모리에만 두면 재시작·재배포 때 reason/entry_time 이 사라져 Overnight 로 산
        종목이 다른 청산 규칙을 타고 보유일 시계도 0으로 돌아간다.
        """
        restored = cls._load_entries("positions")
        if restored:
            logger.info(f"[포지션복원] {len(restored)}종목: "
                        f"{[(t, i.get('reason')) for t, i in restored.items()]}")
        return restored

    @staticmethod
    def _load_entries(state_name: str) -> Dict[str, dict]:
        restored: Dict[str, dict] = {}
        for ticker, info in (load_state(state_name, {}) or {}).items():
            try:
                info = dict(info)
                info["entry_time"] = datetime.fromisoformat(info["entry_time"])
                restored[ticker] = info
            except (KeyError, TypeError, ValueError) as e:
                logger.error(f"[포지션복원] {ticker} 메타데이터 손상 — 건너뜀: {e}")
        return restored

    @staticmethod
    def _serializable(entries: Dict[str, dict]) -> Dict[str, dict]:
        data = {}
        for ticker, info in entries.items():
            row = {}
            for k in _PERSISTED_FIELDS:
                if k not in info:
                    continue
                v = info[k]
                # 잔고·시세에서 온 numpy 스칼라는 json 이 직렬화하지 못한다
                row[k] = v.item() if hasattr(v, "item") else v
            row["entry_time"] = info["entry_time"].isoformat()
            data[ticker] = row
        return data

    def _persist(self) -> None:
        save_state("positions", self._serializable(self.holdings))
        save_state("unfilled_entries", self._serializable(self._unfilled_entries))

    def set_candidate_stocks(self, candidate_stocks):
         self.candidate_stocks = candidate_stocks

    def reset_daily(self):
         """일별 초기화.

         거래정지는 해제될 수 있으므로 매도 차단 목록을 비운다. 비우지 않으면
         컨테이너가 수주간 떠 있는 동안 영구 차단이 되어 포지션이 holdings에
         남은 채 투자 비율만 잠식한다.
         """
         if self._unsellable_tickers:
              logger.info(f"[일별초기화] 매도 차단 해제: {sorted(self._unsellable_tickers)}")
              self._unsellable_tickers.clear()
         cutoff = datetime.now().timestamp() - REBUY_COOLDOWN_SECONDS
         self._recently_sold = {t: ts for t, ts in self._recently_sold.items() if ts > cutoff}
         if self._unfilled_entries:
              logger.info(f"[일별초기화] 미체결 매수 기록 정리: {sorted(self._unfilled_entries)}")
              self._unfilled_entries.clear()
              self._persist()

    def open_tickers(self) -> List[str]:
         """청산 판단 대상 — 매도 주문이 걸려 있거나 매도 차단된 종목은 제외."""
         return [t for t, i in self.holdings.items()
                 if not i.get("pending_sell") and t not in self._unsellable_tickers]

    # ------------------------------------------------------------------ #
    # 잔고 동기화
    # ------------------------------------------------------------------ #

    async def update_holdings(self) -> bool:
         """증권사 잔고와 대사한다. 조회 실패 시 False (로컬 상태 유지)."""
         async with self._order_lock:
             return await self._update_holdings_inner()

    async def _update_holdings_inner(self) -> bool:
         account_info = await self.api_client.get_account_summary()
         if not account_info:
              return False

         now = datetime.now()
         broker = {p["ticker"]: p for p in account_info.get("positions", []) if p.get("ticker")}
         reconciled: Dict[str, dict] = {}

         for ticker, pos in broker.items():
              info = self.holdings.get(ticker)
              info = dict(info) if info else await self._adopt_position(ticker, now)

              info["ticker"] = ticker
              info["name"] = pos.get("name") or info.get("name") or ticker
              info["quantity"] = pos.get("quantity", 0)
              info["sellable_quantity"] = pos.get("sellable_quantity", info["quantity"])
              if pos.get("buy_price", 0) > 0:
                   info["buy_price"] = pos["buy_price"]   # 실제 매입 평단으로 교정
              info["current_price"] = pos.get("current_price", 0)
              info["profit_loss"] = pos.get("eval_profit_loss", 0)
              info["high_price"] = max(info.get("high_price") or 0, info["current_price"])
              info.pop("unconfirmed", None)
              if not info.get("seen_at_broker"):
                   info["seen_at_broker"] = True
                   if info.get("origin") == "bot":
                        # 접수는 체결이 아니다. 잔고에 잡힌 지금이 매수가 확인된 시점이고,
                        # 수량과 평단도 이제 실제 값이다.
                        self.opened_positions.append(dict(info))

              pending = info.get("pending_sell")
              if pending and info["sellable_quantity"] > 0:
                   age = (now - datetime.fromisoformat(pending["at"])).total_seconds()
                   if age > PENDING_SELL_GRACE_SECONDS:
                        logger.warning(
                             f"[매도미체결] {ticker}: 접수 {age:.0f}초 후에도 매도가능수량 "
                             f"{info['sellable_quantity']}주 — 주문이 풀린 것으로 보고 재시도 허용"
                        )
                        info["pending_sell"] = None
              reconciled[ticker] = info

         for ticker, info in self.holdings.items():
              if ticker in broker:
                   continue
              if info.get("pending_sell"):
                   logger.info(f"[청산확정] {ticker}: 잔고에서 제거됨")
                   self.closed_positions.append(dict(info))
                   self._recently_sold[ticker] = now.timestamp()
                   continue
              age = (now - info["entry_time"]).total_seconds()
              if age < ENTRY_FILL_GRACE_SECONDS:
                   reconciled[ticker] = info   # 매수 체결이 아직 잔고에 안 잡힘
                   continue
              if info.get("origin") == "bot" and not info.get("seen_at_broker"):
                   # 한 번도 잔고에 잡힌 적 없는 매수. 체결이 늦는 것일 수 있으니 기록을
                   # 남겨 두고(늦게 잡히면 복원), 오늘은 같은 종목을 다시 사지 않는다.
                   logger.warning(f"[매수미체결] {ticker}: 접수 {age:.0f}초 후에도 잔고에 없음")
                   self._unfilled_entries[ticker] = info
                   self.unfilled_alerts.append(dict(info))
                   continue
              logger.warning(f"[포지션소실] {ticker}: 잔고에 없음 — 외부 매도로 보고 제거")

         self.holdings = reconciled
         self._persist()
         return True

    async def _adopt_position(self, ticker: str, now: datetime) -> dict:
         """잔고에는 있는데 로컬 메타데이터가 없는 포지션."""
         late = self._unfilled_entries.pop(ticker, None)
         if late:
              logger.warning(f"[매수체결확인] {ticker}: 유예 시간 이후 체결 — 메타데이터 복원")
              return {**late, "seen_at_broker": False}

         if self.position_recoverer is not None:
              try:
                   recovered = await self.position_recoverer(ticker)
              except Exception as e:
                   logger.error(f"[포지션복원] {ticker} DB 조회 실패: {e}")
                   recovered = None
              if recovered:
                   logger.warning(
                        f"[포지션복원] {ticker}: DB 매수 기록에서 복원 "
                        f"(reason={recovered.get('reason')} entry={recovered.get('entry_time')})"
                   )
                   return {**recovered, "origin": "db"}

         logger.warning(f"[포지션편입] {ticker}: 출처 불명 — Standard 청산 규칙 적용")
         self.adopted_unknown.append(ticker)
         return {"entry_time": now, "reason": "Standard", "origin": "unknown"}

    async def check_entry_condition(self, ticker: str, ohlcv_data: pd.DataFrame,
                                    market_regime: str = "NORMAL") -> bool:
         status = get_trading_time_status()
         if status not in ["REGULAR", "OPENING_AUCTION"]:
              return False

         if ticker in self.holdings: return False
         if len(self.holdings) >= self.max_stocks: return False

         if ohlcv_data.empty or len(ohlcv_data) < 20: return False

         # Entry Logic: Dip buying or strong momentum
         ma5 = ohlcv_data["close"].rolling(5).mean().iloc[-1]
         ma20 = ohlcv_data["close"].rolling(20).mean().iloc[-1]
         current_price = ohlcv_data["close"].iloc[-1]

         # RSI 14 - Wilder's smoothing (screener와 일관성 유지)
         delta = ohlcv_data["close"].diff()
         gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
         loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
         rs = gain / loss.replace(0, float('nan'))
         rsi14 = (100 - (100 / (1 + rs))).iloc[-1]
         if pd.isna(rsi14):
             rsi14 = 100.0  # 전부 상승 → RSI 100

         # Condition A: Strong momentum pullback (RSI between 40 and 60, price bounded by MAs)
         if ma20 < current_price < ma5 and 40 <= rsi14 <= 60:
             return True

         # Condition B: Oversold bounce (RSI < 30)
         if rsi14 < 30:
             return True

         return False

    def _b_overnight_decision(self, profit_ratio: float, days_held: int,
                              now_str: str) -> tuple[bool, str]:
        """B 패치 Overnight 청산 로직 (실제 매매용)."""
        # (1) 하드 스탑 — 시점 무관
        if profit_ratio <= -config.OVERNIGHT_HARD_STOP:
            return True, f"Overnight Hard Stop {profit_ratio:.2%}"
        # (2) D+0: 무조건 보유
        if days_held < 1:
            return False, "Hold Overnight (D+0)"
        # (3) D+1: +5% 조기 익절
        if days_held == 1:
            if profit_ratio >= config.OVERNIGHT_TAKE_PROFIT:
                return True, f"Overnight D+1 TP {profit_ratio:.2%}"
            return False, f"Hold Overnight (D+1 {profit_ratio:+.2%})"
        # (4) D+2 이상: 오전 또는 종가권 강제 청산
        if days_held >= config.OVERNIGHT_MAX_HOLD_DAYS:
            if config.OVERNIGHT_SELL_START <= now_str <= config.OVERNIGHT_SELL_END:
                return True, f"Overnight D+{days_held} Morning Exit at {now_str}"
            if now_str >= "14:00":
                return True, f"Overnight D+{days_held} Close Exit at {now_str}"
            return False, f"Hold Overnight (D+{days_held} pre-window)"
        return False, "Hold Overnight"

    def _shadow_c_overnight_decision(self, holding_info: dict, profit_ratio: float,
                                     days_held: int, now_str: str) -> tuple[bool, str]:
        """C 패치 트레일링 스탑 로직 (Shadow 모드 — 로깅 전용, 매매 영향 없음)."""
        buy_price = holding_info.get("buy_price", 0) or 1
        current_price = holding_info.get("current_price", 0)
        high = holding_info.get("high_price", current_price) or current_price

        # 하드 스탑 (B와 동일)
        if profit_ratio <= -config.OVERNIGHT_HARD_STOP:
            return True, f"Hard Stop {profit_ratio:.2%}"
        # D+0 보유
        if days_held < 1:
            return False, "Hold D+0"
        # D+1: 트레일링 + 러너 TP
        if days_held == 1:
            activation_level = buy_price * (1 + config.OVERNIGHT_TRAILING_ACTIVATION)
            if high > activation_level and current_price > 0:
                drop = 1 - current_price / high
                if drop >= config.OVERNIGHT_TRAILING_STOP:
                    return True, f"D+1 Trailing {drop:.2%} peak +{high/buy_price-1:.2%}"
            if profit_ratio >= config.OVERNIGHT_RUNNER_TP:
                return True, f"D+1 Runner TP {profit_ratio:.2%}"
            return False, f"Hold D+1 peak +{high/buy_price-1:.2%}"
        # D+2 이상: B와 동일 강제 청산
        if days_held >= config.OVERNIGHT_MAX_HOLD_DAYS:
            if config.OVERNIGHT_SELL_START <= now_str <= config.OVERNIGHT_SELL_END:
                return True, f"D+{days_held} Morning"
            if now_str >= "14:00":
                return True, f"D+{days_held} Close"
            return False, f"Hold D+{days_held}"
        return False, "Hold"

    async def _current_price(self, ticker: str, holding_info: dict) -> int:
         """현재가. 시세 조회가 실패하면 마지막 잔고 동기화 값을 쓴다.

         조회 실패를 이유로 청산 판단을 건너뛰면 급락 중 API 가 흔들릴 때 손절이 멈춘다.
         """
         price_data = await self.api_client.get_current_price(ticker)
         if price_data and price_data.get("price"):
              return price_data["price"]
         fallback = holding_info.get("current_price", 0)
         if fallback:
              logger.warning(f"[시세조회실패] {ticker}: 잔고 기준가 {fallback:,}원으로 청산 판단")
         return fallback

    async def check_exit_condition(self, ticker: str, market_regime: str = "NORMAL") -> tuple[bool, str]:
         holding_info = self.holdings.get(ticker)
         if holding_info is None:
              return False, "Not held"
         if holding_info.get("pending_sell"):
              return False, "Sell order pending"

         current_price = await self._current_price(ticker, holding_info)
         status = get_trading_time_status()

         if not current_price:
              return False, "Invalid current price"

         holding_info["current_price"] = current_price
         buy_price = holding_info.get("buy_price", 0)
         if buy_price <= 0:
              return False, "Invalid buy price"
         profit_ratio = current_price / buy_price - 1
         holding_info["high_price"] = max(holding_info.get("high_price") or current_price, current_price)

         strategy_type = holding_info.get("reason", "Standard")
         now = datetime.now()
         # 보유일은 거래일 기준. 달력일로 세면 금요일 매수분이 월요일에 D+3 이 되어
         # D+1 익절 구간 없이 바로 강제 청산된다.
         days_held = market_calendar.trading_days_between(holding_info["entry_time"], now)

         # --- 1. OVERNIGHT EXIT LOGIC (v2: 2026-04-24) ---
         # Shadow C: 트레일링 스탑 로직을 로그로 병행 기록. 실제 매매 영향 없음.
         if strategy_type == "Overnight":
              now_str = now.strftime("%H:%M")

              b_exit, b_reason = self._b_overnight_decision(profit_ratio, days_held, now_str)

              if getattr(config, "OVERNIGHT_SHADOW_C_ENABLED", False):
                   c_exit, c_reason = self._shadow_c_overnight_decision(
                        holding_info, profit_ratio, days_held, now_str
                   )
                   high_gain = holding_info["high_price"] / buy_price - 1
                   logger.info(
                        f"[SHADOW_C] {ticker} D+{days_held} pnl={profit_ratio:+.2%} "
                        f"peak={high_gain:+.2%} "
                        f"B={'EXIT' if b_exit else 'HOLD'} C={'EXIT' if c_exit else 'HOLD'} "
                        f"C_reason={c_reason}"
                   )

              return b_exit, b_reason

         # --- 2. MOMENTUM / INTRADAY / STANDARD LOGIC ---
         is_short_term = strategy_type in ("Momentum", "Intraday")
         trailing_threshold = 0.02 if is_short_term else self.trailing_stop
         take_profit_threshold = 0.03 if is_short_term else self.take_profit_ratio
         max_holding_days = 1 if is_short_term else 5

         if profit_ratio >= take_profit_threshold:
              return True, f"목표 수익권 도달 ({profit_ratio:.2%})"

         # 손절가는 진입 시점에 정해 고정한다. 매 틱 재계산하면 장중 리스크 상태가
         # 바뀌는 순간 손절선이 뛰어 멀쩡한 포지션이 즉시 청산된다.
         stop_price = holding_info.get("stop_price")
         if not stop_price:
              stop_price = await self.risk_manager.calculate_dynamic_stoploss(ticker, buy_price)
              holding_info["stop_price"] = stop_price
              self._persist()
         if current_price <= stop_price:
              return True, f"리스크 관리 손절 (하단 지지선 {stop_price:.0f} 돌파)"

         if profit_ratio <= -self.stop_loss_ratio:
              return True, f"최대 허용 손실 초과 ({profit_ratio:.2%})"

         trailing = 1 - (current_price / holding_info["high_price"])
         if trailing >= trailing_threshold and holding_info["high_price"] > buy_price * 1.015:
              return True, f"고점 대비 하락 (트레일링 스탑 {trailing:.2%})"

         if status == "CLOSING_AUCTION":
              return True, "장 마감 전 동시호가 청산"

         if days_held >= max_holding_days:
              return True, f"최대 보유 기간 경과 ({days_held}거래일)"

         return False, "Hold"

    def in_rebuy_cooldown(self, ticker: str) -> Optional[int]:
        """매도 후 재매수 쿨다운 중이면 경과 분, 아니면 None."""
        sold_at = self._recently_sold.get(ticker)
        if sold_at is None:
            return None
        elapsed = datetime.now().timestamp() - sold_at
        return int(elapsed / 60) if elapsed < REBUY_COOLDOWN_SECONDS else None

    async def entry(self, ticker: str, quantity: int = 0, price: int = 0,
                    reason: str = "Momentum", name: str = "", score: float = 0.0):
        """시장가 매수. 수량은 리스크 한도 안에서 정하며 quantity 는 상한으로만 쓴다.

        주문을 내지 않은 경우 `_rejected` 결과를 반환한다 (API 실패와 구분).
        """
        async with self._order_lock:
            if ticker in self.holdings:
                return self._reject(ticker, "이미 보유 중")
            if ticker in self._unfilled_entries:
                return self._reject(ticker, "오늘 접수한 매수 주문의 체결 여부 미확인")

            can, msg = await self.risk_manager.can_trade(ticker, "buy")
            if not can:
                return self._reject(ticker, msg)

            price_data = await self.api_client.get_current_price(ticker)
            current_price = price_data["price"] if price_data else 0
            if not current_price:
                return self._reject(ticker, "현재가 조회 실패")
            limit = price_data.get("upper_limit") or 0
            if limit and current_price >= limit:
                return self._reject(ticker, "상한가 — 시장가 매수 불가")

            local_positions = {
                t: float(i.get("buy_price", 0)) * i.get("quantity", 0)
                for t, i in {**self._unfilled_entries, **self.holdings}.items()
            }
            plan = await self.risk_manager.plan_buy(ticker, current_price, local_positions)
            if not plan.ok:
                return self._reject(ticker, plan.reason)
            buy_qty = min(quantity, plan.quantity) if quantity > 0 else plan.quantity

            logger.info(f"[매수시도] {ticker} {buy_qty}주 (reason={reason})")
            result = await self.api_client.market_buy(ticker, buy_qty)

            accepted = result.get("rt_cd") == "0"
            if accepted or result.get("_unconfirmed"):
                stock_name = name or next(
                    (c.get("name") for c in self.candidate_stocks if c.get("ticker") == ticker), ticker)
                stop_price = None
                if reason != "Overnight":
                    stop_price = await self.risk_manager.calculate_dynamic_stoploss(ticker, current_price)
                # 접수 여부를 모르는 주문도 임시로 올려 둔다. 체결됐다면 메타데이터가
                # 보존되고, 아니면 동기화가 유예 시간 뒤에 제거한다.
                self.holdings[ticker] = {
                    "ticker": ticker,
                    "name": stock_name,
                    "quantity": buy_qty,
                    "sellable_quantity": buy_qty,
                    "buy_price": current_price,
                    "current_price": current_price,
                    "high_price": current_price,
                    "entry_time": datetime.now(),
                    "reason": reason,
                    "stop_price": stop_price,
                    "pending_sell": None,
                    "origin": "bot",
                    "unconfirmed": not accepted,
                    "seen_at_broker": False,
                    "score": float(score),
                }
                self._persist()
            return result

    @staticmethod
    def _reject(ticker: str, reason: str) -> dict:
        logger.info(f"[매수거부] {ticker}: {reason}")
        return rejection(reason)

    def set_buy_trade_id(self, ticker: str, trade_id: Optional[int]) -> None:
        if ticker in self.holdings:
            self.holdings[ticker]["buy_trade_id"] = trade_id
            self._persist()

    async def exit(self, ticker: str, quantity: int = 0, reason: str = ""):
        """시장가 매도.

        접수(rt_cd=0)는 체결이 아니다. 접수 후에는 포지션을 지우지 않고 pending_sell 로
        표시해 두고, 잔고에서 사라진 것을 동기화가 확인했을 때 제거한다. 15:20 동시호가
        매도를 접수 즉시 지우면 15:30 체결 전까지 잔고 동기화가 다시 넣어 재매도한다.
        """
        async with self._order_lock:
            info = self.holdings.get(ticker)
            if info is None:
                logger.warning(f"[매도거부] {ticker}: 미보유 종목")
                return None
            if info.get("pending_sell") or ticker in self._unsellable_tickers:
                return None

            sellable = info.get("sellable_quantity", info.get("quantity", 0))
            if sellable <= 0:
                logger.info(f"[매도보류] {ticker}: 매도가능수량 0 — 미체결 주문 대기")
                return None
            sell_qty = min(quantity, sellable) if quantity > 0 else sellable

            can, msg = await self.risk_manager.can_trade(ticker, "sell")
            if not can:
                logger.warning(f"[매도거부] {ticker}: {msg}")
                return None

            logger.info(f"[매도시도] {ticker} {sell_qty}주 (reason={reason})")
            result = await self.api_client.market_sell(ticker, sell_qty)

            accepted = result.get("rt_cd") == "0"
            if accepted or result.get("_unconfirmed"):
                # 손익은 여기서 기록하지 않는다. 접수된 주문이 체결 없이 풀릴 수 있고,
                # 그러면 같은 포지션의 손익이 두 번 잡힌다. 잔고에서 사라진 것이
                # 확인되면 closed_positions 로 넘어가고 trader 가 그때 기록한다.
                info["pending_sell"] = {
                    "at": datetime.now().isoformat(),
                    "reason": reason,
                    "quantity": sell_qty,
                    "price": await self._current_price(ticker, info),
                    "confirmed": accepted,
                }
                info["sellable_quantity"] = sellable - sell_qty
                self._recently_sold[ticker] = datetime.now().timestamp()
                self._persist()
            elif result.get("_unsellable"):
                self._unsellable_tickers.add(ticker)
                logger.warning(f"[매도차단] {ticker}: 거래정지/매매불가 — 당일 재시도하지 않음")
            return result
