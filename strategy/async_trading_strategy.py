import logging
from datetime import datetime
import pandas as pd
import asyncio

import config
from core.trader_api import AsyncKisAPI
from risk.async_risk_manager import AsyncRiskManager
from utils.utils import get_trading_time_status

logger = logging.getLogger("auto_trade.trading_strategy")

class AsyncTradingStrategy:
    def __init__(self, api_client: AsyncKisAPI, risk_manager: AsyncRiskManager, candidate_stocks=None):
        self.api_client = api_client
        self.risk_manager = risk_manager
        self.candidate_stocks = candidate_stocks or []
        
        self.holdings = {}
        self.order_history = []
        self.pending_orders = {}
        self._selling_tickers: set = set()   # Issue #20: 매도 진행 중 종목 (중복 주문 방지)
        self._recently_sold: dict = {}        # Issue #21: 최근 매도 {ticker: unix_ts} (30분 쿨다운)
        self._holdings_lock = asyncio.Lock()  # Holdings 동시 접근 방지

        self.take_profit_ratio = config.TAKE_PROFIT_RATIO
        self.stop_loss_ratio = config.STOP_LOSS_RATIO
        self.trailing_stop = config.TRAILING_STOP
        self.max_stocks = config.MAX_STOCKS
        
    def set_candidate_stocks(self, candidate_stocks):
         self.candidate_stocks = candidate_stocks

    async def update_holdings(self):
         async with self._holdings_lock:
             await self._update_holdings_inner()

    async def _update_holdings_inner(self):
         account_info = await self.api_client.get_account_summary()
         if not account_info:
              return
              
         positions = account_info.get("positions", [])
         holdings_backup = self.holdings.copy()
         self.holdings = {}
         
         for position in positions:
              ticker = position.get("ticker", "")
              if not ticker: continue
              
              current_info = holdings_backup.get(ticker, {}).copy()
              current_info["ticker"] = ticker
              current_info["name"] = position.get("name", "")
              current_info["quantity"] = position.get("quantity", 0)
              current_info["buy_price"] = position.get("buy_price", 0)
              current_info["current_price"] = position.get("current_price", 0)
              current_info["profit_loss"] = position.get("eval_profit_loss", 0)
              
              if "entry_time" not in current_info:
                   current_info["entry_time"] = datetime.now()
              if "high_price" not in current_info:
                   current_info["high_price"] = position.get("current_price", 0)
              elif position.get("current_price", 0) > current_info["high_price"]:
                   current_info["high_price"] = position.get("current_price", 0)
                   
              self.holdings[ticker] = current_info
              
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

    async def check_exit_condition(self, ticker: str, market_regime: str = "NORMAL") -> tuple[bool, str]:
         if ticker not in self.holdings:
              return False, "Not held"
              
         holding_info = self.holdings[ticker]
         price_data = await self.api_client.get_current_price(ticker)
         current_price = price_data["price"] if price_data else 0
         status = get_trading_time_status()
         
         if not current_price: 
              return False, "Invalid current price"
         
         holding_info["current_price"] = current_price
         buy_price = holding_info.get("buy_price", 0)
         if buy_price <= 0:
              return False, "Invalid buy price"
         profit_ratio = current_price / buy_price - 1
         holding_info["high_price"] = max(holding_info.get("high_price", current_price), current_price)
         
         # Identify Strategy Type (assume 'reason' was stored during entry)
         # For backward compatibility, if 'reason' doesn't exist, we treat it as standard
         strategy_type = holding_info.get("reason", "Standard")
         
         # --- 1. OVERNIGHT EXIT LOGIC (v2: 2026-04-24) ---
         # Bug history: 이전 로직은 "now_str >= '09:05'" 문자열 사전식 비교 때문에
         # 15:10 매수 직후 exit 체크에서 '15:10' >= '09:05' 참 → 즉시 청산되어
         # 26건 연속 슬리피지 손실 누적.
         # Fix: today > entry_date 가드 + D+2 오전 강제 청산 + D+1 조기 TP.
         # Shadow C (2026-04-24): 트레일링 스탑 로직을 로그로 병행 기록. 실제 매매 영향 없음.
         if strategy_type == "Overnight":
              now_str = datetime.now().strftime("%H:%M")
              entry_date = holding_info["entry_time"].date()
              today = datetime.now().date()
              days_held = (today - entry_date).days

              # B 결정 계산 (실제 매매)
              b_exit, b_reason = self._b_overnight_decision(
                   profit_ratio, days_held, now_str
              )

              # C 결정 계산 (Shadow 로깅만)
              if getattr(config, "OVERNIGHT_SHADOW_C_ENABLED", False):
                   c_exit, c_reason = self._shadow_c_overnight_decision(
                        holding_info, profit_ratio, days_held, now_str
                   )
                   high = holding_info.get("high_price", current_price)
                   high_gain = (high / holding_info["buy_price"] - 1) if holding_info["buy_price"] else 0
                   logger.info(
                        f"[SHADOW_C] {ticker} D+{days_held} pnl={profit_ratio:+.2%} "
                        f"peak={high_gain:+.2%} "
                        f"B={'EXIT' if b_exit else 'HOLD'} C={'EXIT' if c_exit else 'HOLD'} "
                        f"C_reason={c_reason}"
                   )

              return b_exit, b_reason
              
         # --- 2. MOMENTUM / INTRADAY / DAY TRADE LOGIC ---
         # 스크리너는 "Overnight" / "Intraday" / "Momentum" 태그 반환 (Issue #9-D)
         # "Momentum"/"Intraday": 짧은 trailing stop, 당일 청산
         is_short_term = strategy_type in ("Momentum", "Intraday")
         trailing_threshold = 0.02 if is_short_term else self.trailing_stop
         take_profit_threshold = 0.03 if is_short_term else self.take_profit_ratio
         max_holding_days = 1 if is_short_term else 5

         # Take Profit
         if profit_ratio >= take_profit_threshold:
              return True, f"목표 수익권 도달 ({profit_ratio:.2%})"
              
         # Dynamic Stop Loss from Risk Manager
         dynamic_sl = await self.risk_manager.calculate_dynamic_stoploss(ticker, holding_info["buy_price"])
         if current_price <= dynamic_sl:
              return True, f"리스크 관리 손절 (하단 지지선 {dynamic_sl:.0f} 돌파)"
              
         # Fallback static stop loss
         if profit_ratio <= -self.stop_loss_ratio:
              return True, f"최대 허용 손실 초과 ({profit_ratio:.2%})"
              
         # Trailing Stop
         trailing = 1 - (current_price / holding_info["high_price"])
         if trailing >= trailing_threshold and holding_info["high_price"] > holding_info["buy_price"] * 1.015:
              return True, f"고점 대비 하락 (트레일링 스탑 {trailing:.2%})"
              
         # Market conditions
         if status == "CLOSING_AUCTION":
              return True, "장 마감 전 동시호가 청산"
              
         holding_days = (datetime.now() - holding_info["entry_time"]).days
         if holding_days >= max_holding_days:
              return True, f"최대 보유 기간 경과 ({holding_days}일)"
              
         return False, "Hold"

    async def entry(self, ticker: str, quantity: int = 0, price: int = 0,
                    reason: str = "Momentum"):
        """시장가 매수 주문 실행.

        quantity가 0이면 리스크 매니저로 수량 자동 계산.
        """
        can, msg = await self.risk_manager.can_trade(ticker, "buy", quantity, price)
        if not can:
            logger.warning(f"[매수거부] {ticker}: {msg}")
            return None

        if ticker in self.holdings:
            logger.warning(f"[매수거부] {ticker}: 이미 보유 중")
            return None

        # 수량이 지정되지 않은 경우 리스크 기반 자동 산정
        if quantity <= 0:
            account = await self.api_client.get_account_summary()
            available = account.get("available_amount", 0)
            if available <= 0:
                logger.warning(f"[매수거부] {ticker}: 가용 잔고 없음")
                return None
            position_amount = await self.risk_manager.calculate_position_size(ticker, available)
            price_data = await self.api_client.get_current_price(ticker)
            current_price = price_data["price"] if price_data else 0
            if not current_price:
                logger.warning(f"[매수거부] {ticker}: 현재가 조회 실패")
                return None
            quantity = max(1, int(position_amount // current_price))

        if quantity <= 0:
            logger.warning(f"[매수거부] {ticker}: 수량 0")
            return None

        logger.info(f"[매수시도] {ticker} {quantity}주 (reason={reason})")
        result = await self.api_client.market_buy(ticker, quantity)

        if result.get("rt_cd") == "0":
            price_data = await self.api_client.get_current_price(ticker)
            current_price = price_data["price"] if price_data else price
            # Find stock name from candidates if available
            stock_name = next((c["name"] for c in self.candidate_stocks if c["ticker"] == ticker), ticker)
            
            self.holdings[ticker] = {
                "ticker": ticker,
                "name": stock_name,
                "quantity": quantity,
                "buy_price": current_price or price,
                "current_price": current_price or price,
                "high_price": current_price or price,
                "entry_time": datetime.now(),
                "reason": reason,
            }
            self.order_history.append({
                "action": "BUY",
                "ticker": ticker,
                "quantity": quantity,
                "price": current_price or price,
                "time": datetime.now().isoformat(),
                "reason": reason,
            })
        return result

    async def exit(self, ticker: str, quantity: int = 0, reason: str = ""):
        """시장가 매도 주문 실행."""
        if ticker not in self.holdings:
            logger.warning(f"[매도거부] {ticker}: 미보유 종목")
            return None

        held_qty = self.holdings[ticker].get("quantity", 0)
        if held_qty <= 0:
            logger.warning(f"[매도거부] {ticker}: 보유수량 0")
            return None

        sell_qty = quantity if quantity > 0 else held_qty

        can, msg = await self.risk_manager.can_trade(ticker, "sell", sell_qty, 0)
        if not can:
            logger.warning(f"[매도거부] {ticker}: {msg}")
            return None

        logger.info(f"[매도시도] {ticker} {sell_qty}주 (reason={reason})")
        result = await self.api_client.market_sell(ticker, sell_qty)

        if result.get("rt_cd") == "0":
            price_data = await self.api_client.get_current_price(ticker)
            current_price = price_data["price"] if price_data else 0
            buy_price = self.holdings[ticker].get("buy_price", 0)
            profit_ratio = (current_price / buy_price - 1) if buy_price > 0 and current_price else 0
            self.order_history.append({
                "action": "SELL",
                "ticker": ticker,
                "quantity": sell_qty,
                "price": current_price or 0,
                "time": datetime.now().isoformat(),
                "reason": reason,
                "profit_ratio": profit_ratio,
            })
            del self.holdings[ticker]
        return result
