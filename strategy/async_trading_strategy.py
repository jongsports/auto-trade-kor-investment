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

        self.take_profit_ratio = config.TAKE_PROFIT_RATIO
        self.stop_loss_ratio = config.STOP_LOSS_RATIO
        self.trailing_stop = config.TRAILING_STOP
        self.max_stocks = config.MAX_STOCKS
        
    def set_candidate_stocks(self, candidate_stocks):
         self.candidate_stocks = candidate_stocks

    async def update_holdings(self):
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
              
    async def check_entry_condition(self, ticker: str, ohlcv_data: pd.DataFrame) -> bool:
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
         rs = gain / loss
         rsi14 = (100 - (100 / (1 + rs))).iloc[-1]

         # Condition A: Strong momentum pullback (RSI between 40 and 60, price bounded by MAs)
         if ma20 < current_price < ma5 and 40 <= rsi14 <= 60:
             return True

         # Condition B: Oversold bounce (RSI < 30)
         if rsi14 < 30:
             return True

         return False

    async def check_exit_condition(self, ticker: str) -> tuple[bool, str]:
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
         
         # --- 1. OVERNIGHT EXIT LOGIC ---
         if strategy_type == "Overnight":
              now_str = datetime.now().strftime("%H:%M")
              if now_str >= config.OVERNIGHT_SELL_TIME:
                   return True, f"Overnight Morning Exit at {now_str}"
              
              # Fallback stop loss for overnight if it drops drastically right at open
              if profit_ratio <= -self.stop_loss_ratio * 1.5: 
                   return True, f"Overnight Hard Stop {profit_ratio:.2%}"
                   
              return False, "Hold Overnight"
              
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
