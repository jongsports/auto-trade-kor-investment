import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import threading

import config
from core.trader_api import AsyncKisAPI
from utils.utils import is_trading_time

logger = logging.getLogger("auto_trade.risk_manager")

class AsyncRiskManager:
    def __init__(self, api_client: AsyncKisAPI):
        self.api_client = api_client
        self.risk_status = "NORMAL"
        self.market_condition = "NORMAL"
        self.max_daily_loss = config.MAX_DAILY_LOSS

        # Limits
        self.max_position_size = config.MAX_STOCK_RATIO
        self.max_total_position = config.MAX_INVESTMENT_RATIO
        self.position_size_multiplier = {"NORMAL": 1.0, "CAUTION": 0.7, "RISK": 0.4}

    async def assess_market_risk(self):
        try:
            # U001 mapping to KOSPI
            kospi_data = await self.api_client.get_ohlcv("U001", "D", 20)
            if kospi_data.empty:
                 logger.warning("KOSPI data empty, skipping risk assessment.")
                 return
                 
            returns = kospi_data["close"].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            adjusted_volatility = volatility * 1.2
            
            logger.info(f"Market Volatility: {adjusted_volatility:.2f}")
            if adjusted_volatility > 20:
                 self.risk_status = "CAUTION" if adjusted_volatility < 30 else "RISK"
            else:
                 self.risk_status = "NORMAL"
                 
            ma20 = kospi_data["close"].rolling(20).mean().iloc[-1]
            ma60 = kospi_data["close"].rolling(60).mean().iloc[-1]
            current = kospi_data["close"].iloc[-1]
            
            if current > ma20 > ma60:
                 self.market_condition = "BULL"
            elif current < ma20 < ma60:
                 self.market_condition = "BEAR"
            else:
                 self.market_condition = "NORMAL"
                 
        except Exception as e:
            logger.error(f"Market risk assessment error: {e}")

    async def calculate_position_size(self, ticker: str, account_balance: float) -> float:
        try:
             price_data = await self.api_client.get_ohlcv(ticker, "D", 20)
             if price_data.empty: return 0
             
             returns = price_data["close"].pct_change().dropna()
             volatility = returns.std() * np.sqrt(252)
             
             # Volatility sizing
             vol_factor = 0.2 / volatility if volatility > 0 else 1.0
             position_size = self.max_position_size * vol_factor
             position_size = min(position_size, self.max_position_size)
             
             return account_balance * position_size
        except Exception as e:
             logger.error(f"Sizing error: {e}")
             return 0

    async def calculate_dynamic_stoploss(self, ticker: str, entry_price: float) -> float:
        """Calculate dynamic stop loss price based on ATR (Average True Range) / Volatility."""
        try:
             price_data = await self.api_client.get_ohlcv(ticker, "D", 20)
             if price_data.empty or len(price_data) < 14:
                 return entry_price * 0.95 # Fallback to 5% loss cut
                 
             # Calculate ATR 14
             high = price_data['high']
             low = price_data['low']
             close_prev = price_data['close'].shift(1)
             
             tr1 = high - low
             tr2 = (high - close_prev).abs()
             tr3 = (low - close_prev).abs()
             
             tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
             atr = tr.rolling(14).mean().iloc[-1]
             
             # Stop loss is entry price minus 1.5 * ATR (can be adjusted via risk_status)
             multiplier = self.position_size_multiplier.get(self.risk_status, 1.0)
             # Higher risk = tighter stop (larger multiplier = wider stop, so we invert)
             atr_factor = 2.0 if self.risk_status == "NORMAL" else (1.5 if self.risk_status == "CAUTION" else 1.0)
             
             dynamic_sl = entry_price - (atr * atr_factor)
             
             # Cap it at a maximum of 10% loss logically
             max_loss_price = entry_price * (1 - config.LOSS_CUT_RATIO * 2)
             
             return max(dynamic_sl, max_loss_price)
             
        except Exception as e:
             logger.error(f"Error calculating dynamic stop loss for {ticker}: {e}")
             return entry_price * 0.95

    async def can_trade(self, ticker: str, order_type: str, quantity: int, price: float) -> tuple[bool, str]:
        if not is_trading_time():
            return False, "Not trading time."
        if self.risk_status == "RISK" and order_type == "buy":
            return False, "Market is in extreme RISK."
        return True, "OK"
