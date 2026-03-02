import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import threading

import config
from core.trader_api import AsyncKisAPI
from utils.utils import is_trading_time, get_trading_time_status

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
            # KOSPI 지수 대용: KODEX 200 ETF (069500) — 모의투자 포함 전 환경에서 데이터 제공
            # KIS API는 지수 자체 OHLCV에 별도 TR(FHKUP03500100)이 필요하므로 ETF로 대체
            kospi_data = await self.api_client.get_ohlcv("069500", "D", 100)
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

            n = len(kospi_data)
            ma20 = kospi_data["close"].rolling(min(20, n)).mean().iloc[-1]
            ma60 = kospi_data["close"].rolling(min(60, n)).mean().iloc[-1]
            current = kospi_data["close"].iloc[-1]

            if current > ma20 > ma60:
                self.market_condition = "BULL"
            elif current < ma20 < ma60:
                self.market_condition = "BEAR"
            else:
                self.market_condition = "NORMAL"

            logger.info(f"Market condition: {self.market_condition} | Risk: {self.risk_status}")

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
        """ATR 기반 동적 손절가 계산."""
        fallback = entry_price * (1 - config.LOSS_CUT_RATIO)
        try:
            price_data = await self.api_client.get_ohlcv(ticker, "D", 20)
            if price_data.empty or len(price_data) < 14:
                return fallback

            # ATR 14 (Wilder's EMA, screener와 동일)
            high = price_data["high"]
            low = price_data["low"]
            close_prev = price_data["close"].shift(1)

            tr = pd.concat([high - low,
                            (high - close_prev).abs(),
                            (low - close_prev).abs()], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1]

            # 리스크 상태에 따라 ATR 배수 조정 (NORMAL=2배, CAUTION=1.5배, RISK=1배)
            atr_factor = 2.0 if self.risk_status == "NORMAL" else (
                1.5 if self.risk_status == "CAUTION" else 1.0
            )

            dynamic_sl = entry_price - (atr * atr_factor)

            # config LOSS_CUT_RATIO(2%)를 최대 손실 하한선으로 사용
            max_loss_price = entry_price * (1 - config.LOSS_CUT_RATIO)

            return max(dynamic_sl, max_loss_price)

        except Exception as e:
            logger.error(f"Error calculating dynamic stop loss for {ticker}: {e}")
            return fallback

    async def can_trade(self, ticker: str, order_type: str, quantity: int, price: float) -> tuple[bool, str]:
        # 매도는 동시호가(CLOSING_AUCTION, 15:20~15:30)도 허용
        if order_type == "sell":
            status = get_trading_time_status()
            if status not in ("REGULAR", "CLOSING_AUCTION", "OPENING_AUCTION"):
                return False, f"매도 불가 시간 (status={status})"
        else:
            if not is_trading_time():
                return False, "Not trading time."
        if self.risk_status == "RISK" and order_type == "buy":
            return False, "Market is in extreme RISK."

        # 잔고 및 포지션 수 체크 (매수 시에만)
        if order_type == "buy":
            try:
                account = await self.api_client.get_account_summary()
                positions = account.get("positions", [])

                # 최대 보유 종목 수 체크
                if len(positions) >= config.MAX_STOCK_COUNT:
                    return False, f"최대 보유 종목 수 초과 ({len(positions)}/{config.MAX_STOCK_COUNT})"

                # 가용 잔고 체크
                available = account.get("available_amount", 0)
                order_amount = quantity * price if quantity > 0 and price > 0 else 0
                if order_amount > 0 and available < order_amount:
                    return False, f"가용 잔고 부족 (필요: {order_amount:,}원, 가용: {available:,}원)"

                # 전체 투자 비율 체크
                total_eval = account.get("total_evaluated_amount", 0)
                if total_eval > 0:
                    invested = sum(
                        int(p.get("current_price", 0)) * int(p.get("quantity", 0))
                        for p in positions
                    )
                    invest_ratio = invested / total_eval
                    if invest_ratio >= config.MAX_INVESTMENT_RATIO:
                        return False, f"최대 투자 비율 초과 ({invest_ratio:.1%}/{config.MAX_INVESTMENT_RATIO:.1%})"

            except Exception as e:
                logger.warning(f"can_trade 잔고 체크 오류: {e}")

        return True, "OK"
