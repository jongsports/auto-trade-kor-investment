import logging
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

import config
from core.trader_api import AsyncKisAPI
from utils.state_store import load_state, save_state
from utils.utils import is_trading_time, get_trading_time_status

logger = logging.getLogger("auto_trade.risk_manager")

# 시장가 매수는 KIS 가 상한가(+30%) 기준으로 증거금을 잡는다.
MARKET_ORDER_MARGIN = 1.30


@dataclass(frozen=True)
class BuyPlan:
    quantity: int
    reason: str = ""   # quantity == 0 일 때의 사유

    @property
    def ok(self) -> bool:
        return self.quantity > 0


def plan_buy_quantity(*, price: float, target_amount: float, equity: float, cash: float,
                      invested: float, position_count: int,
                      max_investment_ratio: float, max_stock_count: int) -> BuyPlan:
    """매수 수량을 정한다. 한도를 지킬 수 없으면 0주(진입 포기)다.

    수량 산정과 한도 검사가 한 곳에 있어야 "최소 1주는 산다" 같은 예외가 한도를
    뚫지 못한다. 과거에는 max(1, ...) 때문에 계좌의 27%짜리 주문이 나갔다.
    """
    if price <= 0 or equity <= 0:
        return BuyPlan(0, "가격 또는 총평가액을 알 수 없음")
    if position_count >= max_stock_count:
        return BuyPlan(0, f"최대 보유 종목 수 도달 ({position_count}/{max_stock_count})")

    room = equity * max_investment_ratio - invested
    if room <= 0:
        return BuyPlan(0, f"투자 비율 한도 도달 ({invested / equity:.1%} 보유 / "
                          f"한도 {max_investment_ratio:.1%})")

    limits = {
        f"종목당 예산 {target_amount:,.0f}원": target_amount,
        f"투자 비율 한도까지 남은 {room:,.0f}원": room,
        f"가용 현금 {cash:,.0f}원(시장가 증거금 {MARKET_ORDER_MARGIN:.0%} 기준)": cash / MARKET_ORDER_MARGIN,
    }
    binding = min(limits, key=limits.get)
    quantity = math.floor(limits[binding] / price)
    if quantity <= 0:
        return BuyPlan(0, f"주가 {price:,.0f}원이 {binding} 초과")
    return BuyPlan(quantity)

class AsyncRiskManager:
    def __init__(self, api_client: AsyncKisAPI):
        self.api_client = api_client
        self.risk_status = "NORMAL"
        self.market_condition = "NORMAL"
        self.max_daily_loss = config.MAX_DAILY_LOSS

        # Limits
        self.max_position_size = config.MAX_STOCK_RATIO
        self.max_total_position = config.MAX_INVESTMENT_RATIO
        self.position_size_multiplier = dict(config.RISK_POSITION_MULTIPLIER)

        # 일일 실현 손익 — 재시작해도 당일 손실 한도가 0으로 돌아가지 않도록 파일에 둔다.
        saved = load_state("daily_pnl", {})
        self._daily_pnl_date: str = str(saved.get("date", ""))
        self._daily_realized_pnl: float = float(saved.get("realized_pnl", 0.0))

    def daily_realized_pnl(self) -> float:
        """오늘의 실현 손익. 날짜가 바뀌었으면 0."""
        today = datetime.now().strftime("%Y%m%d")
        return self._daily_realized_pnl if self._daily_pnl_date == today else 0.0

    def record_trade_pnl(self, pnl_amount: float):
        """매도 후 실현 손익 기록."""
        today = datetime.now().strftime("%Y%m%d")
        self._daily_realized_pnl = self.daily_realized_pnl() + pnl_amount
        self._daily_pnl_date = today
        save_state("daily_pnl", {"date": today, "realized_pnl": self._daily_realized_pnl})
        logger.info(f"[일일 손익] 누적: {self._daily_realized_pnl:+,.0f}원")

    async def assess_market_risk(self):
        try:
            # KOSPI 지수 대용: KODEX 200 ETF (069500) — 모의투자 포함 전 환경에서 데이터 제공
            # KIS API는 지수 자체 OHLCV에 별도 TR(FHKUP03500100)이 필요하므로 ETF로 대체
            kospi_data = await self.api_client.get_ohlcv("069500", "D", 100)
            if kospi_data.empty:
                logger.warning("KOSPI data empty, skipping risk assessment.")
                return

            returns = kospi_data["close"].pct_change().dropna()
            # Issue #23: EWMA 변동성 (span=20) — 단순 std보다 안정적이고 최근 변동에 가중
            # FHKST01010400이 30행만 반환하므로 단순 std는 극단값에 과민 반응
            ewma_var = returns.ewm(span=min(20, len(returns))).var().iloc[-1]
            volatility = np.sqrt(ewma_var) * np.sqrt(252) * 100
            simple_vol = returns.std() * np.sqrt(252) * 100
            logger.info(f"Market Volatility: EWMA={volatility:.2f}% (simple={simple_vol:.2f}%, samples={len(returns)})")
            if volatility > config.VOLATILITY_RISK:
                self.risk_status = "RISK"
            elif volatility > config.VOLATILITY_CAUTION:
                self.risk_status = "CAUTION"
            else:
                self.risk_status = "NORMAL"

            n = len(kospi_data)
            ma20 = kospi_data["close"].rolling(min(20, n)).mean().iloc[-1]
            ma60 = kospi_data["close"].rolling(min(60, n)).mean().iloc[-1]
            current = kospi_data["close"].iloc[-1]

            if volatility > config.VOLATILITY_VOLATILE:
                # 고변동성 구간: 방향에 따라 VOLATILE_UP / VOLATILE_DOWN 세분화
                if current >= ma20:
                    self.market_condition = "VOLATILE_UP"
                else:
                    self.market_condition = "VOLATILE_DOWN"
            elif current > ma20 > ma60:
                self.market_condition = "BULL"
            elif current < ma20 < ma60:
                self.market_condition = "BEAR"
            else:
                self.market_condition = "NORMAL"

            logger.info(f"Market condition: {self.market_condition} | Risk: {self.risk_status}")

        except Exception as e:
            logger.error(f"Market risk assessment error: {e}")
            # 실패 시 보수적 기본값으로 전환 (stale BULL 상태 방지)
            self.risk_status = "CAUTION"
            if self.market_condition in ("BULL",):
                self.market_condition = "NORMAL"

    async def position_size_ratio(self, ticker: str) -> float:
        """총평가액 대비 종목당 목표 비중 — 변동성 역비례 (R4).

        기준 변동성 20%에서 max_position_size 배정.
        저변동(10%) → 확대, 고변동(40%) → 축소. 체제·리스크 배율 추가 적용.
        데이터를 못 받으면 0 (진입 포기).
        """
        try:
            price_data = await self.api_client.get_ohlcv(ticker, "D", 20)
            if price_data.empty:
                return 0.0
            returns = price_data["close"].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            vol_factor = 0.2 / volatility if volatility > 0 else 1.0

            # P4: config 기반 통합 배율 (단일 소스)
            regime = self.market_condition or "NORMAL"
            rp = config.get_regime_params(regime)
            regime_mult = rp.get("position_size_multiplier", 1.0)
            risk_mult = config.RISK_POSITION_MULTIPLIER.get(self.risk_status, 1.0)

            position_size = self.max_position_size * vol_factor * regime_mult * risk_mult
            # 상한: max×1.5 / 하한: max×0.3 (2.0→1.5 하향: 단일 종목 과집중 방지)
            position_size = max(self.max_position_size * 0.3,
                                min(position_size, self.max_position_size * 1.5))
            logger.info(
                f"[PosSizing-RM] {ticker}: vol={volatility:.2%} × regime={regime_mult:.2f} "
                f"× risk={risk_mult:.2f} → {position_size:.2%}"
            )
            return float(position_size)
        except Exception as e:
            logger.error(f"포지션 사이징 오류 {ticker}: {e}")
            return 0.0

    async def plan_buy(self, ticker: str, price: float) -> BuyPlan:
        """현재 계좌 상태에서 이 종목을 몇 주 살 수 있는지 정한다."""
        account = await self.api_client.get_account_summary()
        if not account:
            # 조회 실패는 "보유 없음"이 아니다. 한도를 확인할 수 없으면 사지 않는다.
            return BuyPlan(0, "잔고 조회 실패 — 한도 확인 불가")

        equity = float(account.get("total_evaluated_amount", 0))
        positions = account.get("positions", [])
        invested = float(sum(p.get("current_price", 0) * p.get("quantity", 0) for p in positions))

        ratio = await self.position_size_ratio(ticker)
        if ratio <= 0:
            return BuyPlan(0, "변동성 데이터 없음 — 사이징 불가")

        return plan_buy_quantity(
            price=price,
            target_amount=equity * ratio,
            equity=equity,
            cash=float(account.get("available_amount", 0)),
            invested=invested,
            position_count=len(positions),
            max_investment_ratio=config.MAX_INVESTMENT_RATIO,
            max_stock_count=config.MAX_STOCK_COUNT,
        )

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

            # LOSS_CUT_RATIO 를 최대 손실 하한선으로 사용
            max_loss_price = entry_price * (1 - config.LOSS_CUT_RATIO)
            if not np.isfinite(dynamic_sl):
                return fallback

            return max(dynamic_sl, max_loss_price)

        except Exception as e:
            logger.error(f"Error calculating dynamic stop loss for {ticker}: {e}")
            return fallback

    async def can_trade(self, ticker: str, order_type: str) -> tuple[bool, str]:
        """주문을 낼 수 있는 시점·상태인지 판단한다.

        금액·종목 수 한도는 여기서 보지 않는다. 수량이 정해지기 전에는 검사할 수
        없으므로 plan_buy 가 수량 산정과 함께 처리한다.
        """
        # 매도는 동시호가(CLOSING_AUCTION, 15:20~15:30)도 허용
        if order_type == "sell":
            status = get_trading_time_status()
            if status not in ("REGULAR", "CLOSING_AUCTION", "OPENING_AUCTION"):
                return False, f"매도 불가 시간 (status={status})"
            return True, "OK"

        if not is_trading_time():
            return False, "Not trading time."

        if self.risk_status == "RISK":
            # Issue #23: BEAR에서만 완전 차단, 나머지 체제는 포지션 축소 후 허용
            if self.market_condition == "BEAR":
                return False, "Market is BEAR + RISK — 매수 차단."
            logger.warning(
                f"[RISK+{self.market_condition}] 고변동성 — 포지션 사이징 RISK 배율"
                f"({self.position_size_multiplier.get('RISK', 0.4)}) 적용 후 허용"
            )

        # 일일 최대 손실 한도
        realized = self.daily_realized_pnl()
        if realized < 0:
            account = await self.api_client.get_account_summary()
            total_eval = account.get("total_evaluated_amount", 0) if account else 0
            if total_eval <= 0:
                return False, "잔고 조회 실패 — 일일 손실 한도 확인 불가"
            daily_loss_ratio = abs(realized) / total_eval
            if daily_loss_ratio >= self.max_daily_loss:
                return False, (f"일일 최대 손실 한도 도달 "
                               f"({daily_loss_ratio:.2%} >= {self.max_daily_loss:.2%})")

        return True, "OK"
