"""
백테스팅 시뮬레이션 엔진.

스크리닝 점수 → 진입 → 청산의 전체 파이프라인을 과거 OHLCV 데이터로 시뮬레이션합니다.

설계 원칙:
- Look-ahead bias 방지: Day N 신호 → Day N+1 시가 진입
- 수수료: 매수/매도 각각 commission (기본 0.015%)
- 슬리피지: 진입 시 +slippage, 청산 시 -slippage (기본 0.01%)
- 수급 데이터는 0으로 처리 (과거 데이터 미지원)
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

import config
from strategy.async_screener import AsyncStockScreener

logger = logging.getLogger("backtest.engine")


@dataclass
class BacktestConfig:
    """백테스트 설정."""
    start_date: str = config.BACKTEST_START_DATE
    end_date: str = config.BACKTEST_END_DATE
    initial_capital: float = config.BACKTEST_INITIAL_CAPITAL
    commission: float = config.BACKTEST_COMMISSION      # 편도 수수료 (0.015%)
    slippage: float = config.BACKTEST_SLIPPAGE          # 슬리피지 (0.01%)
    score_threshold: int = 55                            # 진입 최소 점수
    max_positions: int = config.MAX_STOCKS               # 최대 동시 보유 종목
    take_profit: float = config.TAKE_PROFIT_RATIO        # 익절 비율 (5%)
    stop_loss: float = config.STOP_LOSS_RATIO            # 손절 비율 (2%)
    trailing_stop: float = config.TRAILING_STOP          # 트레일링 스탑 (3%)
    max_hold_days: int = config.MAX_HOLD_DAYS            # 최대 보유일
    position_size_ratio: float = config.MAX_STOCK_RATIO  # 종목당 자본 비중 (10%)


@dataclass
class BacktestTrade:
    """개별 거래 기록."""
    ticker: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    entry_score: int
    exit_reason: str = ""
    commission_paid: float = 0.0
    slippage_paid: float = 0.0

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        gross = (self.exit_price - self.entry_price) * self.quantity
        return gross - self.commission_paid - self.slippage_paid

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0 or self.exit_price is None:
            return 0.0
        cost = self.entry_price * self.quantity
        return self.pnl / cost if cost > 0 else 0.0

    @property
    def hold_days(self) -> int:
        if self.exit_date is None:
            return 0
        return (self.exit_date - self.entry_date).days


@dataclass
class Position:
    """열린 포지션 상태."""
    ticker: str
    entry_date: datetime
    entry_price: float
    quantity: int
    entry_score: int
    high_price: float  # 트레일링 스탑용 최고가
    hold_days: int = 0


class BacktestEngine:
    """
    백테스팅 시뮬레이션 엔진.

    사용법:
        cfg = BacktestConfig(start_date="2023-01-01", end_date="2023-12-31")
        engine = BacktestEngine(cfg, ohlcv_data)
        result = engine.run()
    """

    def __init__(
        self,
        config: BacktestConfig,
        ohlcv_data: Dict[str, pd.DataFrame],
    ):
        """
        Args:
            config: 백테스트 설정
            ohlcv_data: {ticker: OHLCV DataFrame} 딕셔너리
        """
        self.cfg = config
        self.ohlcv_data = ohlcv_data

        # 런타임 상태
        self.cash: float = config.initial_capital
        self.positions: Dict[str, Position] = {}  # ticker → Position
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Dict] = []  # 날짜별 자산 기록

        # 스크리닝 로직 재활용 (API 없이 지표 계산 전용)
        self._screener = _OfflineScreener()

        # 날짜 범위
        self.start_dt = pd.to_datetime(config.start_date)
        self.end_dt = pd.to_datetime(config.end_date)

    # ------------------------------------------------------------------ #
    # 메인 실행                                                             #
    # ------------------------------------------------------------------ #

    def run(self) -> Dict:
        """
        백테스트 전체 실행.

        Returns:
            dict: {trades, equity_curve, final_capital, metrics_input}
        """
        logger.info(
            f"백테스트 시작 | {self.cfg.start_date} ~ {self.cfg.end_date} "
            f"| 초기 자본 {self.cfg.initial_capital:,.0f}원 "
            f"| 종목 수 {len(self.ohlcv_data)}개"
        )

        # 전체 영업일 생성
        all_dates = pd.bdate_range(start=self.start_dt, end=self.end_dt)

        for current_date in all_dates:
            self._simulate_day(current_date)

        # 미청산 포지션 강제 청산 (마지막 날 종가)
        self._force_close_all(all_dates[-1])

        logger.info(
            f"백테스트 완료 | 총 거래: {len(self.trades)}건 "
            f"| 최종 자본: {self._total_equity(all_dates[-1]):,.0f}원"
        )

        return {
            "trades": self.trades,
            "equity_curve": pd.DataFrame(self.equity_curve),
            "final_capital": self.cash,
            "config": self.cfg,
        }

    # ------------------------------------------------------------------ #
    # 일별 시뮬레이션                                                        #
    # ------------------------------------------------------------------ #

    def _simulate_day(self, date: pd.Timestamp) -> None:
        """하루 시뮬레이션: 청산 체크 → 신규 진입 → 자산 기록."""
        # 1. 기존 포지션 청산 체크 (당일 시가/종가 기준)
        to_exit = []
        for ticker, pos in self.positions.items():
            row = self._get_row(ticker, date)
            if row is None:
                continue

            pos.hold_days += 1
            should_exit, reason = self._check_exit(pos, row)
            if should_exit:
                to_exit.append((ticker, row, reason))

        for ticker, row, reason in to_exit:
            self._execute_exit(ticker, row, reason, date)

        # 2. 신규 진입 체크 (전일 신호 → 당일 시가 진입)
        if len(self.positions) < self.cfg.max_positions:
            for ticker, df in self.ohlcv_data.items():
                if ticker in self.positions:
                    continue

                # 전일까지 데이터로 신호 계산 (look-ahead bias 방지)
                hist = self._get_history_until(ticker, date, lookback=130)
                if hist is None or len(hist) < 30:
                    continue

                score = self._calculate_score(ticker, hist)
                if score >= self.cfg.score_threshold:
                    row = self._get_row(ticker, date)
                    if row is not None:
                        self._execute_entry(ticker, row, score, date)

                if len(self.positions) >= self.cfg.max_positions:
                    break

        # 3. 당일 자산 기록
        total = self._total_equity(date)
        self.equity_curve.append({"date": date, "equity": total, "cash": self.cash})

    # ------------------------------------------------------------------ #
    # 진입 / 청산                                                           #
    # ------------------------------------------------------------------ #

    def _check_exit(self, pos: Position, row: pd.Series) -> Tuple[bool, str]:
        """청산 조건 검사."""
        current_price = float(row["close"])
        profit_ratio = current_price / pos.entry_price - 1

        # 최고가 갱신 (트레일링 스탑용)
        pos.high_price = max(pos.high_price, current_price)

        # 1. 익절
        if profit_ratio >= self.cfg.take_profit:
            return True, f"익절 {profit_ratio:.2%}"

        # 2. 손절
        if profit_ratio <= -self.cfg.stop_loss:
            return True, f"손절 {profit_ratio:.2%}"

        # 3. 트레일링 스탑 (최고가 대비 하락, 최고가가 진입가 이상일 때만)
        if pos.high_price > pos.entry_price:
            trailing = 1 - current_price / pos.high_price
            if trailing >= self.cfg.trailing_stop:
                return True, f"트레일링스탑 {trailing:.2%}"

        # 4. 최대 보유일 초과
        if pos.hold_days >= self.cfg.max_hold_days:
            return True, f"최대보유일({self.cfg.max_hold_days}일)"

        return False, "보유"

    def _execute_entry(
        self,
        ticker: str,
        row: pd.Series,
        score: int,
        date: pd.Timestamp,
    ) -> None:
        """매수 실행 (당일 시가 + 슬리피지)."""
        entry_price_raw = float(row["open"])
        entry_price = entry_price_raw * (1 + self.cfg.slippage)

        # 포지션 크기 계산
        alloc = self.cash * self.cfg.position_size_ratio
        quantity = int(alloc / entry_price)
        if quantity <= 0:
            return

        cost = entry_price * quantity
        commission = cost * self.cfg.commission
        slippage_cost = entry_price_raw * quantity * self.cfg.slippage
        total_cost = cost + commission

        if total_cost > self.cash:
            quantity = int(self.cash / (entry_price * (1 + self.cfg.commission)))
            if quantity <= 0:
                return
            cost = entry_price * quantity
            commission = cost * self.cfg.commission
            slippage_cost = entry_price_raw * quantity * self.cfg.slippage
            total_cost = cost + commission

        self.cash -= total_cost

        self.positions[ticker] = Position(
            ticker=ticker,
            entry_date=date.to_pydatetime(),
            entry_price=entry_price,
            quantity=quantity,
            entry_score=score,
            high_price=entry_price,
        )

        # 거래 기록 생성 (exit는 나중에 채움)
        trade = BacktestTrade(
            ticker=ticker,
            entry_date=date.to_pydatetime(),
            exit_date=None,
            entry_price=entry_price,
            exit_price=None,
            quantity=quantity,
            entry_score=score,
            commission_paid=commission,
            slippage_paid=slippage_cost,
        )
        self.trades.append(trade)

        logger.debug(
            f"[{date.date()}] 매수 {ticker} | "
            f"점수:{score} 가격:{entry_price:.0f} 수량:{quantity} "
            f"비용:{total_cost:,.0f}원"
        )

    def _execute_exit(
        self,
        ticker: str,
        row: pd.Series,
        reason: str,
        date: pd.Timestamp,
    ) -> None:
        """매도 실행 (당일 종가 - 슬리피지)."""
        pos = self.positions.pop(ticker, None)
        if pos is None:
            return

        exit_price_raw = float(row["close"])
        exit_price = exit_price_raw * (1 - self.cfg.slippage)

        proceeds = exit_price * pos.quantity
        commission = proceeds * self.cfg.commission
        slippage_cost = exit_price_raw * pos.quantity * self.cfg.slippage
        net_proceeds = proceeds - commission

        self.cash += net_proceeds

        # 해당 종목의 열린 거래 기록에 청산 정보 채우기
        for trade in reversed(self.trades):
            if trade.ticker == ticker and trade.exit_date is None:
                trade.exit_date = date.to_pydatetime()
                trade.exit_price = exit_price
                trade.exit_reason = reason
                trade.commission_paid += commission
                trade.slippage_paid += slippage_cost
                break

        logger.debug(
            f"[{date.date()}] 매도 {ticker} | "
            f"사유:{reason} 가격:{exit_price:.0f} "
            f"PnL:{(exit_price - pos.entry_price) * pos.quantity:+,.0f}원"
        )

    def _force_close_all(self, date: pd.Timestamp) -> None:
        """백테스트 종료 시 미청산 포지션 강제 청산."""
        tickers = list(self.positions.keys())
        for ticker in tickers:
            row = self._get_row(ticker, date)
            if row is not None:
                self._execute_exit(ticker, row, "백테스트종료", date)
            else:
                # 데이터 없으면 진입가로 청산
                pos = self.positions.pop(ticker)
                for trade in reversed(self.trades):
                    if trade.ticker == ticker and trade.exit_date is None:
                        trade.exit_date = date.to_pydatetime()
                        trade.exit_price = pos.entry_price
                        trade.exit_reason = "백테스트종료(데이터없음)"
                        break
                self.cash += pos.entry_price * pos.quantity

    # ------------------------------------------------------------------ #
    # 스코어링                                                              #
    # ------------------------------------------------------------------ #

    def _calculate_score(self, ticker: str, hist: pd.DataFrame) -> int:
        """
        기존 calculate_stock_score 로직을 오프라인으로 실행.
        수급 데이터는 0으로 처리 (과거 데이터 미지원).
        """
        # investor_trend를 0으로 설정 (과거 수급 데이터 없음)
        investor_trend = {"foreign_net_buy": 0, "institution_net_buy": 0}

        score_dict = self._screener.calculate_stock_score(
            ticker=ticker,
            ohlcv_data=hist,
            investor_trend=investor_trend,
            is_overnight_window=False,
            news_score=0.0,
        )
        return score_dict["total"]

    # ------------------------------------------------------------------ #
    # 유틸리티                                                              #
    # ------------------------------------------------------------------ #

    def _get_row(self, ticker: str, date: pd.Timestamp) -> Optional[pd.Series]:
        """특정 날짜의 OHLCV 행 반환."""
        df = self.ohlcv_data.get(ticker)
        if df is None or df.empty:
            return None
        mask = df["date"] == date
        if not mask.any():
            return None
        return df[mask].iloc[0]

    def _get_history_until(
        self, ticker: str, date: pd.Timestamp, lookback: int = 130
    ) -> Optional[pd.DataFrame]:
        """
        해당 날짜 이전 데이터만 반환 (look-ahead bias 방지).
        당일 데이터는 제외 (전일까지만).
        """
        df = self.ohlcv_data.get(ticker)
        if df is None or df.empty:
            return None
        hist = df[df["date"] < date].tail(lookback).reset_index(drop=True)
        return hist if len(hist) >= 30 else None

    def _total_equity(self, date: pd.Timestamp) -> float:
        """현재 총 자산 (현금 + 보유 포지션 평가액)."""
        equity = self.cash
        for ticker, pos in self.positions.items():
            row = self._get_row(ticker, date)
            price = float(row["close"]) if row is not None else pos.entry_price
            equity += price * pos.quantity
        return equity


# ------------------------------------------------------------------ #
# 오프라인 스크리너 (API 없는 순수 지표 계산용)                           #
# ------------------------------------------------------------------ #

class _OfflineScreener(AsyncStockScreener):
    """
    API 없이 지표 계산만 수행하는 AsyncStockScreener 서브클래스.
    백테스팅 엔진 내부 전용.
    """

    def __init__(self):
        # AsyncStockScreener.__init__은 api_client를 요구하므로 직접 설정
        self.api_client = None
        self.market_codes = {"KOSPI": "J", "KOSDAQ": "K"}
        self.candidate_stocks = []
        self.momentum_days = config.MOMENTUM_DAYS
        self.min_gap_up = config.MIN_GAP_UP
        self.min_volume_ratio = config.MIN_VOLUME_RATIO
        self.min_amount_ratio = config.MIN_AMOUNT_RATIO
        self.min_ma5_ratio = config.MIN_MA5_RATIO
        self.news_analyzer = None
