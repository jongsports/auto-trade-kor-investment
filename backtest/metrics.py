"""
백테스트 성과 지표 계산 모듈.

계산 항목:
- 총 수익률, 연환산 수익률(CAGR)
- 최대 낙폭(MDD) 및 MDD 기간
- 샤프 비율 (무위험 수익률 2.5% 기준)
- 승률, 평균 수익, 평균 손실, 손익비(Profit Factor)
- 총 거래수, 평균 보유일
"""
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from backtest.engine import BacktestTrade

logger = logging.getLogger("backtest.metrics")

RISK_FREE_RATE_ANNUAL = 0.025  # 무위험 수익률 2.5% (국고채 기준)
TRADING_DAYS_PER_YEAR = 252


def calculate_metrics(
    trades: List[BacktestTrade],
    equity_curve: pd.DataFrame,
    initial_capital: float,
) -> Dict:
    """
    백테스트 성과 지표 종합 계산.

    Args:
        trades: 완결된 거래 목록 (exit_date가 None인 미청산 제외)
        equity_curve: date/equity 컬럼의 DataFrame
        initial_capital: 초기 자본금

    Returns:
        dict: 성과 지표 딕셔너리
    """
    closed = [t for t in trades if t.exit_date is not None and t.exit_price is not None]

    if not closed:
        logger.warning("완결된 거래가 없습니다. 빈 지표를 반환합니다.")
        return _empty_metrics(initial_capital)

    # ── 수익/손실 분류 ────────────────────────────────────────────────
    pnls = [t.pnl for t in closed]
    pnl_pcts = [t.pnl_pct for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_trades = len(closed)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total_trades if total_trades > 0 else 0.0

    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    avg_win_pct = np.mean([t.pnl_pct for t in closed if t.pnl > 0]) if wins else 0.0
    avg_loss_pct = np.mean([t.pnl_pct for t in closed if t.pnl <= 0]) if losses else 0.0

    # 손익비 (Profit Factor)
    total_win = sum(wins)
    total_loss = abs(sum(losses))
    profit_factor = total_win / total_loss if total_loss > 0 else float("inf")

    # 기대값 (Expected Value per trade)
    expected_value = np.mean(pnl_pcts)

    # ── 자산 곡선 기반 지표 ───────────────────────────────────────────
    if equity_curve.empty:
        total_return = 0.0
        cagr = 0.0
        mdd = 0.0
        mdd_duration_days = 0
        sharpe = 0.0
        daily_vol = 0.0
    else:
        eq = equity_curve["equity"].values
        dates = pd.to_datetime(equity_curve["date"])

        # 총 수익률
        final_equity = eq[-1]
        total_return = (final_equity - initial_capital) / initial_capital

        # CAGR
        n_days = (dates.iloc[-1] - dates.iloc[0]).days
        n_years = n_days / 365.25
        if n_years > 0 and final_equity > 0:
            cagr = (final_equity / initial_capital) ** (1 / n_years) - 1
        else:
            cagr = 0.0

        # MDD (최대 낙폭)
        mdd, mdd_duration_days = _calculate_mdd(eq, dates)

        # 샤프 비율
        daily_returns = pd.Series(eq).pct_change().dropna()
        sharpe, daily_vol = _calculate_sharpe(daily_returns)

    # ── 보유 기간 통계 ─────────────────────────────────────────────────
    hold_days_list = [t.hold_days for t in closed]
    avg_hold_days = np.mean(hold_days_list) if hold_days_list else 0.0

    # ── 비용 분석 ─────────────────────────────────────────────────────
    total_commission = sum(t.commission_paid for t in closed)
    total_slippage = sum(t.slippage_paid for t in closed)
    total_pnl = sum(pnls)

    metrics = {
        # 수익률
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "total_pnl": total_pnl,
        "final_equity": equity_curve["equity"].iloc[-1] if not equity_curve.empty else initial_capital,
        # 리스크
        "mdd_pct": mdd * 100,
        "mdd_duration_days": mdd_duration_days,
        "daily_volatility_pct": daily_vol * 100,
        "sharpe_ratio": sharpe,
        # 거래 통계
        "total_trades": total_trades,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate_pct": win_rate * 100,
        "avg_win_pct": avg_win_pct * 100,
        "avg_loss_pct": avg_loss_pct * 100,
        "avg_win_krw": avg_win,
        "avg_loss_krw": avg_loss,
        "profit_factor": profit_factor,
        "expected_value_pct": expected_value * 100,
        "avg_hold_days": avg_hold_days,
        # 비용
        "total_commission": total_commission,
        "total_slippage": total_slippage,
        # 요약
        "summary": _build_summary(
            total_return, cagr, mdd, sharpe, win_rate, profit_factor, total_trades
        ),
    }

    _log_metrics(metrics)
    return metrics


def _calculate_mdd(equity: np.ndarray, dates: pd.Series) -> tuple:
    """최대 낙폭(MDD) 및 MDD 지속 기간 계산."""
    peak = equity[0]
    peak_date = dates.iloc[0]
    mdd = 0.0
    mdd_duration = 0
    current_dd_start = dates.iloc[0]

    for i, (val, date) in enumerate(zip(equity, dates)):
        if val > peak:
            peak = val
            peak_date = date
            current_dd_start = date

        drawdown = (peak - val) / peak if peak > 0 else 0.0
        if drawdown > mdd:
            mdd = drawdown
            mdd_duration = (date - current_dd_start).days

    return mdd, mdd_duration


def _calculate_sharpe(daily_returns: pd.Series) -> tuple:
    """샤프 비율 계산 (연환산)."""
    if daily_returns.empty or daily_returns.std() == 0:
        return 0.0, 0.0

    daily_rf = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR
    excess = daily_returns - daily_rf
    sharpe = (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    daily_vol = daily_returns.std()
    return round(sharpe, 2), daily_vol


def _build_summary(
    total_return: float,
    cagr: float,
    mdd: float,
    sharpe: float,
    win_rate: float,
    profit_factor: float,
    total_trades: int,
) -> str:
    """성과 요약 문자열 생성."""
    return (
        f"총수익률 {total_return*100:+.1f}% | CAGR {cagr*100:+.1f}% | "
        f"MDD {mdd*100:.1f}% | 샤프 {sharpe:.2f} | "
        f"승률 {win_rate*100:.1f}% | PF {profit_factor:.2f} | "
        f"거래 {total_trades}건"
    )


def _empty_metrics(initial_capital: float) -> Dict:
    return {
        "total_return_pct": 0.0,
        "cagr_pct": 0.0,
        "total_pnl": 0.0,
        "final_equity": initial_capital,
        "mdd_pct": 0.0,
        "mdd_duration_days": 0,
        "daily_volatility_pct": 0.0,
        "sharpe_ratio": 0.0,
        "total_trades": 0,
        "win_count": 0,
        "loss_count": 0,
        "win_rate_pct": 0.0,
        "avg_win_pct": 0.0,
        "avg_loss_pct": 0.0,
        "avg_win_krw": 0.0,
        "avg_loss_krw": 0.0,
        "profit_factor": 0.0,
        "expected_value_pct": 0.0,
        "avg_hold_days": 0.0,
        "total_commission": 0.0,
        "total_slippage": 0.0,
        "summary": "거래 없음",
    }


def _log_metrics(m: Dict) -> None:
    logger.info("=" * 60)
    logger.info("[ 백테스트 성과 지표 ]")
    logger.info(f"  수익률 : 총 {m['total_return_pct']:+.2f}% | CAGR {m['cagr_pct']:+.2f}%")
    logger.info(f"  리스크  : MDD {m['mdd_pct']:.2f}% ({m['mdd_duration_days']}일) | 샤프 {m['sharpe_ratio']:.2f}")
    logger.info(f"  거래    : {m['total_trades']}건 | 승률 {m['win_rate_pct']:.1f}% | PF {m['profit_factor']:.2f}")
    logger.info(f"  평균    : 수익 {m['avg_win_pct']:+.2f}% | 손실 {m['avg_loss_pct']:+.2f}% | 보유 {m['avg_hold_days']:.1f}일")
    logger.info(f"  비용    : 수수료 {m['total_commission']:,.0f}원 | 슬리피지 {m['total_slippage']:,.0f}원")
    logger.info("=" * 60)
