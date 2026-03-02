"""
백테스트 리포트 생성기.

출력:
- trades_{timestamp}.csv  : 거래 내역 전체
- equity_{timestamp}.csv  : 날짜별 자산 곡선
- metrics_{timestamp}.json: 성과 지표 JSON
- report_{timestamp}.png  : 자산 곡선 + 낙폭 차트 (4분할)
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from backtest.engine import BacktestConfig, BacktestTrade

logger = logging.getLogger("backtest.reporter")

REPORT_DIR = Path("backtest_results")


class BacktestReporter:
    """
    백테스트 결과 저장 및 시각화.

    사용법:
        reporter = BacktestReporter(result, metrics, config)
        reporter.save_all()
    """

    def __init__(
        self,
        result: Dict,
        metrics: Dict,
        cfg: BacktestConfig,
        run_label: str = "",
    ):
        self.trades: List[BacktestTrade] = result["trades"]
        self.equity_curve: pd.DataFrame = result["equity_curve"]
        self.metrics = metrics
        self.cfg = cfg
        self.label = run_label or datetime.now().strftime("%Y%m%d_%H%M%S")
        REPORT_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ #
    # 통합 저장                                                             #
    # ------------------------------------------------------------------ #

    def save_all(self) -> Dict[str, Path]:
        """모든 리포트 파일 저장. 저장된 경로 딕셔너리 반환."""
        paths = {}
        paths["trades_csv"] = self.save_trades_csv()
        paths["equity_csv"] = self.save_equity_csv()
        paths["metrics_json"] = self.save_metrics_json()

        try:
            paths["chart_png"] = self.save_chart()
        except ImportError:
            logger.warning("matplotlib 미설치 — 차트 생성 건너뜀 (pip install matplotlib)")
        except Exception as e:
            logger.warning(f"차트 생성 실패: {e}")

        logger.info("리포트 저장 완료:")
        for key, path in paths.items():
            logger.info(f"  {key}: {path}")
        return paths

    # ------------------------------------------------------------------ #
    # CSV                                                                  #
    # ------------------------------------------------------------------ #

    def save_trades_csv(self) -> Path:
        """거래 내역 CSV 저장."""
        closed = [t for t in self.trades if t.exit_date is not None]
        if not closed:
            logger.warning("완결된 거래 없음 — trades CSV 빈 파일 저장")

        rows = []
        for t in closed:
            rows.append(
                {
                    "ticker": t.ticker,
                    "entry_date": t.entry_date.strftime("%Y-%m-%d") if t.entry_date else "",
                    "exit_date": t.exit_date.strftime("%Y-%m-%d") if t.exit_date else "",
                    "entry_price": round(t.entry_price, 0),
                    "exit_price": round(t.exit_price, 0) if t.exit_price else 0,
                    "quantity": t.quantity,
                    "pnl_krw": round(t.pnl, 0),
                    "pnl_pct": round(t.pnl_pct * 100, 2),
                    "hold_days": t.hold_days,
                    "exit_reason": t.exit_reason,
                    "entry_score": t.entry_score,
                    "commission": round(t.commission_paid, 0),
                    "slippage": round(t.slippage_paid, 0),
                }
            )

        df = pd.DataFrame(rows)
        path = REPORT_DIR / f"trades_{self.label}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"거래 내역 저장: {path} ({len(df)}건)")
        return path

    def save_equity_csv(self) -> Path:
        """자산 곡선 CSV 저장."""
        path = REPORT_DIR / f"equity_{self.label}.csv"
        if not self.equity_curve.empty:
            self.equity_curve.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"자산 곡선 저장: {path}")
        return path

    # ------------------------------------------------------------------ #
    # JSON                                                                 #
    # ------------------------------------------------------------------ #

    def save_metrics_json(self) -> Path:
        """성과 지표 JSON 저장."""
        path = REPORT_DIR / f"metrics_{self.label}.json"

        output = {
            "run_label": self.label,
            "config": {
                "start_date": self.cfg.start_date,
                "end_date": self.cfg.end_date,
                "initial_capital": self.cfg.initial_capital,
                "commission": self.cfg.commission,
                "slippage": self.cfg.slippage,
                "score_threshold": self.cfg.score_threshold,
                "max_positions": self.cfg.max_positions,
                "take_profit": self.cfg.take_profit,
                "stop_loss": self.cfg.stop_loss,
                "trailing_stop": self.cfg.trailing_stop,
                "max_hold_days": self.cfg.max_hold_days,
            },
            "metrics": {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.metrics.items()
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"성과 지표 저장: {path}")
        return path

    # ------------------------------------------------------------------ #
    # 차트                                                                 #
    # ------------------------------------------------------------------ #

    def save_chart(self) -> Path:
        """
        4분할 차트 저장.
        1. 자산 곡선 (초기 자본 대비)
        2. 낙폭 (Drawdown)
        3. 거래별 수익률 분포 (히스토그램)
        4. 누적 PnL
        """
        import matplotlib
        matplotlib.use("Agg")  # GUI 없는 환경 대응
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib import rcParams

        # 한국어 폰트 설정 (macOS / Linux / Windows 순 시도)
        _korean_fonts = ["AppleGothic", "Malgun Gothic", "NanumGothic", "DejaVu Sans"]
        from matplotlib.font_manager import findfont, FontProperties
        for _f in _korean_fonts:
            if findfont(FontProperties(family=_f)) != findfont(FontProperties()):
                rcParams["font.family"] = _f
                break
        rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"백테스트 결과 | {self.cfg.start_date} ~ {self.cfg.end_date}\n"
            f"{self.metrics.get('summary', '')}",
            fontsize=11,
        )

        eq = self.equity_curve
        if not eq.empty:
            dates = pd.to_datetime(eq["date"])
            equity = eq["equity"].values
            initial = self.cfg.initial_capital

            # ── 1. 자산 곡선 ─────────────────────────────────────────────
            ax1 = axes[0, 0]
            ax1.plot(dates, equity / initial * 100 - 100, color="#2196F3", linewidth=1.5)
            ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax1.set_title("자산 곡선 (초기자본 대비 %)", fontsize=10)
            ax1.set_ylabel("수익률 (%)")
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%y.%m"))
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax1.tick_params(axis="x", rotation=30)
            ax1.fill_between(
                dates,
                equity / initial * 100 - 100,
                0,
                where=equity >= initial,
                alpha=0.2,
                color="#4CAF50",
                label="수익 구간",
            )
            ax1.fill_between(
                dates,
                equity / initial * 100 - 100,
                0,
                where=equity < initial,
                alpha=0.2,
                color="#F44336",
                label="손실 구간",
            )
            ax1.legend(fontsize=8)

            # ── 2. 낙폭(Drawdown) ────────────────────────────────────────
            ax2 = axes[0, 1]
            peak = pd.Series(equity).cummax()
            drawdown = (equity - peak) / peak * 100
            ax2.fill_between(dates, drawdown, 0, color="#F44336", alpha=0.6)
            ax2.set_title(f"낙폭 (MDD: {self.metrics.get('mdd_pct', 0):.1f}%)", fontsize=10)
            ax2.set_ylabel("낙폭 (%)")
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%y.%m"))
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax2.tick_params(axis="x", rotation=30)

        else:
            axes[0, 0].text(0.5, 0.5, "데이터 없음", ha="center", va="center")
            axes[0, 1].text(0.5, 0.5, "데이터 없음", ha="center", va="center")

        # ── 3. 거래별 수익률 히스토그램 ─────────────────────────────────
        ax3 = axes[1, 0]
        closed = [t for t in self.trades if t.exit_date is not None]
        if closed:
            pnl_pcts = [t.pnl_pct * 100 for t in closed]
            colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnl_pcts]
            ax3.hist(pnl_pcts, bins=30, color="#90CAF9", edgecolor="white", alpha=0.8)
            ax3.axvline(0, color="black", linewidth=1)
            ax3.axvline(
                sum(pnl_pcts) / len(pnl_pcts),
                color="orange",
                linewidth=1.5,
                linestyle="--",
                label=f"평균 {sum(pnl_pcts)/len(pnl_pcts):.2f}%",
            )
            ax3.set_title(
                f"거래별 수익률 분포 (승률 {self.metrics.get('win_rate_pct', 0):.1f}%)",
                fontsize=10,
            )
            ax3.set_xlabel("수익률 (%)")
            ax3.set_ylabel("빈도")
            ax3.legend(fontsize=8)

        # ── 4. 누적 PnL ──────────────────────────────────────────────────
        ax4 = axes[1, 1]
        if closed:
            sorted_trades = sorted(closed, key=lambda t: t.exit_date)
            cumulative = []
            total = 0
            exit_dates = []
            for t in sorted_trades:
                total += t.pnl
                cumulative.append(total)
                exit_dates.append(t.exit_date)

            ax4.plot(exit_dates, cumulative, color="#9C27B0", linewidth=1.5)
            ax4.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax4.fill_between(
                exit_dates,
                cumulative,
                0,
                where=[c >= 0 for c in cumulative],
                alpha=0.2,
                color="#4CAF50",
            )
            ax4.fill_between(
                exit_dates,
                cumulative,
                0,
                where=[c < 0 for c in cumulative],
                alpha=0.2,
                color="#F44336",
            )
            ax4.set_title(f"누적 PnL (총 {sum(t.pnl for t in closed):+,.0f}원)", fontsize=10)
            ax4.set_ylabel("누적 손익 (원)")
            ax4.xaxis.set_major_formatter(mdates.DateFormatter("%y.%m"))
            ax4.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        path = REPORT_DIR / f"report_{self.label}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"차트 저장: {path}")
        return path

    # ------------------------------------------------------------------ #
    # 콘솔 출력                                                             #
    # ------------------------------------------------------------------ #

    def print_summary(self) -> None:
        """성과 요약을 콘솔에 출력."""
        m = self.metrics
        print("\n" + "=" * 65)
        print(f"  백테스트 결과 | {self.cfg.start_date} ~ {self.cfg.end_date}")
        print("=" * 65)
        print(f"  초기 자본    : {self.cfg.initial_capital:>15,.0f} 원")
        print(f"  최종 자본    : {m.get('final_equity', 0):>15,.0f} 원")
        print(f"  총 수익      : {m.get('total_pnl', 0):>+15,.0f} 원")
        print("-" * 65)
        print(f"  총 수익률    : {m.get('total_return_pct', 0):>+14.2f} %")
        print(f"  CAGR         : {m.get('cagr_pct', 0):>+14.2f} %")
        print(f"  MDD          : {m.get('mdd_pct', 0):>14.2f} % ({m.get('mdd_duration_days', 0)}일)")
        print(f"  샤프 비율    : {m.get('sharpe_ratio', 0):>14.2f}")
        print("-" * 65)
        print(f"  총 거래수    : {m.get('total_trades', 0):>14} 건")
        print(f"  승률         : {m.get('win_rate_pct', 0):>14.1f} %  ({m.get('win_count',0)}승 {m.get('loss_count',0)}패)")
        print(f"  손익비(PF)   : {m.get('profit_factor', 0):>14.2f}")
        print(f"  평균 수익    : {m.get('avg_win_pct', 0):>+14.2f} %")
        print(f"  평균 손실    : {m.get('avg_loss_pct', 0):>+14.2f} %")
        print(f"  평균 보유일  : {m.get('avg_hold_days', 0):>14.1f} 일")
        print("-" * 65)
        print(f"  수수료 합계  : {m.get('total_commission', 0):>15,.0f} 원")
        print(f"  슬리피지 합계: {m.get('total_slippage', 0):>15,.0f} 원")
        print("=" * 65 + "\n")
