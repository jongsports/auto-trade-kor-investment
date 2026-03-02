"""
백테스팅 실행 진입점.

사용법:
    # 기본 실행 (config.yaml 설정 사용)
    python run_backtest.py

    # 날짜/자본 오버라이드
    python run_backtest.py --start 2023-01-01 --end 2023-12-31 --capital 100000000

    # 샘플 데이터로 API 없이 테스트
    python run_backtest.py --sample

    # 특정 종목만 테스트
    python run_backtest.py --tickers 005930 000660 035720

    # 점수 임계값 조정
    python run_backtest.py --threshold 60

KIS API 연결 시 실제 과거 데이터를 수집합니다.
캐시가 있으면 API 없이 재실행 가능합니다 (data/backtest_cache/).
"""
import argparse
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent))

import config
config.setup_logging()

from backtest.data_collector import BacktestDataCollector
from backtest.engine import BacktestConfig, BacktestEngine
from backtest.metrics import calculate_metrics
from backtest.reporter import BacktestReporter

logger = logging.getLogger("backtest.runner")


# ── 기본 테스트 종목 (KOSPI 대형주 + KOSDAQ 모멘텀) ──────────────────────
DEFAULT_TICKERS = [
    "005930",  # 삼성전자
    "000660",  # SK하이닉스
    "035720",  # 카카오
    "035420",  # NAVER
    "051910",  # LG화학
    "068270",  # 셀트리온
    "207940",  # 삼성바이오로직스
    "373220",  # LG에너지솔루션
    "000270",  # 기아
    "005380",  # 현대차
]


def parse_args():
    parser = argparse.ArgumentParser(description="한국 주식 자동매매 전략 백테스팅")
    parser.add_argument("--start", type=str, default=config.BACKTEST_START_DATE,
                        help="백테스트 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=config.BACKTEST_END_DATE,
                        help="백테스트 종료일 (YYYY-MM-DD)")
    parser.add_argument("--capital", type=int, default=config.BACKTEST_INITIAL_CAPITAL,
                        help="초기 자본금 (원)")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="테스트할 종목코드 목록 (예: 005930 000660)")
    parser.add_argument("--threshold", type=int, default=55,
                        help="진입 최소 스크리닝 점수 (기본 55)")
    parser.add_argument("--commission", type=float, default=config.BACKTEST_COMMISSION,
                        help="수수료율 (기본 0.00015 = 0.015%%)")
    parser.add_argument("--slippage", type=float, default=config.BACKTEST_SLIPPAGE,
                        help="슬리피지율 (기본 0.0001 = 0.01%%)")
    parser.add_argument("--sample", action="store_true",
                        help="KIS API 없이 합성 샘플 데이터로 테스트")
    parser.add_argument("--no-chart", action="store_true",
                        help="차트 생성 건너뜀")
    parser.add_argument("--refresh", action="store_true",
                        help="캐시 무시하고 API에서 데이터 재수집")
    parser.add_argument("--label", type=str, default="",
                        help="리포트 파일명 접두어 (기본: 타임스탬프)")
    return parser.parse_args()


def load_data_sample(tickers, start_date, end_date):
    """샘플 합성 데이터 생성."""
    logger.info("샘플 데이터 모드: 합성 OHLCV 생성 중...")
    collector = BacktestDataCollector(api_client=None)
    data = {}
    import numpy as np
    seed = 42
    for i, ticker in enumerate(tickers):
        df = BacktestDataCollector.generate_sample_data(
            ticker=ticker,
            start_date=(
                str(int(start_date[:4]) - 1) + start_date[4:]
            ),  # 1년 전부터 생성 (지표 계산 warm-up)
            end_date=end_date,
            initial_price=float(np.random.default_rng(seed + i).integers(10000, 200000)),
            seed=seed + i,
        )
        collector.save_sample_data(ticker, df)
        data[ticker] = collector.get_ohlcv(ticker, start_date, end_date)
        if not data[ticker].empty:
            logger.info(f"  {ticker}: {len(data[ticker])}행 생성")
        else:
            logger.warning(f"  {ticker}: 데이터 없음")
            del data[ticker]
    return data


def load_data_api(tickers, start_date, end_date, force_refresh):
    """KIS API에서 실제 데이터 수집."""
    logger.info("KIS API 모드: 실제 과거 데이터 수집 중...")

    if not config.APP_KEY or not config.APP_SECRET:
        logger.error(
            ".env 파일에 APP_KEY / APP_SECRET 미설정.\n"
            "  → '--sample' 옵션으로 샘플 데이터 테스트 가능"
        )
        sys.exit(1)

    from core.trader_api import AsyncKisAPI

    api = AsyncKisAPI(
        app_key=config.APP_KEY,
        app_secret=config.APP_SECRET,
        account_number=config.ACCOUNT_NUMBER,
        demo_mode=config.DEMO_MODE,
    )
    if not api.connect():
        logger.error("KIS API 연결 실패. 토큰을 확인하세요.")
        sys.exit(1)

    collector = BacktestDataCollector(api_client=api)
    data = collector.batch_collect(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        force_refresh=force_refresh,
    )
    return data


def main():
    args = parse_args()
    tickers = args.tickers or DEFAULT_TICKERS

    logger.info(f"백테스팅 시작: {args.start} ~ {args.end} | 종목 {len(tickers)}개")
    logger.info(f"  자본: {args.capital:,}원 | 임계점수: {args.threshold} | "
                f"수수료: {args.commission*100:.3f}% | 슬리피지: {args.slippage*100:.3f}%")

    # ── 1. 데이터 수집 ─────────────────────────────────────────────────
    if args.sample:
        ohlcv_data = load_data_sample(tickers, args.start, args.end)
    else:
        ohlcv_data = load_data_api(tickers, args.start, args.end, args.refresh)

    if not ohlcv_data:
        logger.error("유효한 OHLCV 데이터가 없습니다. 종료합니다.")
        sys.exit(1)

    logger.info(f"데이터 로드 완료: {len(ohlcv_data)}개 종목")

    # ── 2. 백테스트 설정 ────────────────────────────────────────────────
    cfg = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_capital=float(args.capital),
        commission=args.commission,
        slippage=args.slippage,
        score_threshold=args.threshold,
    )

    # ── 3. 엔진 실행 ────────────────────────────────────────────────────
    engine = BacktestEngine(cfg, ohlcv_data)
    result = engine.run()

    # ── 4. 성과 지표 계산 ───────────────────────────────────────────────
    metrics = calculate_metrics(
        trades=result["trades"],
        equity_curve=result["equity_curve"],
        initial_capital=cfg.initial_capital,
    )

    # ── 5. 리포트 생성 ──────────────────────────────────────────────────
    reporter = BacktestReporter(result, metrics, cfg, run_label=args.label)
    reporter.print_summary()

    if args.no_chart:
        paths = {
            "trades_csv": reporter.save_trades_csv(),
            "equity_csv": reporter.save_equity_csv(),
            "metrics_json": reporter.save_metrics_json(),
        }
    else:
        paths = reporter.save_all()

    print("\n저장된 파일:")
    for key, path in paths.items():
        print(f"  {key}: {path}")

    return metrics


if __name__ == "__main__":
    main()
