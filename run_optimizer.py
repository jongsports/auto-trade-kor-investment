"""
베이지안 파라미터 최적화 실행 진입점.

사용법:
    # 샘플 데이터로 NORMAL 체제 최적화 (5회 탐색)
    python run_optimizer.py --sample --regime NORMAL --trials 5

    # 샘플 데이터로 BULL 체제 50회 탐색 후 config.yaml 저장
    python run_optimizer.py --sample --regime BULL --trials 50 --save

    # KIS API 실데이터로 BEAR 체제 100회 탐색
    python run_optimizer.py --regime BEAR --trials 100
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
config.setup_logging()

logger = logging.getLogger("optimizer.runner")

# 기본 테스트 종목 (run_backtest.py와 동일)
DEFAULT_TICKERS = [
    "005930", "000660", "035720", "035420", "051910",
    "068270", "207940", "373220", "000270", "005380",
]


def parse_args():
    parser = argparse.ArgumentParser(description="베이지안 파라미터 최적화 (optuna)")
    parser.add_argument("--regime", type=str, default=None,
                        choices=["BULL", "NORMAL", "BEAR", "VOLATILE",
                                 "VOLATILE_UP", "VOLATILE_DOWN"],
                        help="최적화할 시장 체제 (기본: None = 체제 무관)")
    parser.add_argument("--trials", type=int, default=50,
                        help="optuna 탐색 횟수 (기본 50)")
    parser.add_argument("--sample", action="store_true",
                        help="KIS API 없이 합성 샘플 데이터 사용")
    parser.add_argument("--save", action="store_true",
                        help="최적 파라미터를 config.yaml에 저장")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="테스트 종목코드 목록")
    parser.add_argument("--start", type=str, default=config.BACKTEST_START_DATE,
                        help="백테스트 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=config.BACKTEST_END_DATE,
                        help="백테스트 종료일 (YYYY-MM-DD)")
    return parser.parse_args()


def load_data(args):
    """데이터 로드 (샘플 또는 KIS API)."""
    tickers = args.tickers or DEFAULT_TICKERS

    if args.sample:
        logger.info("샘플 데이터 모드로 데이터 생성 중...")
        import numpy as np
        from backtest.data_collector import BacktestDataCollector

        collector = BacktestDataCollector(api_client=None, sample=True)
        data = {}
        seed = 42
        for i, ticker in enumerate(tickers):
            df = BacktestDataCollector.generate_sample_data(
                ticker=ticker,
                start_date=str(int(args.start[:4]) - 1) + args.start[4:],
                end_date=args.end,
                initial_price=float(np.random.default_rng(seed + i).integers(10000, 200000)),
                seed=seed + i,
            )
            collector.save_sample_data(ticker, df)
            ohlcv = collector.get_ohlcv(ticker, args.start, args.end)
            if not ohlcv.empty:
                data[ticker] = ohlcv
        logger.info(f"샘플 데이터 {len(data)}종목 생성 완료")
        return data

    # KIS API 모드
    if not config.APP_KEY or not config.APP_SECRET:
        logger.error(".env 파일에 APP_KEY / APP_SECRET 미설정. --sample 옵션 사용.")
        sys.exit(1)

    from core.trader_api import AsyncKisAPI
    from backtest.data_collector import BacktestDataCollector

    api = AsyncKisAPI(
        app_key=config.APP_KEY,
        app_secret=config.APP_SECRET,
        account_number=config.ACCOUNT_NUMBER,
        demo_mode=config.DEMO_MODE,
    )
    if not api.connect():
        logger.error("KIS API 연결 실패.")
        sys.exit(1)

    collector = BacktestDataCollector(api_client=api)
    data = collector.batch_collect(tickers=tickers, start_date=args.start, end_date=args.end)
    return data


def main():
    args = parse_args()

    logger.info(
        f"최적화 시작 | regime={args.regime or '전체'} "
        f"| trials={args.trials} | sample={args.sample}"
    )

    # 데이터 로드
    ohlcv_data = load_data(args)
    if not ohlcv_data:
        logger.error("유효한 데이터 없음. 종료합니다.")
        sys.exit(1)

    # 최적화 실행
    from backtest.optimizer import RegimeOptimizer

    optimizer = RegimeOptimizer(
        ohlcv_data=ohlcv_data,
        regime=args.regime,
        n_trials=args.trials,
    )
    best_params = optimizer.optimize()

    # config.yaml 저장 (--save 옵션)
    if args.save:
        optimizer.save_best_params()
    else:
        print("\n(--save 옵션 없음: config.yaml 업데이트 건너뜀)")

    return best_params


if __name__ == "__main__":
    main()
