import argparse
import asyncio
import logging
import signal
import sys

from core.async_trader import AsyncAutoTrader
import config

def parse_args():
    parser = argparse.ArgumentParser(description="KIS OpenAPI Async Trading Bot")
    parser.add_argument("--mode", type=str, default="run", choices=["run", "once"])
    parser.add_argument("--demo", action="store_true", help="Run via VTS Demo Server")
    return parser.parse_args()

async def run_until_signalled(trader: AsyncAutoTrader, logger: logging.Logger):
    """SIGTERM/SIGINT 를 받으면 trader.stop() 으로 정리하고 끝낸다.

    docker stop 과 재배포는 SIGTERM 을 보낸다. 핸들러가 없으면 PID 1 인 파이썬은
    신호를 무시하다가 SIGKILL 로 죽어, 진행 중이던 주문의 기록이 남지 않는다.
    """
    loop = asyncio.get_running_loop()
    stop_requested = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop_requested.set)
        except NotImplementedError:   # Windows
            pass

    run_task = asyncio.create_task(trader.start())
    stop_task = asyncio.create_task(stop_requested.wait())
    await asyncio.wait({run_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)

    if stop_task.done():
        logger.info("종료 신호 수신 — 정리 후 종료합니다.")
        await trader.stop()
    stop_task.cancel()
    await asyncio.gather(run_task, stop_task, return_exceptions=True)

    # 엔진이 예외로 죽었으면 로그를 남기고 0이 아닌 코드로 끝낸다. 조용히 0으로 끝나면
    # 컨테이너가 재시작 루프를 돌아도 배포 확인이 알아채지 못한다.
    if not run_task.cancelled() and run_task.exception() is not None:
        logger.critical("엔진이 예외로 종료됨", exc_info=run_task.exception())
        raise run_task.exception()

async def main_async():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("main")

    trader = AsyncAutoTrader(demo_mode=args.demo)

    if args.mode == "run":
        await run_until_signalled(trader, logger)
    elif args.mode == "once":
        logger.info("Once mode is executing setup and single screening...")
        trader.api_client.connect()
        await trader.api_client.init_session()
        await trader._run_screening()
        await trader.stop()

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_async())
