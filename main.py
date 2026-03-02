import argparse
import asyncio
import logging
import sys

from core.async_trader import AsyncAutoTrader
import config

def parse_args():
    parser = argparse.ArgumentParser(description="KIS OpenAPI Async Trading Bot")
    parser.add_argument("--mode", type=str, default="run", choices=["run", "once"])
    parser.add_argument("--demo", action="store_true", help="Run via VTS Demo Server")
    return parser.parse_args()

async def main_async():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("main")
    
    trader = AsyncAutoTrader(demo_mode=args.demo)
    
    if args.mode == "run":
        try:
             await trader.start()
        except KeyboardInterrupt:
             logger.info("Gracefully shutting down...")
             await trader.stop()
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
