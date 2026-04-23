"""KIS 뉴스 API 원본 응답 확인 — rt_cd/msg1/output 구조 노출."""
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from core.trader_api import AsyncKisAPI
import config


async def main():
    print(f"모드: demo={config.DEMO_MODE}")
    print(f"Base URL: {'openapivts' if config.DEMO_MODE else 'openapi'}.koreainvestment.com\n")

    api = AsyncKisAPI(
        app_key=config.APP_KEY,
        app_secret=config.APP_SECRET,
        account_number=config.ACCOUNT_NUMBER,
    )
    await api.init_session()

    try:
        for ticker in ["005930", "034020"]:
            print(f"=== {ticker} 원본 응답 ===")
            res = await api._fetch(
                "GET",
                "/uapi/domestic-stock/v1/quotations/news-title",
                "FHKST01011800",
                params={
                    "FID_NEWS_OFER_ENTP_CODE": "",
                    "FID_INPUT_ISCD": ticker,
                    "FID_INPUT_DATE_1": "",
                    "FID_INPUT_TIME_1": "",
                    "FID_RANK_SORT_CLS_CODE": "0",
                },
            )
            print(f"rt_cd = {res.get('rt_cd')!r}")
            print(f"msg_cd = {res.get('msg_cd')!r}")
            print(f"msg1 = {res.get('msg1')!r}")
            print(f"message = {res.get('message')!r}")
            output = res.get("output")
            print(f"output type = {type(output).__name__}")
            print(f"output len = {len(output) if hasattr(output, '__len__') else 'N/A'}")
            if output:
                print("output[0] keys:", list(output[0].keys())[:10] if isinstance(output, list) and output else "empty")
                print("output[0] sample:", json.dumps(output[0], ensure_ascii=False, indent=2)[:500] if isinstance(output, list) and output else "")
            # 전체 응답 키도 확인
            print(f"all top keys: {list(res.keys())}\n")
    finally:
        await api.close()


if __name__ == "__main__":
    asyncio.run(main())
