"""투자자 동향 API 원본 응답 — 실전 환경 (CLI --demo 미사용) 기준."""
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
    # 실전 모드로 명시적 호출 (main.py와 동일)
    api = AsyncKisAPI(
        app_key=config.APP_KEY,
        app_secret=config.APP_SECRET,
        account_number=config.ACCOUNT_NUMBER,
        demo_mode=False,
    )
    await api.init_session()
    print(f"base_url = {api.base_url}\n")

    try:
        for ticker in ["005930", "069500"]:   # 삼성전자, KODEX200
            print(f"=== {ticker} (Primary: HHPTJ04160200) ===")
            try:
                res = await api._fetch(
                    "GET",
                    "/uapi/domestic-stock/v1/quotations/investor-trend-estimate",
                    "HHPTJ04160200",
                    params={"MKSC_SHRN_ISCD": ticker},
                )
                print(f"rt_cd={res.get('rt_cd')!r} msg_cd={res.get('msg_cd')!r}")
                print(f"msg1={res.get('msg1')!r}")
                out2 = res.get("output2")
                print(f"output2 type={type(out2).__name__} len={len(out2) if hasattr(out2,'__len__') else '-'}")
                if out2:
                    print(f"output2[0] sample: {json.dumps(out2[0], ensure_ascii=False)[:300]}")
            except Exception as e:
                print(f"✗ 예외: {e}")

            print(f"\n=== {ticker} (Fallback: FHKST01010900) ===")
            try:
                res = await api._fetch(
                    "GET",
                    "/uapi/domestic-stock/v1/quotations/inquire-investor",
                    "FHKST01010900",
                    params={
                        "FID_COND_MRKT_DIV_CODE": "J",
                        "FID_INPUT_ISCD": ticker,
                    },
                )
                print(f"rt_cd={res.get('rt_cd')!r} msg_cd={res.get('msg_cd')!r}")
                print(f"msg1={res.get('msg1')!r}")
                out = res.get("output")
                print(f"output type={type(out).__name__} len={len(out) if hasattr(out,'__len__') else '-'}")
                if out:
                    print(f"output[0] sample: {json.dumps(out[0], ensure_ascii=False)[:300]}")
            except Exception as e:
                print(f"✗ 예외: {e}")
            print()
    finally:
        await api.close()


if __name__ == "__main__":
    asyncio.run(main())
