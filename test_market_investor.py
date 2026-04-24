"""시장별 투자자매매동향(시세) API 동작 검증 — FHPTJ04030000."""
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
    api = AsyncKisAPI(
        app_key=config.APP_KEY,
        app_secret=config.APP_SECRET,
        account_number=config.ACCOUNT_NUMBER,
        demo_mode=False,   # 실전 (postman 샘플 기준)
    )
    await api.init_session()
    print(f"base_url = {api.base_url}\n")

    markets = [
        ("코스피 종합", "KSP", "0001"),
        ("코스닥 종합", "KSQ", "1001"),
    ]

    try:
        for name, iscd, iscd2 in markets:
            print(f"=== {name} (ISCD={iscd} ISCD_2={iscd2}) ===")
            try:
                res = await api._fetch(
                    "GET",
                    "/uapi/domestic-stock/v1/quotations/inquire-investor-time-by-market",
                    "FHPTJ04030000",
                    params={
                        "FID_INPUT_ISCD": iscd,
                        "FID_INPUT_ISCD_2": iscd2,
                    },
                )
                print(f"rt_cd={res.get('rt_cd')!r} msg_cd={res.get('msg_cd')!r}")
                print(f"msg1={res.get('msg1')!r}")

                # output / output1 / output2 모두 확인
                for key in ("output", "output1", "output2"):
                    val = res.get(key)
                    if val is not None:
                        if isinstance(val, list):
                            print(f"{key}: list, len={len(val)}")
                            if val:
                                print(f"  [0] keys: {list(val[0].keys())[:15]}")
                                print(f"  [0] sample: {json.dumps(val[0], ensure_ascii=False)[:400]}")
                        elif isinstance(val, dict):
                            print(f"{key}: dict, keys={list(val.keys())[:15]}")
                            print(f"  sample: {json.dumps(val, ensure_ascii=False)[:400]}")
                print()
            except Exception as e:
                print(f"✗ 예외: {e}\n")
    finally:
        await api.close()


if __name__ == "__main__":
    asyncio.run(main())
