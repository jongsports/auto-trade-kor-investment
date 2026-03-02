import asyncio
import logging
from core.trader_api import AsyncKisAPI
from datetime import datetime, timedelta

async def test_ohlcv_tr_ids():
    logging.basicConfig(level=logging.DEBUG)
    api = AsyncKisAPI(
        app_key="", # Will load from env
        app_secret="", 
        account_number="", 
        demo_mode=True
    )
    # Manually load from .env since we are in a script
    import os
    from dotenv import load_dotenv
    load_dotenv()
    api.app_key = os.getenv("APP_KEY")
    api.app_secret = os.getenv("APP_SECRET")
    api.account_number = os.getenv("ACCOUNT_NUMBER")
    
    api.connect()
    await api.init_session()
    
    ticker = "086520" # EcoPro (KOSDAQ)
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = "20240101"
    
    # Try FHKST01010400 (Standard)
    print(f"\n--- Testing FHKST01010400 for {ticker} ---")
    res1 = await api._fetch("GET", "/uapi/domestic-stock/v1/quotations/inquire-daily-price", "FHKST01010400", params={
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": ticker,
        "FID_PERIOD_DIV_CODE": "D",
        "FID_ORG_ADJ_PRC": "0",
        "FID_INPUT_DATE_1": start_date,
        "FID_INPUT_DATE_2": end_date
    })
    print(f"FHKST01010400 Output for {ticker}: {res1}")

    # Try FHKST03010100 (Chart)
    print(f"\n--- Testing FHKST03010100 ---")
    res2 = await api._fetch("GET", "/uapi/domestic-stock/v1/quotations/inquire-daily-item-chartprice", "FHKST03010100", params={
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": ticker,
        "FID_PERIOD_DIV_CODE": "D",
        "FID_ORG_ADJ_PRC": "0"
    })
    print(f"FHKST03010100 Output: {res2}")
    
    await api.close()

if __name__ == "__main__":
    asyncio.run(test_ohlcv_tr_ids())
