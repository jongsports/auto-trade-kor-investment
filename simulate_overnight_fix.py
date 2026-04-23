"""
진단 전용 — Overnight 전략 버그 수정 효과 시뮬레이션.

목적:
  check_exit_condition 의 시간비교 버그로 오버나이트 포지션이
  매수 직후 즉시 청산되고 있음. 이 스크립트는 "정상적으로 익일
  시가까지 보유했다면" 각 거래의 가상 PnL이 얼마였을지 계산한다.

입력: PostgreSQL trades 테이블의 Overnight SELL 레코드
출력: 실제 PnL vs 가상 PnL(익일시가 청산) 비교표

읽기 전용. 어떤 매매도 발생하지 않음.
"""
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

import asyncpg
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()
from core.trader_api import AsyncKisAPI
import config


async def fetch_overnight_sells():
    conn = await asyncpg.connect(
        host="192.168.45.53", port=5432,
        user="trader", password="trader2024",
        database="auto_trade_kor",
    )
    try:
        rows = await conn.fetch(
            """
            SELECT ticker, name, trade_date, buy_price, price AS sell_price,
                   pnl_ratio AS actual_pnl
            FROM trades
            WHERE strategy='Overnight' AND action='SELL'
              AND buy_price > 0
            ORDER BY trade_date
            """
        )
        return [dict(r) for r in rows]
    finally:
        await conn.close()


async def simulate(api: AsyncKisAPI, records: list) -> pd.DataFrame:
    out = []
    for rec in records:
        ticker = rec["ticker"]
        buy_date: datetime = rec["trade_date"]  # date object
        buy_price = int(rec["buy_price"])
        # 익일부터 +5 영업일 구간 OHLCV 조회
        start = buy_date.strftime("%Y%m%d")
        end = (buy_date + timedelta(days=7)).strftime("%Y%m%d")
        try:
            df = await api.get_ohlcv_by_range(ticker, start, end, "D")
        except Exception as e:
            print(f"  ! {ticker} {buy_date} OHLCV 조회 실패: {e}")
            continue
        if df is None or df.empty:
            print(f"  ! {ticker} {buy_date} 데이터 없음")
            continue
        df = df.sort_values("date").reset_index(drop=True)
        # buy_date 이후 첫 행이 익일
        next_rows = df[df["date"] > pd.Timestamp(buy_date)]
        if next_rows.empty:
            print(f"  ! {ticker} {buy_date} 익일 데이터 없음 (최신거래)")
            continue
        nd = next_rows.iloc[0]
        nd2 = next_rows.iloc[1] if len(next_rows) > 1 else None

        pnl_open_next = nd["open"] / buy_price - 1          # 익일 시가 청산
        pnl_close_next = nd["close"] / buy_price - 1        # 익일 종가 청산
        pnl_open_d2 = (nd2["open"] / buy_price - 1) if nd2 is not None else None

        out.append({
            "ticker": ticker,
            "buy_date": buy_date,
            "buy_price": buy_price,
            "actual_pnl_pct": float(rec["actual_pnl"]) * 100,
            "next_open_pct": pnl_open_next * 100,
            "next_close_pct": pnl_close_next * 100,
            "d2_open_pct": (pnl_open_d2 * 100) if pnl_open_d2 is not None else None,
        })
        print(f"  ✓ {ticker} {buy_date} buy={buy_price} nextOpen={nd['open']:.0f} "
              f"actual={float(rec['actual_pnl'])*100:+.2f}%  fixed={pnl_open_next*100:+.2f}%")
    return pd.DataFrame(out)


async def main():
    print("1) DB에서 Overnight SELL 기록 조회…")
    records = await fetch_overnight_sells()
    print(f"   → {len(records)}건\n")

    print("2) KIS API 초기화 (demo={})...".format(config.DEMO_MODE))
    api = AsyncKisAPI(
        app_key=config.APP_KEY,
        app_secret=config.APP_SECRET,
        account_number=config.ACCOUNT_NUMBER,
    )
    await api.init_session()
    try:
        print("3) 익일 OHLCV 수집 + 가상 PnL 계산…")
        df = await simulate(api, records)
    finally:
        await api.close()

    if df.empty:
        print("\n집계 가능한 데이터가 없습니다.")
        return

    print("\n" + "=" * 70)
    print("결과 요약 (단위: %)")
    print("=" * 70)
    print(df.to_string(index=False))
    print("\n--- 통계 ---")
    print(f"거래 수                 : {len(df)}")
    print(f"실제 평균 PnL           : {df['actual_pnl_pct'].mean():+.3f}%  "
          f"(합계 {df['actual_pnl_pct'].sum():+.2f}%)")
    print(f"익일 시가 청산 평균 PnL : {df['next_open_pct'].mean():+.3f}%  "
          f"(합계 {df['next_open_pct'].sum():+.2f}%)")
    print(f"익일 종가 청산 평균 PnL : {df['next_close_pct'].mean():+.3f}%  "
          f"(합계 {df['next_close_pct'].sum():+.2f}%)")
    if df["d2_open_pct"].notna().any():
        print(f"D+2 시가 청산 평균 PnL  : {df['d2_open_pct'].mean():+.3f}%  "
              f"(합계 {df['d2_open_pct'].sum():+.2f}%)")

    win_actual = (df["actual_pnl_pct"] > 0).sum()
    win_fix = (df["next_open_pct"] > 0).sum()
    print(f"\n승률 (실제)      : {win_actual}/{len(df)} = {win_actual/len(df)*100:.1f}%")
    print(f"승률 (수정안)    : {win_fix}/{len(df)} = {win_fix/len(df)*100:.1f}%")

    df.to_csv("overnight_fix_simulation.csv", index=False)
    print("\n저장: overnight_fix_simulation.csv")


if __name__ == "__main__":
    asyncio.run(main())
