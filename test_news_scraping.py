"""뉴스 스크래핑 정상 동작 확인 — 읽기 전용 진단 스크립트."""
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from core.trader_api import AsyncKisAPI
from data.async_news_analyzer import AsyncNewsAnalyzer
import config


# 대표 샘플 종목 (대형주 3 + 최근 Overnight 3 + 최근 Intraday 2)
SAMPLES = [
    ("005930", "삼성전자"),
    ("000660", "SK하이닉스"),
    ("035420", "NAVER"),
    ("034020", "두산에너빌리티"),
    ("097230", "HJ중공업"),
    ("007660", "이수페타시스"),
    ("044380", "주연테크"),
    ("003720", "삼영"),
]


async def main():
    print("=" * 70)
    print("KIS 뉴스 API (FHKST01011800) 정상 동작 검증")
    print("=" * 70)
    print(f"모드: demo={config.DEMO_MODE}\n")

    api = AsyncKisAPI(
        app_key=config.APP_KEY,
        app_secret=config.APP_SECRET,
        account_number=config.ACCOUNT_NUMBER,
    )
    await api.init_session()
    analyzer = AsyncNewsAnalyzer(api_client=api)

    total_tickers = len(SAMPLES)
    ok_count = 0
    empty_count = 0
    negative_count = 0
    fail_count = 0

    try:
        for ticker, name in SAMPLES:
            print(f"▶ {ticker} ({name})")
            try:
                # 1. 직접 API 호출 — 원본 제목 리스트 확인
                titles = await api.get_news_titles(ticker, count=10)
                print(f"  API 응답: {len(titles)}건")
                for i, t in enumerate(titles[:3], 1):
                    # 너무 긴 제목은 잘라서 표시
                    short = t[:60] + ("..." if len(t) > 60 else "")
                    print(f"    {i}. {short}")
                if len(titles) > 3:
                    print(f"    ... 외 {len(titles)-3}건")

                # 2. 분석기 호출 — 부정 키워드 필터 결과
                result = await analyzer.analyze_stock_news(ticker, name)
                neg = result.get("has_negative", False)
                hits = result.get("negative_hits", [])
                score = result.get("score", 0.0)
                print(f"  분석: score={score:+.1f} 부정키워드={'YES' if neg else 'NO'}"
                      f"{' → ' + ','.join(hits) if hits else ''}")

                if titles:
                    ok_count += 1
                    if neg:
                        negative_count += 1
                else:
                    empty_count += 1
            except Exception as e:
                fail_count += 1
                print(f"  ✗ 오류: {e}")
            print()
    finally:
        await api.close()

    print("=" * 70)
    print(f"총 {total_tickers}종목 중:")
    print(f"  ✓ 정상 수집    : {ok_count}")
    print(f"  ⚠ 빈 응답      : {empty_count}")
    print(f"  ✗ API 오류     : {fail_count}")
    print(f"  🚨 부정 키워드  : {negative_count}")
    print("=" * 70)

    if fail_count > 0:
        print("\n❌ 일부 종목 뉴스 조회 실패 — 권한/네트워크 점검 필요")
        sys.exit(1)
    if ok_count == 0:
        print("\n❌ 뉴스가 한 건도 수집되지 않음 — KIS API 경로 확인 필요")
        sys.exit(1)
    print(f"\n✅ 뉴스 스크래핑 정상 동작 ({ok_count}/{total_tickers} 종목 수집)")


if __name__ == "__main__":
    asyncio.run(main())
