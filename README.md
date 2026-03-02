# 한국 주식 자동매매 시스템

한국투자증권(KIS) OpenAPI를 활용한 비동기 기반 국내 주식 자동매매 시스템입니다.

## 주요 기능

- **비동기 아키텍처**: `asyncio` 기반 고성능 병렬 처리
- **종목 스크리닝**: 기술적 지표 + 수급 + 뉴스 감성 분석 100점 시스템
- **리스크 관리**: 포지션 크기 제어, 손절/익절, 일일 손실 한도
- **알림**: Telegram / Email 실시간 알림

## 프로젝트 구조

```
.
├── core/
│   ├── trader_api.py          # KIS OpenAPI 비동기 래퍼 (AsyncKisAPI)
│   └── async_trader.py        # 전체 오케스트레이션 (AsyncAutoTrader)
├── strategy/
│   ├── async_screener.py      # 종목 스크리닝 (AsyncStockScreener)
│   └── async_trading_strategy.py  # 진입/청산 전략 (AsyncTradingStrategy)
├── risk/
│   └── async_risk_manager.py  # 리스크 관리 (AsyncRiskManager)
├── data/
│   └── async_news_analyzer.py # 뉴스 감성 분석 (AsyncNewsAnalyzer)
├── utils/
│   ├── notifier.py            # 알림 모듈
│   └── utils.py               # 공통 유틸리티
├── config_dir/
│   └── config.yaml            # 전략 파라미터 설정
├── config.py                  # 설정 로더
├── main.py                    # 진입점
└── requirements.txt
```

## 스크리닝 점수 시스템 (100점)

| 항목 | 배점 | 세부 항목 |
|------|------|-----------|
| 기술적 지표 | 40점 | MACD+Hist(12) + Stochastic(10) + ADX(8) + OBV(10) |
| 거래량 | 20점 | 급증(10) + MA20 위(5) + ATR 범위(5) |
| 수급 | 30점 | 외국인(15) + 기관(15) |
| 뉴스 감성 | 10점 | AsyncNewsAnalyzer |
| 오버나이트 보너스 | +15~20점 | 조건 충족 시 가산 |

### 세션별 임계값
- 09~10시: 60점
- 10~14시: 55점
- 14~15:10: 50점
- 오버나이트: 45점

## 설치 및 실행

### 1. 환경 설정
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 인증 정보 설정
```bash
cp .env.example .env
# .env 파일에 KIS API 키 입력
```

### 3. 실행
```bash
python main.py
```

## 환경변수 (.env)

```
APP_KEY=한국투자증권_APP_KEY
APP_SECRET=한국투자증권_APP_SECRET
ACCOUNT_NUMBER=계좌번호
TELEGRAM_TOKEN=텔레그램_봇_토큰
TELEGRAM_CHAT_ID=텔레그램_채널_ID
```

> ⚠️ `.env` 파일은 절대 커밋하지 마세요. `.gitignore`에 포함되어 있습니다.

## API 설정

| 구분 | 엔드포인트 | Rate Limit |
|------|-----------|------------|
| 모의투자 | `openapivts.koreainvestment.com:29443` | Semaphore(1), 1.1s 지연 |
| 실전투자 | `openapi.koreainvestment.com:9443` | Semaphore(10) |

## 라이선스

MIT
