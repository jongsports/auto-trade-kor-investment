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
├── backtest/
│   ├── data_collector.py      # KIS API OHLCV 수집 + 캐시 (FHKST03010100)
│   ├── engine.py              # 백테스팅 시뮬레이션 엔진
│   ├── metrics.py             # 성과 지표 계산
│   └── reporter.py            # CSV + 차트 리포트 생성
├── config_dir/
│   └── config.yaml            # 전략 파라미터 설정
├── config.py                  # 설정 로더
├── main.py                    # 자동매매 진입점
├── run_backtest.py            # 백테스팅 진입점
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

### 3. 자동매매 실행
```bash
python main.py
```

---

## 백테스팅

### 프로젝트 구조

```
backtest/
├── __init__.py           # 패키지 진입점
├── data_collector.py     # KIS API OHLCV 수집 + 로컬 캐시 관리
├── engine.py             # 시뮬레이션 엔진 (스크리닝→진입→청산)
├── metrics.py            # 성과 지표 계산
└── reporter.py           # CSV + 차트 리포트 생성
run_backtest.py           # CLI 실행 스크립트
data/backtest_cache/      # 수집된 OHLCV 캐시 (자동 생성)
backtest_results/         # 결과 파일 저장 디렉토리 (자동 생성)
```

### 명령어

#### 기본 실행 (config.yaml 설정 그대로)
```bash
python run_backtest.py
```

#### 샘플 데이터로 테스트 (KIS API 없이)
```bash
python run_backtest.py --sample
```
> API 키 없이도 합성 OHLCV 데이터로 전체 파이프라인을 즉시 검증할 수 있습니다.

#### 날짜 범위 지정
```bash
python run_backtest.py --start 2023-01-01 --end 2023-12-31
```

#### 초기 자본 설정
```bash
python run_backtest.py --capital 50000000   # 5천만원
```

#### 특정 종목만 테스트
```bash
python run_backtest.py --tickers 005930 000660 035720
```

#### 진입 점수 임계값 조정
```bash
python run_backtest.py --threshold 60   # 기본값 55
```

#### 수수료 / 슬리피지 조정
```bash
python run_backtest.py --commission 0.00015 --slippage 0.0001
# 수수료 기본값: 0.015% (KIS 기준)
# 슬리피지 기본값: 0.01%
```

#### 데이터 캐시 강제 갱신
```bash
python run_backtest.py --refresh
# 기존 캐시를 무시하고 KIS API에서 데이터를 재수집합니다.
```

#### 차트 없이 CSV/JSON만 저장
```bash
python run_backtest.py --no-chart
```

#### 리포트 파일명 접두어 지정
```bash
python run_backtest.py --label v1_baseline
# 출력: trades_v1_baseline.csv, report_v1_baseline.png ...
```

#### 옵션 조합 예시
```bash
# 2022~2023년 삼성전자·하이닉스 샘플 테스트, 점수 60점 기준
python run_backtest.py --sample --start 2022-01-01 --end 2023-12-31 \
  --tickers 005930 000660 --threshold 60 --label test_2y

# 실전 API로 전체 종목 2024년 백테스트 (캐시 갱신)
python run_backtest.py --start 2024-01-01 --end 2024-12-31 \
  --refresh --label production_2024
```

### 전체 옵션 목록

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--start` | config.yaml | 백테스트 시작일 (YYYY-MM-DD) |
| `--end` | config.yaml | 백테스트 종료일 (YYYY-MM-DD) |
| `--capital` | 100,000,000 | 초기 자본금 (원) |
| `--tickers` | 기본 10개 종목 | 테스트 종목코드 목록 (공백 구분) |
| `--threshold` | 55 | 진입 최소 스크리닝 점수 |
| `--commission` | 0.00015 | 수수료율 (편도, 0.015%) |
| `--slippage` | 0.0001 | 슬리피지율 (0.01%) |
| `--sample` | False | 합성 데이터 사용 (API 불필요) |
| `--refresh` | False | 캐시 무시, API 재수집 |
| `--no-chart` | False | 차트 생성 건너뜀 |
| `--label` | 타임스탬프 | 리포트 파일명 접두어 |

### 결과 파일

백테스트 완료 후 `backtest_results/` 에 저장됩니다.

| 파일 | 내용 |
|------|------|
| `trades_{label}.csv` | 거래 내역 전체 (진입일/청산일/수익률/사유) |
| `equity_{label}.csv` | 날짜별 자산 곡선 |
| `metrics_{label}.json` | 성과 지표 JSON (승률/MDD/샤프/CAGR 등) |
| `report_{label}.png` | 자산곡선 · 낙폭 · 수익분포 · 누적PnL 4분할 차트 |

### 성과 지표 항목

| 지표 | 설명 |
|------|------|
| `total_return_pct` | 총 수익률 (%) |
| `cagr_pct` | 연환산 수익률 CAGR (%) |
| `mdd_pct` | 최대 낙폭 MDD (%) |
| `mdd_duration_days` | MDD 지속 기간 (일) |
| `sharpe_ratio` | 샤프 비율 (무위험수익률 2.5% 기준) |
| `win_rate_pct` | 승률 (%) |
| `profit_factor` | 손익비 (총수익 / 총손실) |
| `avg_win_pct` | 평균 수익 거래 수익률 (%) |
| `avg_loss_pct` | 평균 손실 거래 수익률 (%) |
| `avg_hold_days` | 평균 보유일 |
| `total_commission` | 수수료 합계 (원) |
| `total_slippage` | 슬리피지 합계 (원) |

### 데이터 수집 방식

- **KIS API**: TR `FHKST03010100` (`inquire-daily-itemchartprice`) 사용
  - 날짜 범위 지정으로 장기 과거 데이터 수집 가능
  - 1회 호출 100건 제한 → 내부에서 100일 청크로 자동 분할
- **캐시**: `data/backtest_cache/{ticker}.csv` 로컬 저장
  - 한 번 수집 후 재실행 시 API 호출 없이 즉시 로드
  - `--refresh` 옵션으로 강제 갱신

### 설계 원칙

- **Look-ahead bias 방지**: Day N 신호 → Day N+1 시가 진입
- **수급 데이터**: 과거 수급 데이터 미지원으로 0 처리 (기술적+거래량 점수만 사용)
- **진입가**: 다음날 시가 × (1 + slippage)
- **청산가**: 당일 종가 × (1 - slippage)
- **청산 우선순위**: 익절 → 손절 → 트레일링스탑 → 최대보유일

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
