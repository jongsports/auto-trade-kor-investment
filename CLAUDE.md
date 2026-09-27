# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

한국투자증권(KIS) OpenAPI 기반 국내 주식 비동기 자동매매 봇. 실제 자금이 오가는 시스템이므로 `main.py`(실주문 경로)는 절대 자동 실행하지 말 것. 검증은 단위 테스트와 `run_backtest.py --sample` 로만.

## 실행 환경

- **반드시 `venv/bin/python`을 쓸 것.** venv는 Python 3.12, 시스템 `python3`는 3.14라 의존성이 다르다.
- 의존성은 `requirements.txt` (pandas / numpy / aiohttp / asyncpg / optuna / matplotlib). pytest 는 없다 — 테스트는 표준 라이브러리 `unittest`.
- **로컬에서 KIS API 를 호출하지 말 것.** 서버와 같은 APP_KEY 라 토큰을 재발급하면 서버 토큰과 충돌할 수 있다. 실제 응답을 확인해야 하면 서버 컨테이너 안에서 기존 토큰으로 조회 TR 만 호출한다.

## 자주 쓰는 명령

```bash
# 전체 테스트 (네트워크 호출 없음, 약 7초)
venv/bin/python -m unittest discover -s tests -t .

# 테스트 한 파일 / 한 케이스
venv/bin/python -m unittest tests.test_exit_rules
venv/bin/python -m unittest tests.test_exit_rules.OvernightExitTest.test_d1_take_profit

# 전체 문법 검사 (CI 와 동일)
venv/bin/python -m compileall -q . -x 'venv|open-trading-api'

# 백테스트 — 합성 데이터 (data/backtest_cache_sample/ 사용)
venv/bin/python run_backtest.py --sample --tickers 005930 000660 \
    --start 2023-01-01 --end 2023-06-30 --label check --no-chart

# 자동매매 — 실주문. 에이전트가 임의로 실행하지 말 것
python main.py            # 실전
python main.py --demo     # KIS 모의투자 서버
```

### 테스트 구조

`tests/` 는 전부 오프라인이다. `tests/helpers.py` 의 `IsolatedStateTestCase` 가 `config.STATE_DIR` 을 임시 디렉터리로 바꾸므로 **상태 파일을 쓰는 코드를 테스트할 때는 반드시 이 클래스를 상속**할 것 (안 그러면 실제 `data/state/` 를 덮는다). 시각에 의존하는 코드는 `freeze_time(모듈, 시각)` 으로 고정한다.

`tests/test_integration_flow.py` 는 `AsyncKisAPI._fetch` 만 가짜(`FakeBroker`)로 바꾸고 스크리너·전략·리스크·트레이더는 실제 코드로 돌린다. 클래스 간 시그니처를 바꿨다면 이 테스트가 잡는다. 새 TR 을 추가하면 `FakeBroker._fetch` 에도 추가해야 한다.

루트의 `test_*.py` 는 테스트가 아니라 라이브 API/네이버를 때리는 수동 조사 스크립트다.

## 아키텍처

`main.py` → `AsyncAutoTrader`(`core/async_trader.py`)가 전부를 오케스트레이션한다. `AsyncKisAPI` 인스턴스 하나를 모든 컴포넌트가 공유한다.

| 경로 | 역할 |
|------|------|
| `core/trader_api.py` | `AsyncKisAPI` — 토큰, TPS 제어, 서킷브레이커, 모든 TR 호출 |
| `core/async_trader.py` | 스케줄러 + 모니터 루프 + 진입/청산 실행, 알림, DB 기록 |
| `strategy/async_screener.py` | 유니버스 조회, 100점 스코어링 |
| `strategy/async_trading_strategy.py` | 포지션 원장(`holdings`), 진입/청산 판단, 주문 |
| `risk/async_risk_manager.py` | 시장 체제 판정, 매수 수량 산정, 일일 손실 한도, 손절가 |
| `utils/market_calendar.py` | 거래일 달력 (개장일 판정의 단일 출처) |
| `utils/state_store.py` | `data/state/*.json` 원자적 저장 |
| `utils/costs.py` | 수수료·거래세 (라이브 손익과 백테스트 공용) |
| `data/trade_db.py` | PostgreSQL 기록 (선택적) |
| `agents/` | 멀티에이전트 레이어 — **기본 비활성** |
| `backtest/` | 백테스트 엔진·수집기·optuna 최적화 |

### 포지션 상태 — 가장 중요한 불변식

`strategy.holdings` 가 포지션 원장이다. 다음 규칙을 깨면 실제로 돈을 잃는다 (전부 과거 사고에서 나온 것).

- **주문 접수는 체결이 아니다.** 매도가 접수되면 포지션을 지우지 않고 `pending_sell` 로 표시한다. 잔고 동기화(`update_holdings`)가 증권사 잔고에서 사라진 것을 확인했을 때 제거한다. 15:20 동시호가 매도는 15:30 에 체결된다.
- **포지션 메타데이터는 영속화된다.** `reason`/`entry_time`/`stop_price`/`buy_trade_id` 는 변경 즉시 `data/state/positions.json` 에 저장된다. 필드를 추가하면 `_PERSISTED_FIELDS` 에도 넣을 것. 값은 JSON 직렬화 가능해야 한다.
- **로컬에서 포지션을 임의로 지우지 말 것.** 잔고에 있는 한 다음 동기화가 다시 넣고, 그때 메타데이터를 잃는다.
- **주문과 동기화는 `_order_lock` 으로 직렬화된다.** `entry`/`exit`/`update_holdings` 는 락을 잡으므로 서로를 락 안에서 호출하면 교착이다 (asyncio.Lock 은 재진입 불가).
- **잔고 조회 실패(`{}`)는 "보유 없음"이 아니다.** 매수는 막고 기존 포지션은 유지한다.
- 잔고에 있는데 메타데이터가 없는 종목은 DB 의 미청산 매수에서 복원하고, 그것도 없으면 `reason="Standard"` 로 편입하며 텔레그램으로 알린다.

### 매수 수량

`risk_manager.plan_buy()` → 순수 함수 `plan_buy_quantity()` 가 수량을 정한다. 종목당 예산(총평가액 × 변동성 기반 비중), 총 투자 비율 한도, 가용 현금(시장가 증거금 130%) 중 가장 빡빡한 것으로 자르고, **1주도 못 사면 0주(진입 포기)** 다. 어떤 경로로도 최소 수량을 강제하지 말 것.

계좌가 작으면(2026-09 기준 약 130만원) 종목당 예산이 수만~십수만원이라 고가주는 살 수 없다. 스크리너는 유니버스 조회 단계에서 살 수 있는 가격대로 제한한다.

### 전략과 `reason` 태그

`reason` 은 청산 규칙을 고르는 load-bearing 문자열이다.

| reason | 진입 | 청산 (`check_exit_condition`) |
|---|---|---|
| `Overnight` | 15:10 스크리닝. 종가 위치 ≥ 0.85, 기술 ≥ 20점, 외국인·기관 동시 순매수 | −4% 하드스탑(항상) / D+0 보유 / D+1 +5% 익절 / D+2 09:05~09:30 또는 14:00 이후 강제 청산 |
| `Intraday`, `Momentum` | 09:05 갭 검증 후 또는 장중 스크리닝 + `check_entry_condition` | 익절 3% / 진입 시 고정한 손절가 / 트레일링 2% / 당일 동시호가 청산 |
| `Standard` | (출처 불명 포지션 편입) | 익절 `profit_cut_ratio` / 손절 / 트레일링 3% / 동시호가 / 5거래일 |

- 보유일(D+N)은 **거래일 기준**이다 (`market_calendar.trading_days_between`).
- 실거래 이력은 사실상 Overnight 뿐이다. Intraday/Momentum 은 `trading.intraday_entry_enabled` 로 끌 수 있다.
- Shadow C(`OVERNIGHT_SHADOW_C_ENABLED`)는 트레일링 변형을 `[SHADOW_C]` 로그로만 남긴다. 주문에 영향 없음.

### 스케줄 (`_scheduled_morning_routine`, 10초 틱 · `"%H:%M"` 정확 일치)

```
07:00 시장 리스크 평가 + 일별 리셋      07:30 토큰 갱신 (휴장일에도 실행)
08:00 프리마켓 스크리닝   08:30 / 08:50 재시도 (후보 0건일 때만)
09:05 갭 검증 + 첫 매수   10:30 / 12:00 / 13:30 / 14:50 장중 스크리닝·진입
11:20 / 13:00 리스크 재평가
15:10 오버나이트 진입     15:30 마감 리포트
```

- `elif` 체인이라 한 틱에 한 이벤트만 발화한다. 휴장일 가드가 체인 중간(07:30 다음)에 있다.
- 한 스텝이 ~50초 이상 걸리면 다음 `HH:MM` 슬롯을 놓친다.
- `_monitor_loop` 는 개장일 09:00~15:30 에 10초마다 잔고 대사 + 청산 판단을 한다.
- 개장일 판정은 두 루프 모두 `_is_open_today()` 를 쓴다. **`utils.is_market_open()` 은 시각까지 검사하므로 "오늘 개장일인가"에 쓰면 안 된다** (그렇게 써서 07:00/08:00 스텝이 4개월간 한 번도 실행되지 않았다).

### 거래일 달력

`market_calendar.is_trading_day()` 가 단일 출처다. KIS 휴장일조회(`CTCA0903R`, 1회에 약 3주치)로 확인된 날짜가 우선이고 `data/state/open_days.json` 에 누적된다. 미확인 날짜만 로컬 `holidays.json` 으로 판단한다. 서버의 `holidays.json` 은 8일짜리 기본 목록이라 신뢰할 수 없다.

주문이 `APBK0919`(장운영일자 상이)로 거부되면 휴장일조회를 다시 해서, 휴장으로 확인되거나 확인할 수 없을 때만 당일 매매를 중단한다.

## 설정

`config_dir/config.yaml` → `config.py` 가 모듈 상수로 로드. **yaml 에 키가 있으면 그 값이 이긴다.** `config.py` 의 숫자는 기본값일 뿐이다.

- yaml 의 `market_regimes.*` 블록에서 라이브가 실제로 읽는 키는 **`position_size_multiplier` 하나뿐**이다. `docs/market-regime-adaptive-system-report.md` 가 설명하는 체제 적응형 임계값·손익비는 구현된 적이 없다.
- `config.py` 에는 정의만 되고 쓰이지 않는 상수가 많다 (`PAPER_TRADING_MODE`, `MAX_HOLD_DAYS`, `TRAILING_STEPS`, `WEIGHT_*` 등). 페이퍼 트레이딩은 구현돼 있지 않다.
- 민감정보는 `.env` 에서만 온다. `.env` 와 `data/token.json` 은 읽지도 수정하지도 말 것.

## KIS API 규율

- **`_fetch` 는 예외를 던지지 않는다.** 모든 실패에서 `{"rt_cd": "-1", ...}` 를 반환한다. 호출자는 `rt_cd == "0"` 을 확인해야 한다.
- **주문 TR 은 `resend_on_error=False`.** 전송 후 네트워크 예외가 나면 재전송하지 않고 `_unconfirmed` 를 반환한다 (재전송하면 중복 체결). 새 주문 TR 을 추가할 때 반드시 지킬 것.
- 일봉: `get_ohlcv(count<=30)` 는 `FHKST01010400`(30행 고정), 그 이상은 `FHKST03010100`(1회 100행). 지표 워밍업이 필요한 곳은 100행을 요청할 것.
- 수급 추정치(`HHPTJ04160200`)의 각 행은 **누계**다. 최신 입력구분 행 하나만 쓴다.
- 레이트 리밋: `Semaphore(1 demo / 20 live)` + 1초 슬라이딩 윈도(1/s demo, 15/s live). 5회 연속 재시도 소진 시 60초 서킷 오픈 — 이 서킷은 주문도 막는다.
- TR ID: 거래계는 모의투자에 `V` 접두 — 잔고 `TTTC8434R`, 매수 `TTTC0012U`, 매도 `TTTC0011U`.

## 에이전트 레이어 (`agents/`)

`config.AGENTS_ENABLED`(yaml `agents.enabled`, 기본 false)일 때만 구성된다. 끈 이유와, 켜기 전에 고쳐야 할 것:

- `coordinator.py` 의 `a.get(...) or b.get(...)` 가 DataFrame 에 걸려 `ValueError` → 매수 거부권이 실제로 작동한 적이 없다
- `market_intel_agent.py` 폴링 루프에 휴장일·장시간 가드가 없다 (주말에도 API 호출, 공유 서킷브레이커를 연다)
- `risk_agent.py` 의 매도 신호가 Overnight 보유 규칙을 무시한다. 포지션 사이징에 50만원 하한이 있다
- 매수 피드백 키(`N1/N2/N3`)와 매도 피드백 키(`Overnight` 등)가 달라 통계가 쌓이지 않는다
- 백테스트가 이 레이어를 전혀 검증하지 않는다

## 백테스트의 한계

**백테스트 엔진은 라이브 전략을 재현하지 않는다.** 진입은 익일 시가, 청산은 종가 판정뿐이고 Overnight(종가 매수 → D+1~2 청산)는 시뮬레이션하지 못한다. 수급 점수는 0으로 채운다. 현재 백테스트 수치로 라이브 전략의 수익성을 판단할 수 없다.

`backtest/optimizer.py` 의 결과를 라이브 설정에 옮길 때는 이 차이를 감안할 것. 라이브 봇은 최적화를 자동 실행하지 않는다.

## 배포

`main` 푸시 → GitHub Actions(self-hosted runner) → 이미지 빌드 → 이미지 안에서 compileall·import·단위 테스트 → `kor-trade-live` 재생성 → 45초 뒤 기동 확인. **평일 08:50~15:35 의 push 는 배포하지 않는다** (수동 실행 `workflow_dispatch` 는 가능).

런타임 상태는 호스트 `/home/jongseung/kor-trade-data/` 에 마운트된다. `data/state/` 는 디렉터리째 마운트(원자적 교체용), 나머지 JSON 은 개별 파일 마운트라 CI 가 `touch` 로 빈 파일을 만든다 — 빈 파일을 읽어도 죽지 않아야 한다.

서버의 `DB_*` 환경변수는 비어 있고 DB 접속은 `data/trade_db.py` 의 하드코딩 폴백으로 동작한다. 폴백을 제거하려면 GitHub Secrets 에 `DB_*` 를 먼저 등록해야 한다.

커밋 메시지는 `fix(scope): 한국어 설명` / `feat: …` 컨벤션을 따른다.

## 참고

- `risk/risk_manager.py`(659줄)는 아무 데서도 import 하지 않는 죽은 코드다. 라이브는 `risk/async_risk_manager.py`.
- `open-trading-api/` 는 KIS 공식 예제 저장소 clone(참고용, import 안 함). TR 파라미터를 확인할 때 `examples_llm/` 을 볼 것.
- `README.md` 와 `.claude/agents/korea-stock-trading-orchestrator.md` 의 수치는 코드와 어긋난 곳이 있다. 인용 전에 코드로 확인할 것.
- 브랜치 `wip/patterns-v2` 에 미검증 패턴 전략(P1/P2/P3)이 보존돼 있다. 배포 대상이 아니다.
