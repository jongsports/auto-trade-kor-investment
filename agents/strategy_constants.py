"""
전략 이름 상수 정의 — v2 (3대 고확률 전략)

기존 S1~S7 → N1/N2/N3 전면 재설계.
AlphaAgent와 RiskAgent가 동일 문자열을 이 모듈에서 단일 소스로 관리합니다.
"""

# ── 신규 전략 식별자 ──────────────────────────────
N1_INSTITUTIONAL_FLOW  = "N1_institutional_flow"   # 기관수급 모멘텀
N2_VOLATILITY_SQUEEZE  = "N2_volatility_squeeze"   # 변동성 수축 폭발
N3_OVERSOLD_REVERSAL   = "N3_oversold_reversal"    # 과매도 반전

# 전체 전략 목록
ALL_STRATEGIES = (N1_INSTITUTIONAL_FLOW, N2_VOLATILITY_SQUEEZE, N3_OVERSOLD_REVERSAL)

# ── 체제별 활성화 규칙 ────────────────────────────
# BULL:         N1*, N2, N3 전부 활성
# NORMAL:       N1*, N2, N3 전부 활성
# BEAR:         N3 only (과매도 반전만)
# VOLATILE_UP:  N2, N3 (스퀴즈 + 반전)
# VOLATILE_DOWN: N3 only
# *N1은 실전에서만 활성 (백테스트 시 수급 데이터 없음)

VOLATILE_ALLOWED = (N2_VOLATILITY_SQUEEZE, N3_OVERSOLD_REVERSAL)
BEAR_ALLOWED     = (N3_OVERSOLD_REVERSAL,)

# 빠른 손절 전략 (ATR×1.0 SL)
FAST_EXIT_STRATEGIES = (N1_INSTITUTIONAL_FLOW,)

# 넓은 손절 전략 (ATR×1.5 SL)
WIDE_EXIT_STRATEGIES = (N2_VOLATILITY_SQUEEZE,)

# ATR TP 배수 (전략별 비대칭 손익비)
STRATEGY_TP_MULTIPLIER = {
    N1_INSTITUTIONAL_FLOW: 3.0,   # 3:1 R:R
    N2_VOLATILITY_SQUEEZE: 4.0,   # 4:1.5 R:R
    N3_OVERSOLD_REVERSAL:  2.5,   # 2.5:1 R:R
}

STRATEGY_SL_MULTIPLIER = {
    N1_INSTITUTIONAL_FLOW: 1.0,
    N2_VOLATILITY_SQUEEZE: 1.5,
    N3_OVERSOLD_REVERSAL:  1.0,
}

# ── 하위 호환: 기존 코드에서 참조하는 상수 (임포트 에러 방지) ──
S1_SUPPLY_DEMAND     = N1_INSTITUTIONAL_FLOW
S2_DIP_BUY           = N3_OVERSOLD_REVERSAL
S3_GAP_MOMENTUM      = N2_VOLATILITY_SQUEEZE
S4_LIMIT_UP_CHASE    = N1_INSTITUTIONAL_FLOW
S5_FOREIGN_STREAK    = N1_INSTITUTIONAL_FLOW
S6_THEME_LEADER      = N2_VOLATILITY_SQUEEZE
S7_BOLLINGER_SQUEEZE = N2_VOLATILITY_SQUEEZE
BEAR_BLOCKED         = ()  # 더 이상 사용 안 함 — BEAR_ALLOWED로 대체
SLOW_EXIT_STRATEGIES = ()
