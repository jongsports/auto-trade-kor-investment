"""
AlphaGenerationAgent v2 — 3대 고확률 전략

  N1. 기관수급 모멘텀   — 외국인+기관 동시 순매수 + 가격 돌파 + 거래량 (3:1 R:R)
  N2. 변동성 수축 폭발  — BB 스퀴즈 → 상단돌파 + OBV + MACD 동시 확인 (4:1.5 R:R)
  N3. 과매도 반전       — RSI<30 + Stochastic 골든크로스 + 장기추세 유효 (2.5:1 R:R)

원칙: 선별 > 빈도 | 손익비 비대칭 | 3개+ 독립 시그널 동시 확인(Confluence)
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent, AgentSignal, MarketContext
from agents.strategy_constants import (
    N1_INSTITUTIONAL_FLOW, N2_VOLATILITY_SQUEEZE, N3_OVERSOLD_REVERSAL,
    ALL_STRATEGIES, VOLATILE_ALLOWED, BEAR_ALLOWED,
    STRATEGY_TP_MULTIPLIER, STRATEGY_SL_MULTIPLIER,
    # 하위 호환
    S1_SUPPLY_DEMAND, S2_DIP_BUY, S3_GAP_MOMENTUM,
    S4_LIMIT_UP_CHASE, S5_FOREIGN_STREAK, S6_THEME_LEADER, S7_BOLLINGER_SQUEEZE,
)

logger = logging.getLogger("agent.alpha")


# ─────────────────────────────────────────────
# 전략 파라미터 (v2 Korean market-tuned)
# ─────────────────────────────────────────────

class StrategyParams:
    # N1: 기관수급 모멘텀
    N1_MIN_FOREIGN_BUY = 0          # 외국인 순매수 > 0
    N1_MIN_INST_BUY = 0             # 기관 순매수 > 0
    N1_VOLUME_RATIO = 3.0           # 거래량 > MA20 × 3배
    N1_BREAKOUT_FROM_MA20 = 0.005   # MA20 대비 +0.5% 이상 돌파
    N1_MIN_ADX = 20                 # 추세 강도

    # N2: 변동성 수축 폭발
    N2_MAX_BB_WIDTH = 0.02          # BB 폭 < 2% (압착 조건)
    N2_SQUEEZE_DAYS = 5             # 압착 유지 최소 일수
    N2_BREAKOUT_CONFIRM = 0.003     # 상단 밴드 돌파 최소 0.3%
    N2_VOLUME_RATIO = 2.0           # 거래량 > MA20 × 2배
    N2_OBV_SLOPE_MIN = 0            # OBV 상승 추세

    # N3: 과매도 반전
    N3_RSI_OVERSOLD = 30            # RSI < 30
    N3_STOCH_CROSS = True           # Stochastic %K > %D 골든크로스
    N3_MIN_ADX = 15                 # 최소 추세 존재
    N3_VOLUME_RATIO = 1.5           # 최소 거래량


P = StrategyParams()


class AlphaGenerationAgent(BaseAgent):
    """
    3가지 고확률 전략으로 매수 신호를 발굴합니다.
    각 전략은 독립 메서드로 구현되어 독립적으로 성과 추적이 가능합니다.
    """

    def __init__(self, api_client=None, config=None):
        super().__init__("AlphaGen", api_client, config)

        # 전략별 성과 추적
        self._strategy_wins: Dict[str, int] = {s: 0 for s in ALL_STRATEGIES}
        self._strategy_total: Dict[str, int] = {s: 0 for s in ALL_STRATEGIES}
        self._strategy_pnl: Dict[str, float] = {s: 0.0 for s in ALL_STRATEGIES}

        # 외국인 연속 매수 추적 (N1용)
        self._foreign_streak_file = os.path.join("data", "foreign_streak.json")
        self._foreign_streak: Dict[str, Any] = self._load_foreign_streak()

    # ─────────────────────────────────────────────
    # 라이프사이클
    # ─────────────────────────────────────────────

    def _load_foreign_streak(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self._foreign_streak_file):
                with open(self._foreign_streak_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"[AlphaGen] foreign_streak 로드 실패: {e}")
        return {}

    def _save_foreign_streak(self) -> None:
        try:
            with open(self._foreign_streak_file, "w") as f:
                json.dump(self._foreign_streak, f, indent=2)
        except Exception as e:
            logger.warning(f"[AlphaGen] foreign_streak 저장 실패: {e}")

    async def initialize(self) -> None:
        logger.info("[AlphaGen] v2 — 3대 고확률 전략 에이전트 초기화 완료")

    async def analyze(
        self, context: MarketContext, candidates: List[Dict[str, Any]]
    ) -> List[AgentSignal]:
        if not candidates:
            return []

        signals: List[AgentSignal] = []
        active_strategies = self._get_active_strategies(context)

        tasks = [
            self._analyze_ticker(ticker_data, context, active_strategies)
            for ticker_data in candidates
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.debug(f"분석 오류: {result}")
                continue
            if result:
                signals.extend(result)

        self._save_foreign_streak()
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals

    def _get_active_strategies(self, context: MarketContext) -> List[str]:
        """시장 체제에 따라 활성화할 전략 결정."""
        regime = getattr(context, "regime", "NORMAL")
        regime_str = regime.value if hasattr(regime, "value") else str(regime)

        if regime_str in ("BEAR", "VOLATILE_DOWN"):
            return [N3_OVERSOLD_REVERSAL]
        elif regime_str in ("VOLATILE", "VOLATILE_UP"):
            return [N2_VOLATILITY_SQUEEZE, N3_OVERSOLD_REVERSAL]
        elif regime_str == "BULL":
            return [N1_INSTITUTIONAL_FLOW, N2_VOLATILITY_SQUEEZE, N3_OVERSOLD_REVERSAL]
        else:  # NORMAL
            return [N1_INSTITUTIONAL_FLOW, N2_VOLATILITY_SQUEEZE, N3_OVERSOLD_REVERSAL]

    async def _analyze_ticker(
        self,
        data: Dict[str, Any],
        context: MarketContext,
        active: List[str],
    ) -> List[AgentSignal]:
        ticker = data.get("ticker") or data.get("code", "")
        if not ticker:
            return []

        ohlcv: Optional[pd.DataFrame] = data.get("ohlcv") or data.get("_ohlcv_snapshot")
        if ohlcv is None or len(ohlcv) < 30:
            return []

        signals: List[AgentSignal] = []
        indicators = self._calc_indicators(ohlcv)
        if indicators is None:
            return []

        investor = data.get("investor_trend") or {
            "foreign_net_buy":     data.get("foreign_net_buy", 0),
            "institution_net_buy": data.get("institution_net_buy", 0),
            "data_available":      True,
        }

        strategy_map = {
            N1_INSTITUTIONAL_FLOW: (self._strategy_n1_institutional_flow, (ticker, ohlcv, indicators, investor, context)),
            N2_VOLATILITY_SQUEEZE: (self._strategy_n2_volatility_squeeze, (ticker, ohlcv, indicators)),
            N3_OVERSOLD_REVERSAL:  (self._strategy_n3_oversold_reversal, (ticker, ohlcv, indicators, investor)),
        }

        for sid, (fn, args) in strategy_map.items():
            if sid not in active:
                continue
            try:
                sig = fn(*args)
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.debug(f"[{sid}] {ticker} 전략 오류: {e}")

        return signals

    # ─────────────────────────────────────────────
    # 공통 지표 사전 계산
    # ─────────────────────────────────────────────

    def _calc_indicators(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """전 전략에서 공통으로 사용하는 지표 계산."""
        try:
            c = df["close"]
            v = df["volume"]
            h = df["high"]
            lo = df["low"]
            n = len(c)

            if n < 20:
                return None

            # Moving Averages
            ma5  = c.rolling(5).mean()
            ma20 = c.rolling(20).mean()
            ma60 = c.rolling(60).mean() if n >= 60 else None

            # Volume
            vol_ma20 = v.rolling(20).mean()

            # OBV (On Balance Volume)
            obv = (v * np.sign(c.diff().fillna(0))).cumsum()
            obv_slope = obv.diff(5).iloc[-1] if n >= 6 else 0

            # RSI (Wilder's EWM)
            delta = c.diff()
            gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / (loss + 1e-9)
            rsi = 100 - (100 / (1 + rs))

            # MACD
            ema12 = c.ewm(span=12, adjust=False).mean()
            ema26 = c.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal_line = macd.ewm(span=9, adjust=False).mean()
            hist = macd - signal_line

            # Bollinger Bands
            bb_mid = ma20
            bb_std = c.rolling(20).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-9)

            # ATR
            tr = pd.concat([
                h - lo,
                (h - c.shift()).abs(),
                (lo - c.shift()).abs(),
            ], axis=1).max(axis=1)
            atr14 = tr.ewm(alpha=1/14, adjust=False).mean()

            # Stochastic
            low14  = lo.rolling(14).min()
            high14 = h.rolling(14).max()
            k_raw = (c - low14) / (high14 - low14 + 1e-9) * 100
            stoch_k = k_raw.rolling(3).mean()
            stoch_d = stoch_k.rolling(3).mean()

            # ADX / DI — Wilder's DM (상호배제 조건 적용)
            up_move = h.diff()
            down_move = -lo.diff()
            plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
            minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
            tr14 = tr.ewm(alpha=1/14, adjust=False).mean()
            plus_di  = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / (tr14 + 1e-9)
            minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / (tr14 + 1e-9)
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
            adx = dx.ewm(alpha=1/14, adjust=False).mean()

            cur = c.iloc[-1]
            prev = c.iloc[-2] if n >= 2 else cur

            return {
                "close": c, "high": h, "low": lo, "volume": v,
                "ma5": ma5, "ma20": ma20, "ma60": ma60,
                "vol_ma20": vol_ma20, "obv_slope": float(obv_slope),
                "rsi": rsi, "rsi_cur": float(rsi.iloc[-1]),
                "macd": macd, "signal": signal_line, "hist": hist,
                "hist_cur": float(hist.iloc[-1]),
                "hist_prev": float(hist.iloc[-2]) if n >= 2 else 0,
                "bb_upper": bb_upper, "bb_lower": bb_lower, "bb_mid": bb_mid,
                "bb_width": bb_width, "bb_width_cur": float(bb_width.iloc[-1]),
                "atr14": atr14, "atr_cur": float(atr14.iloc[-1]),
                "stoch_k": stoch_k, "stoch_d": stoch_d,
                "stoch_k_cur": float(stoch_k.iloc[-1]),
                "stoch_d_cur": float(stoch_d.iloc[-1]),
                "stoch_k_prev": float(stoch_k.iloc[-2]) if n >= 2 else 0,
                "stoch_d_prev": float(stoch_d.iloc[-2]) if n >= 2 else 0,
                "adx_cur": float(adx.iloc[-1]),
                "plus_di_cur": float(plus_di.iloc[-1]),
                "minus_di_cur": float(minus_di.iloc[-1]),
                "cur": cur, "prev": prev,
                "vol_cur": float(v.iloc[-1]),
                "vol_ratio": float(v.iloc[-1] / (vol_ma20.iloc[-1] + 1e-9)),
                "ma20_cur": float(ma20.iloc[-1]),
                "ma5_cur": float(ma5.iloc[-1]),
            }
        except Exception as e:
            logger.debug(f"지표 계산 오류: {e}")
            return None

    # ─────────────────────────────────────────────
    # N1: 기관수급 모멘텀 (Institutional Flow Momentum)
    # 조건: 외국인+기관 동시 순매수 + MA20 돌파 + 거래량 3배 + ADX>20
    # R:R = ATR×3 : ATR×1 (3:1)
    # ─────────────────────────────────────────────

    def _strategy_n1_institutional_flow(
        self,
        ticker: str,
        df: pd.DataFrame,
        ind: Dict,
        investor: Dict,
        context: MarketContext,
    ) -> Optional[AgentSignal]:
        # 조건 1: 외국인 + 기관 동시 순매수
        foreign = investor.get("foreign_net_buy", 0)
        inst = investor.get("institution_net_buy", 0)
        if foreign <= P.N1_MIN_FOREIGN_BUY or inst <= P.N1_MIN_INST_BUY:
            return None

        # 조건 2: MA20 돌파 (현재가 > MA20 × 1.005)
        ma20 = ind["ma20_cur"]
        cur = ind["cur"]
        if ma20 <= 0 or cur <= ma20 * (1 + P.N1_BREAKOUT_FROM_MA20):
            return None

        # 조건 3: 거래량 폭발 (MA20 × 3배 이상)
        if ind["vol_ratio"] < P.N1_VOLUME_RATIO:
            return None

        # 조건 4: 추세 존재 (ADX > 20)
        if ind["adx_cur"] < P.N1_MIN_ADX:
            return None

        # 조건 5: MACD 방향 확인 (히스토그램 양수)
        if ind["hist_cur"] <= 0:
            return None

        # 조건 6: +DI > -DI (상승 추세 방향)
        if ind["plus_di_cur"] <= ind["minus_di_cur"]:
            return None

        # Score & Confidence
        flow_score = min(foreign, 100000) / 100000 * 10  # 수급 강도 0~10
        vol_bonus = min(ind["vol_ratio"] - 3.0, 2.0) * 5  # 초과 거래량 보너스
        score = 70.0 + flow_score + vol_bonus + ind["adx_cur"] * 0.3
        score = min(score, 95)

        atr = ind["atr_cur"]
        tp_mult = STRATEGY_TP_MULTIPLIER[N1_INSTITUTIONAL_FLOW]
        sl_mult = STRATEGY_SL_MULTIPLIER[N1_INSTITUTIONAL_FLOW]
        entry = cur
        target = entry + atr * tp_mult
        stop = entry - atr * sl_mult

        confidence = 0.60 + flow_score * 0.02 + ind["vol_ratio"] * 0.01
        confidence = min(confidence, 0.90)

        return self._make_signal(
            "BUY", ticker, confidence=confidence,
            score=score, strategy=N1_INSTITUTIONAL_FLOW,
            entry_price=entry, target_price=round(target, 0), stop_price=round(stop, 0),
            metadata={
                "foreign_buy": foreign, "inst_buy": inst,
                "vol_ratio": round(ind["vol_ratio"], 2),
                "adx": round(ind["adx_cur"], 1),
                "breakout_pct": round((cur / ma20 - 1) * 100, 2),
            },
        )

    # ─────────────────────────────────────────────
    # N2: 변동성 수축 폭발 (Volatility Squeeze Explosion)
    # 조건: BB폭<2% 5일+ → 상단돌파 + OBV상승 + MACD 골든크로스
    # R:R = ATR×4 : ATR×1.5 (2.67:1)
    # ─────────────────────────────────────────────

    def _strategy_n2_volatility_squeeze(
        self,
        ticker: str,
        df: pd.DataFrame,
        ind: Dict,
    ) -> Optional[AgentSignal]:
        # 조건 1: BB 폭 수축 (최근 5일 모두 < 2%)
        bb_width = ind["bb_width"]
        n = len(bb_width)
        if n < P.N2_SQUEEZE_DAYS + 1:
            return None

        recent_bb = bb_width.iloc[-(P.N2_SQUEEZE_DAYS + 1):-1]  # 어제까지 5일
        if recent_bb.max() > P.N2_MAX_BB_WIDTH:
            return None

        # 조건 2: 오늘 상단 돌파
        cur = ind["cur"]
        bb_upper_val = float(ind["bb_upper"].iloc[-1])
        breakout_pct = (cur - bb_upper_val) / (bb_upper_val + 1e-9)
        if breakout_pct < P.N2_BREAKOUT_CONFIRM:
            return None

        # 조건 3: 거래량 폭발 (스퀴즈 후 폭발이므로 중요)
        if ind["vol_ratio"] < P.N2_VOLUME_RATIO:
            return None

        # 조건 4: OBV 상승 추세 (5일 기울기 양수)
        if ind["obv_slope"] <= P.N2_OBV_SLOPE_MIN:
            return None

        # 조건 5: MACD 히스토그램 양수 + 이전 음수→양수 전환 선호
        if ind["hist_cur"] <= 0:
            return None
        macd_golden = ind["hist_prev"] <= 0 and ind["hist_cur"] > 0
        macd_bonus = 5 if macd_golden else 0

        # 조건 6: Stochastic 과매수 아님 (K < 85)
        if ind["stoch_k_cur"] > 85:
            return None

        # Score & Confidence
        squeeze_tightness = max(0, P.N2_MAX_BB_WIDTH - recent_bb.mean()) / P.N2_MAX_BB_WIDTH * 10
        score = 68.0 + squeeze_tightness + breakout_pct * 500 + ind["vol_ratio"] * 2 + macd_bonus
        score = min(score, 95)

        atr = ind["atr_cur"]
        tp_mult = STRATEGY_TP_MULTIPLIER[N2_VOLATILITY_SQUEEZE]
        sl_mult = STRATEGY_SL_MULTIPLIER[N2_VOLATILITY_SQUEEZE]
        entry = cur
        target = entry + atr * tp_mult
        stop = entry - atr * sl_mult

        confidence = 0.58 + squeeze_tightness * 0.02 + breakout_pct * 5 + ind["vol_ratio"] * 0.01
        confidence = min(confidence, 0.88)

        return self._make_signal(
            "BUY", ticker, confidence=confidence,
            score=score, strategy=N2_VOLATILITY_SQUEEZE,
            entry_price=entry, target_price=round(target, 0), stop_price=round(stop, 0),
            metadata={
                "bb_width_avg": round(float(recent_bb.mean()) * 100, 2),
                "breakout_pct": round(breakout_pct * 100, 2),
                "vol_ratio": round(ind["vol_ratio"], 2),
                "obv_slope": round(ind["obv_slope"], 0),
                "macd_golden": macd_golden,
            },
        )

    # ─────────────────────────────────────────────
    # N3: 과매도 반전 (Oversold Reversal)
    # 조건: RSI<30 + Stochastic %K>%D 골든크로스 + 장기추세(MA60) 유효
    # R:R = ATR×2.5 : ATR×1 (2.5:1)
    # ─────────────────────────────────────────────

    def _strategy_n3_oversold_reversal(
        self,
        ticker: str,
        df: pd.DataFrame,
        ind: Dict,
        investor: Dict,
    ) -> Optional[AgentSignal]:
        # 조건 1: RSI 과매도 (< 30)
        if ind["rsi_cur"] > P.N3_RSI_OVERSOLD:
            return None

        # 조건 2: Stochastic %K > %D 골든크로스 (이전 K<D → 현재 K>D)
        if not (ind["stoch_k_cur"] > ind["stoch_d_cur"]
                and ind["stoch_k_prev"] <= ind["stoch_d_prev"]):
            return None

        # 조건 3: 장기 상승추세 유효 (현재가 > MA60, 또는 MA60 없으면 MA20)
        if ind["ma60"] is not None:
            ma60_cur = float(ind["ma60"].iloc[-1])
            if ind["cur"] < ma60_cur:
                return None
        else:
            if ind["cur"] < ind["ma20_cur"]:
                return None

        # 조건 4: 최소 거래량 (너무 적으면 유동성 부족)
        if ind["vol_ratio"] < P.N3_VOLUME_RATIO:
            return None

        # 조건 5: ADX > 15 (최소한의 추세 존재 — 완전 횡보 제외)
        if ind["adx_cur"] < P.N3_MIN_ADX:
            return None

        # 보너스: 외국인 순매수 전환 시 추가 점수
        foreign_bonus = 0
        if investor.get("data_available", True) and investor.get("foreign_net_buy", 0) > 0:
            foreign_bonus = 8

        # Score & Confidence
        oversold_depth = max(0, P.N3_RSI_OVERSOLD - ind["rsi_cur"])  # RSI가 낮을수록 높음
        score = 65.0 + oversold_depth * 1.5 + ind["vol_ratio"] * 2 + foreign_bonus
        score = min(score, 95)

        atr = ind["atr_cur"]
        tp_mult = STRATEGY_TP_MULTIPLIER[N3_OVERSOLD_REVERSAL]
        sl_mult = STRATEGY_SL_MULTIPLIER[N3_OVERSOLD_REVERSAL]
        entry = ind["cur"]
        target = entry + atr * tp_mult
        stop = entry - atr * sl_mult

        confidence = 0.55 + oversold_depth * 0.02 + (foreign_bonus / 80)
        confidence = min(confidence, 0.85)

        return self._make_signal(
            "BUY", ticker, confidence=confidence,
            score=score, strategy=N3_OVERSOLD_REVERSAL,
            entry_price=entry, target_price=round(target, 0), stop_price=round(stop, 0),
            metadata={
                "rsi": round(ind["rsi_cur"], 1),
                "stoch_k": round(ind["stoch_k_cur"], 1),
                "stoch_d": round(ind["stoch_d_cur"], 1),
                "adx": round(ind["adx_cur"], 1),
                "foreign_buy": investor.get("foreign_net_buy", 0),
                "vol_ratio": round(ind["vol_ratio"], 2),
            },
        )

    # ─────────────────────────────────────────────
    # 전략 성과 통계
    # ─────────────────────────────────────────────

    def strategy_stats(self) -> Dict[str, Dict]:
        stats = {}
        for sid in self._strategy_total:
            total = self._strategy_total[sid]
            wins  = self._strategy_wins[sid]
            pnl   = self._strategy_pnl[sid]
            stats[sid] = {
                "total": total,
                "win_rate": wins / total if total > 0 else 0,
                "avg_pnl": pnl / total if total > 0 else 0,
            }
        return stats

    def update_strategy_outcome(self, strategy: str, won: bool, pnl: float) -> None:
        if strategy in self._strategy_total:
            self._strategy_total[strategy] += 1
            if won:
                self._strategy_wins[strategy] += 1
            self._strategy_pnl[strategy] += pnl
