"""
베이지안 파라미터 최적화 (optuna TPESampler).

사용 흐름:
  opt = RegimeOptimizer(ohlcv_data, regime="BULL", n_trials=50)
  best = opt.optimize()
  opt.save_best_params()   # config.yaml 자동 업데이트

최적화 대상 (7파라미터):
  1. score_threshold     (35~65)
  2. take_profit         (0.02~0.15)
  3. stop_loss           (0.01~0.06)
  4. trailing_stop       (0.01~0.07)
  5. max_hold_days       (1~10)
  6. max_positions       (2~8)
  7. position_size_ratio (0.05~0.25)

수수료·슬리피지는 시장이 정하는 값이지 고를 수 있는 값이 아니므로 탐색하지 않는다.

목적 함수: sharpe 최대화, MDD > 15% 시 패널티(-1.0) 적용

주의: 백테스트 엔진은 라이브 전략(Overnight 종가 매수 등)을 그대로 재현하지 않는다.
여기서 나온 값을 라이브 설정에 옮기기 전에 그 차이를 확인할 것. 라이브 봇은 이
모듈을 자동 실행하지 않는다.
"""
from __future__ import annotations

import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("backtest.optimizer")

# config.yaml 경로
_CONFIG_YAML = Path(__file__).parent.parent / "config_dir" / "config.yaml"


class RegimeOptimizer:
    """optuna TPESampler 기반 체제별 파라미터 최적화."""

    def __init__(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        regime: Optional[str] = None,
        n_trials: int = 50,
        base_score_threshold: int = 45,
    ):
        self.ohlcv_data = ohlcv_data
        self.regime = regime
        self.n_trials = n_trials
        self.base_score_threshold = base_score_threshold
        self._study = None
        self._best_params: Optional[Dict] = None

    # ------------------------------------------------------------------ #
    # 목적 함수
    # ------------------------------------------------------------------ #

    def _objective(self, trial) -> float:
        """
        9파라미터 탐색 → BacktestConfig 주입 → Sharpe 반환.
        MDD > 15% 시 -1.0 패널티 부과.
        """
        import config as _app_cfg
        from backtest.engine import BacktestConfig, BacktestEngine
        from backtest.metrics import calculate_metrics

        # 1. 탐색 파라미터 제안
        score_threshold     = trial.suggest_int("score_threshold", 35, 65)
        take_profit         = trial.suggest_float("take_profit", 0.02, 0.15)
        stop_loss           = trial.suggest_float("stop_loss", 0.01, 0.06)
        trailing_stop       = trial.suggest_float("trailing_stop", 0.01, 0.07)
        max_hold_days       = trial.suggest_int("max_hold_days", 1, 10)
        max_positions       = trial.suggest_int("max_positions", 2, 8)
        position_size_ratio = trial.suggest_float("position_size_ratio", 0.05, 0.25)

        # 2. BacktestConfig 구성 (regime은 engine.py가 오버라이드하므로 score_threshold만 넘김)
        cfg = BacktestConfig(
            score_threshold=score_threshold,
            take_profit=take_profit,
            stop_loss=stop_loss,
            trailing_stop=trailing_stop,
            max_hold_days=max_hold_days,
            max_positions=max_positions,
            position_size_ratio=position_size_ratio,
            market_regime=self.regime,
        )

        # 3. 엔진 실행
        try:
            engine = BacktestEngine(cfg, self.ohlcv_data)
            result = engine.run()
        except Exception as e:
            logger.debug(f"trial {trial.number} 엔진 오류: {e}")
            return -9.0

        # 4. 성과 지표 계산
        if not result["trades"]:
            return -9.0

        try:
            metrics = calculate_metrics(
                trades=result["trades"],
                equity_curve=result["equity_curve"],
                initial_capital=cfg.initial_capital,
            )
        except Exception as e:
            logger.debug(f"trial {trial.number} 지표 오류: {e}")
            return -9.0

        # 키 이름을 틀리게 읽으면 모든 trial 이 0.0 동점이 되어 첫 난수 샘플이
        # "최적"으로 뽑힌다 (과거 "sharpe" 로 읽어 실제로 그렇게 됐다).
        sharpe = float(metrics["sharpe_ratio"])
        mdd    = float(metrics["mdd_pct"])

        # MDD > 15% 패널티
        if mdd > 15.0:
            sharpe -= 1.0

        return sharpe

    # ------------------------------------------------------------------ #
    # 최적화 실행
    # ------------------------------------------------------------------ #

    def optimize(self) -> Dict:
        """
        n_trials 회 탐색 후 최적 파라미터 딕셔너리 반환.
        콘솔에 "Best sharpe: X.XX" 출력.
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("optuna 미설치. `pip install optuna>=3.0.0` 후 재실행.")

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=42)
        self._study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"regime_{self.regime or 'default'}",
        )

        label = self.regime or "기본값"
        logger.info(f"[Optimizer] {label} 체제 최적화 시작 ({self.n_trials}회)")

        self._study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=False,
        )

        best = self._study.best_params
        best_val = self._study.best_value
        self._best_params = best

        print(f"\nBest sharpe: {best_val:.4f}")
        print(f"Best params ({label}):")
        for k, v in best.items():
            print(f"  {k}: {v}")

        return best

    # ------------------------------------------------------------------ #
    # 결과 저장
    # ------------------------------------------------------------------ #

    def save_best_params(self) -> None:
        """
        최적 파라미터를 config.yaml의 해당 market_regime 블록에 업데이트.
        regime이 None이면 trading 섹션 기본값에 업데이트.
        """
        if self._best_params is None:
            raise RuntimeError("optimize() 먼저 호출 필요.")
        if self._study.best_value <= 0:
            raise RuntimeError(
                f"최적 샤프가 {self._study.best_value:.2f} 로 0 이하다 — "
                f"손실 나는 파라미터를 설정에 쓰지 않는다."
            )

        with open(_CONFIG_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        p = self._best_params

        if self.regime:
            # market_regimes.{regime} 블록 업데이트
            regimes = data.setdefault("market_regimes", {})
            block = regimes.setdefault(self.regime, {})
            block["momentum_tp_ratio"] = round(float(p["take_profit"]), 4)
            block["momentum_sl_ratio"] = round(float(p["stop_loss"]), 4)
            block["trailing_stop"]     = round(float(p["trailing_stop"]), 4)
            block["max_hold_days"]     = int(p["max_hold_days"])
            block["max_stock_count"]   = int(p["max_positions"])
            # score_threshold_boost: base - optimized
            boost = self.base_score_threshold - int(p["score_threshold"])
            block["score_threshold_boost"] = boost
        else:
            # trading 섹션 기본값 업데이트
            trading = data.setdefault("trading", {})
            trading["profit_cut_ratio"] = round(float(p["take_profit"]), 4)
            trading["loss_cut_ratio"]   = round(float(p["stop_loss"]), 4)
            trading["max_hold_days"]    = int(p["max_hold_days"])
            trading["max_stock_count"]  = int(p["max_positions"])

        with open(_CONFIG_YAML, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        logger.info(f"[Optimizer] config.yaml 업데이트 완료 (regime={self.regime})")
        print(f"\nconfig.yaml 업데이트 완료: {_CONFIG_YAML}")
