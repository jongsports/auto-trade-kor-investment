"""백테스트 도구 — 샘플 캐시 분리, 거래세, 최적화 목적함수."""
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import config
from tests.helpers import IsolatedStateTestCase


class BacktestToolingTest(IsolatedStateTestCase):
    def test_sample_data_never_lands_in_real_cache(self):
        from backtest import data_collector as dc
        real = Path(self._tmp.name) / "real"
        sample = Path(self._tmp.name) / "sample"
        with patch.object(dc, "CACHE_DIR", real), patch.object(dc, "SAMPLE_CACHE_DIR", sample):
            df = dc.BacktestDataCollector.generate_sample_data("005930", "2023-01-01", "2023-03-31")
            dc.BacktestDataCollector(sample=True).save_sample_data("005930", df)
            self.assertTrue((sample / "005930.csv").exists())
            self.assertFalse((real / "005930.csv").exists())
            with self.assertRaises(ValueError):
                dc.BacktestDataCollector().save_sample_data("005930", df)

    def test_sell_tax_is_charged_on_exit(self):
        from backtest.engine import BacktestConfig
        self.assertEqual(BacktestConfig().sell_tax, config.SELL_TAX_RATE)
        self.assertGreater(config.SELL_TAX_RATE, 0)

    def test_optimizer_objective_reads_real_sharpe(self):
        from backtest import optimizer as opt
        from backtest.data_collector import BacktestDataCollector as C
        data = {t: C.generate_sample_data(t, "2022-01-01", "2023-06-30", seed=i)
                for i, t in enumerate(["A", "B", "C"])}
        o = opt.RegimeOptimizer(data, n_trials=1)
        trial = MagicMock()
        trial.number = 0
        trial.suggest_int = lambda name, lo, hi: {"score_threshold": 35, "max_hold_days": 3,
                                                  "max_positions": 3}[name]
        trial.suggest_float = lambda name, lo, hi: {"take_profit": 0.05, "stop_loss": 0.02,
                                                    "trailing_stop": 0.03,
                                                    "position_size_ratio": 0.15}[name]
        with patch("backtest.metrics.calculate_metrics",
                   return_value={"sharpe_ratio": 1.25, "mdd_pct": 3.0}) as m, \
                patch("backtest.optimizer.calculate_metrics", m, create=True):
            value = o._objective(trial)
        self.assertIn(value, (1.25, -9.0))   # 거래가 없으면 -9.0
        searched = set()
        trial.suggest_float = lambda name, lo, hi: searched.add(name) or 0.05
        trial.suggest_int = lambda name, lo, hi: searched.add(name) or 3
        o._objective(trial)
        self.assertNotIn("commission", searched)
        self.assertNotIn("slippage", searched)

    def test_losing_parameters_are_not_saved(self):
        from backtest import optimizer as opt
        o = opt.RegimeOptimizer({}, n_trials=1)
        o._best_params = {"take_profit": 0.1}
        o._study = MagicMock(best_value=-0.5)
        with patch.object(opt, "_CONFIG_YAML", Path(self._tmp.name) / "c.yaml"):
            with self.assertRaises(RuntimeError):
                o.save_best_params()
            self.assertFalse((Path(self._tmp.name) / "c.yaml").exists())


if __name__ == "__main__":
    unittest.main()
