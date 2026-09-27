"""상태 저장소, 비용 모델, 알림, 설정."""
import json
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import config
from tests.helpers import IsolatedStateTestCase, run
from utils import costs
from utils.notifier import AsyncTelegramNotifier, split_message
from utils.state_store import load_state, save_state


class StateStoreTest(IsolatedStateTestCase):
    def test_round_trip(self):
        self.assertTrue(save_state("x", {"a": 1, "한글": "값"}))
        self.assertEqual(load_state("x", None), {"a": 1, "한글": "값"})

    def test_missing_or_empty_or_corrupt_returns_default(self):
        self.assertEqual(load_state("none", {"d": 1}), {"d": 1})
        (Path(config.STATE_DIR) / "empty.json").write_text("")     # CI 가 touch 로 만든 파일
        self.assertEqual(load_state("empty", []), [])
        (Path(config.STATE_DIR) / "bad.json").write_text("{not json")
        self.assertEqual(load_state("bad", 7), 7)

    def test_failed_write_keeps_previous_content(self):
        save_state("x", {"v": 1})
        with patch("utils.state_store.json.dump", side_effect=OSError("disk full")):
            self.assertFalse(save_state("x", {"v": 2}))
        self.assertEqual(load_state("x", None), {"v": 1})
        self.assertEqual([f for f in os.listdir(config.STATE_DIR) if f.endswith(".tmp")], [])


class CostModelTest(unittest.TestCase):
    def test_flat_trade_loses_the_round_trip_cost(self):
        amount, ratio = costs.net_pnl(10_000, 10_000, 10)
        expected = -(100_000 * config.COMMISSION_RATE
                     + 100_000 * (config.COMMISSION_RATE + config.SELL_TAX_RATE))
        self.assertAlmostEqual(amount, expected)
        self.assertLess(ratio, 0)

    def test_breakeven_move_covers_costs(self):
        cost_ratio = 2 * config.COMMISSION_RATE + config.SELL_TAX_RATE
        amount, _ = costs.net_pnl(10_000, 10_000 * (1 + cost_ratio * 1.01), 10)
        self.assertGreater(amount, 0)

    def test_invalid_inputs(self):
        self.assertEqual(costs.net_pnl(0, 10_000, 1), (0.0, 0.0))


class NotifierTest(unittest.TestCase):
    def test_long_message_is_split_on_line_boundaries(self):
        msg = "\n".join(f"{i:04d} " + "x" * 95 for i in range(100))   # 10,099자
        chunks = split_message(msg)
        self.assertTrue(all(len(c) <= 4096 for c in chunks))
        self.assertEqual("\n".join(chunks), msg)

    def test_single_overlong_line_is_split(self):
        chunks = split_message("y" * 9000)
        self.assertEqual([len(c) for c in chunks], [4096, 4096, 808])

    def test_html_parse_error_falls_back_to_plain_text(self):
        n = AsyncTelegramNotifier.__new__(AsyncTelegramNotifier)
        n.chat_id, n.api_url = "1", "http://x"
        sent = []

        class Resp:
            def __init__(self, status): self.status = status
            async def text(self): return "can't parse entities"
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False

        class Session:
            def post(self, url, json=None, timeout=None):
                sent.append(json)
                return Resp(400 if "parse_mode" in json else 200)

        ok = run(n._send_chunk(Session(), "<b>장애</b> 오류: <class 'ValueError'> a & b"))
        self.assertTrue(ok)
        self.assertEqual(len(sent), 2)
        self.assertNotIn("parse_mode", sent[1])
        self.assertEqual(sent[1]["text"], "장애 오류: <class 'ValueError'> a & b")


class ConfigTest(unittest.TestCase):
    def test_loss_and_profit_cut_are_not_the_random_optuna_sample(self):
        self.assertNotAlmostEqual(config.LOSS_CUT_RATIO, 0.0466)
        self.assertNotAlmostEqual(config.PROFIT_CUT_RATIO, 0.1436)

    def test_agents_are_disabled_by_default(self):
        self.assertFalse(config.AGENTS_ENABLED)


if __name__ == "__main__":
    unittest.main()
