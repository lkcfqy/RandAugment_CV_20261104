"""No learned model is imported by this test suite."""
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
import unittest


class ControlTests(unittest.TestCase):
    def controls(self):
        # A missing implementation is an explicit RED assertion, not an import error.
        self.assertIsNotNone(importlib.util.find_spec('controls'), 'precheck controls not implemented')
        return importlib.import_module('controls')

    def test_failed_attempts_consume_budget_and_eighth_is_rejected(self):
        c = self.controls()
        with tempfile.TemporaryDirectory() as t:
            ledger = c.AttemptLedger(Path(t) / 'attempts.json', maximum=7)
            for i in range(7):
                ledger.reserve('CPU_F')
                ledger.finish(False, 'deliberate control-test failure')
            with self.assertRaises(c.GuardViolation):
                ledger.reserve('CPU_F')
            saved = json.loads((Path(t) / 'attempts.json').read_text())
            self.assertEqual(len(saved), 7)
            self.assertTrue(all(x['status'] == 'failed' for x in saved))

    def test_unfinished_attempt_cannot_be_replaced(self):
        c = self.controls()
        with tempfile.TemporaryDirectory() as t:
            ledger = c.AttemptLedger(Path(t) / 'attempts.json', maximum=7)
            ledger.reserve('CPU_F')
            with self.assertRaises(c.GuardViolation):
                ledger.reserve('CPU_C')
            self.assertEqual(len(json.loads((Path(t) / 'attempts.json').read_text())), 1)

    def test_launch_is_single_use_even_after_failure(self):
        c = self.controls()
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'once.json'
            c.claim_launch(p, {'scope': 'control-test'})
            with self.assertRaises(c.GuardViolation):
                c.claim_launch(p, {'scope': 'control-test'})
            self.assertEqual(json.loads(p.read_text())['scope'], 'control-test')

    def test_nested_prediction_payload_cannot_be_persisted(self):
        c = self.controls()
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'result.json'
            for key in ['pred_logits', 'pred_boxes', 'count', 'scores', 'boxes']:
                with self.subTest(key=key), self.assertRaises(c.GuardViolation):
                    c.save_engineering_json(p, {'nested': [{key: [123]}]})
                self.assertFalse(p.exists())
            c.save_engineering_json(p, {'max_abs_error': 0.0001, 'mask_equal': True})
            self.assertEqual(json.loads(p.read_text())['mask_equal'], True)

    def test_wall_watchdog_kills_real_sleeping_child(self):
        c = self.controls()
        with tempfile.TemporaryDirectory() as t:
            r = c.supervise([sys.executable, '-c', 'import time; time.sleep(30)'], Path(t), wall_seconds=0.25, rss_limit=8*1024**3, mps_limit=8*1024**3)
            self.assertEqual(r['termination_reason'], 'wall_limit')
            self.assertLess(r['elapsed_seconds'], 3)
            self.assertIsNotNone(r['returncode'])
            with self.assertRaises(ProcessLookupError):
                os.kill(r['pid'], 0)

    def test_mps_guard_stops_child_using_reported_allocation(self):
        c = self.controls()
        with tempfile.TemporaryDirectory() as t:
            code = "import pathlib,time; pathlib.Path('memory.json').write_text('{\"mps_driver_bytes\": 101}'); time.sleep(30)"
            r = c.supervise([sys.executable, '-c', code], Path(t), wall_seconds=2, rss_limit=8*1024**3, mps_limit=100)
            self.assertEqual(r['termination_reason'], 'mps_limit')
            self.assertLess(r['elapsed_seconds'], 2)

    def test_successful_child_is_not_mislabeled_as_watchdog_failure(self):
        c = self.controls()
        with tempfile.TemporaryDirectory() as t:
            r = c.supervise([sys.executable, '-c', 'pass'], Path(t), wall_seconds=2, rss_limit=8*1024**3, mps_limit=8*1024**3)
            self.assertIsNone(r['termination_reason'])
            self.assertEqual(r['returncode'], 0)

    def test_concurrent_memory_updates_remain_atomic(self):
        c = self.controls()
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/'memory.json'
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(lambda i: c.save_engineering_json(p, {'sample':i}), range(40)))
            self.assertIn(json.loads(p.read_text())['sample'], range(40))


if __name__ == '__main__':
    unittest.main(verbosity=2)
