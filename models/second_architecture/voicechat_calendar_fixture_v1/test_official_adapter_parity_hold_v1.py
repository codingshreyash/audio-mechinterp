"""Preserved source-only parity HOLD reproductions; no model or real WAV is run.

These are expected failures against source commit 546757a. Remove the
decorators only after the comparator rejects each missing-evidence case.
"""

import contextlib
import io
import unittest

from .official_adapter_parity import INJECTION_KEYS, compare_stock_adapter
from . import test_official_adapter_parity as parity_fixture


def fixture():
    case = parity_fixture.OfficialAdapterParityTests(
        "test_pinned_official_function_and_real_adapter_fake_outputs_pass")
    case.setUp()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return case._arms()
    finally:
        case.doCleanups()


class MissingEvidenceHoldReproductions(unittest.TestCase):
    @unittest.expectedFailure
    def test_rejects_missing_core_result_fields_in_all_arms(self):
        trace, calls, stocks = fixture()
        for phase in ("pass1", "pass2"):
            for key in ("tokens_text", "tokens_len", "text", "tokens_audio"):
                trace[phase]["raw_result"].pop(key, None)
                for stock in stocks:
                    stock[phase]["result"].pop(key, None)
        with self.assertRaises(ValueError):
            compare_stock_adapter(*stocks, trace, calls)

    @unittest.expectedFailure
    def test_rejects_missing_replay_injection_in_all_arms(self):
        trace, calls, stocks = fixture()
        self.assertIsNotNone(trace["replay"])
        for call in [calls[1], *[stock["pass2"]["kwargs"] for stock in stocks]]:
            for key in INJECTION_KEYS:
                call.pop(key, None)
        with self.assertRaises(ValueError):
            compare_stock_adapter(*stocks, trace, calls)

    @unittest.expectedFailure
    def test_rejects_empty_tts_capture_for_generated_audio(self):
        trace, calls, stocks = fixture()
        self.assertIsNotNone(trace["pass2"]["raw_result"]["audio"])
        for phase in ("pass1", "pass2"):
            trace[phase]["codec_codes_per_step"] = []
            for stock in stocks[:2]:
                stock[phase]["codes"] = []
        with self.assertRaises(ValueError):
            compare_stock_adapter(*stocks, trace, calls)


if __name__ == "__main__":
    unittest.main()
