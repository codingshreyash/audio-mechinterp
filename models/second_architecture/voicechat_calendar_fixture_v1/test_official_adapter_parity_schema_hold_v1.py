"""Source-only reproductions of over-strict result-schema requirements.

The pinned helper supports a function-token fallback and does not require a
tokens_audio result field. These were failures against 2e17bd7 and now guard
the source-supported schema.
"""

import copy
import unittest

from .official_adapter_parity import compare_stock_adapter
from .test_official_adapter_parity_hold_v1 import fixture


class SchemaFallbackHoldReproductions(unittest.TestCase):
    def test_accepts_only_tokens_function_fallback_when_consistent(self):
        trace, calls, stocks = fixture()
        for phase in ("pass1", "pass2"):
            for result in [trace[phase]["raw_result"],
                           *[stock[phase]["result"] for stock in stocks]]:
                result["tokens_function"] = copy.deepcopy(result["tokens_function_pred"])
                del result["tokens_function_pred"]
        self.assertEqual(
            compare_stock_adapter(*stocks, trace, calls)["discrete_identity"], True)

    def test_accepts_missing_optional_tokens_audio_with_tts_codes(self):
        trace, calls, stocks = fixture()
        self.assertTrue(trace["pass2"]["codec_codes_per_step"])
        for phase in ("pass1", "pass2"):
            for result in [trace[phase]["raw_result"],
                           *[stock[phase]["result"] for stock in stocks]]:
                del result["tokens_audio"]
        self.assertEqual(
            compare_stock_adapter(*stocks, trace, calls)["discrete_identity"], True)

    def test_accepts_no_function_stream_on_no_call_path(self):
        from .test_official_adapter_parity import OfficialAdapterParityTests

        case = OfficialAdapterParityTests("test_no_call_has_no_injection_and_still_compares_raw_audio")
        case.setUp()
        try:
            trace, calls, stocks = case._arms(positions=[])
            for phase in ("pass1", "pass2"):
                for result in [trace[phase]["raw_result"],
                               *[stock[phase]["result"] for stock in stocks]]:
                    result.pop("tokens_function_pred", None)
                    result.pop("tokens_function", None)
            self.assertTrue(compare_stock_adapter(*stocks, trace, calls)["discrete_identity"])
        finally:
            case.doCleanups()


if __name__ == "__main__":
    unittest.main()
