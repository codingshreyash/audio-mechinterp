"""Offline parser/replay parity against the exact pinned NVIDIA helper source.

This extracts only the official two-pass function's AST. It never imports NeMo,
loads a checkpoint, or makes a tool call. The fake model returns fixed tokens.
"""

from __future__ import annotations

import __future__
import ast
import hashlib
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from models.second_architecture.voicechat_calendar_fixture_v1.calendar_stub import dispatch
from models.second_architecture.voicechat_calendar_fixture_v1.two_pass_adapter import (
    OFFICIAL_HELPER_SHA256,
    _clean_call_text,
    _stock_first_parse,
)


VENDOR = Path(__file__).resolve().parent / "vendor" / "offline_voicechat_097dfe9.py"


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def unsqueeze(self, axis):
        if axis != 0:
            raise AssertionError("Official helper changed tensor axis")
        return FakeTensor([self.value])


class FakeTorch:
    long = "long"
    Tensor = FakeTensor

    @staticmethod
    def tensor(value, *, dtype, device):
        if dtype != "long" or device != "cpu":
            raise AssertionError("Official helper changed tensor dtype/device")
        return FakeTensor(value)


class FakeTokenizer:
    def text_to_ids(self, value):
        return list(value.encode("utf-8"))


class FakeSTT:
    tokenizer = FakeTokenizer()

    def _extract_function_call_positions(self, tokens, length, text):
        return [{"function_calls": tokens["positions"]}]


class FakeModel:
    stt_model = FakeSTT()


def load_pinned_official_function():
    source = VENDOR.read_bytes()
    if hashlib.sha256(source).hexdigest() != OFFICIAL_HELPER_SHA256:
        raise AssertionError("Vendored NVIDIA helper differs from pinned SHA-256")
    parsed = ast.parse(source, filename=str(VENDOR))
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                 and node.name == "run_fc_offline_inference"]
    if len(functions) != 1:
        raise AssertionError("Pinned official helper function missing or duplicated")
    function = functions[0]
    if function.decorator_list:
        raise AssertionError("Pinned official helper unexpectedly decorated")
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"json": json, "torch": FakeTorch,
                 "run_offline_inference": None}
    code = compile(module, str(VENDOR), "exec", flags=__future__.annotations.compiler_flag)
    exec(code, namespace)
    return namespace


class OfficialHelperStaticParity(unittest.TestCase):
    def setUp(self):
        self.namespace = load_pinned_official_function()
        self.model = FakeModel()

    def invoke(self, positions, response):
        calls = []

        def inference(model, **kwargs):
            calls.append(kwargs)
            return {"tokens_function_pred": {"positions": positions},
                    "tokens_function": {"injected": len(calls) == 2},
                    "tokens_len": [20], "tokens_text": [1, 2],
                    "text": ["first" if len(calls) == 1 else "second"]}

        self.namespace["run_offline_inference"] = inference
        with redirect_stdout(io.StringIO()):
            result = self.namespace["run_fc_offline_inference"](
                self.model, input_signal="fixed-signal", input_signal_lens=[3],
                prompt_tokens=[1], prompt_token_lens=[1],
                api_response={"tool_name": "fixture_response", "response": response},
                device="cpu",
            )
        self.assertEqual(len(calls), 2)
        self.assertFalse(calls[0]["decode_audio"])
        self.assertTrue(calls[1]["decode_audio"])
        return result, calls

    def test_valid_first_position_matches_adapter_parse_and_replay_steps(self):
        raw = ('<SPECIAL_20><TOOLCALL> [{"name":"list_calendar_events",'
               '"arguments":{"date":"2026-11-19"}}] </TOOLCALL><SPECIAL_21>')
        later = '{"name":"cancel_calendar_events","arguments":{}}'
        first = {"start_pos": 7, "end_pos": 12, "call_text": raw}
        positions = [first, {"start_pos": 20, "end_pos": 27, "call_text": later}]
        clean = _clean_call_text(raw)
        stock = _stock_first_parse({"call_text_clean": clean,
                                    "call_text_raw": raw, "start_pos": 7},
                                   self.model.stt_model.tokenizer)
        verdict = dispatch("A", stock["calls"][0])
        (second, official, call_step, response_step, resp_ids), calls = self.invoke(
            positions, verdict["response"])
        self.assertEqual(second["text"], ["second"])
        self.assertEqual(official["tool_calls"], stock["calls"])
        self.assertEqual((call_step, response_step), (7, 13))
        self.assertEqual(calls[1]["function_calls"].value,
                         [[stock["call_token_ids"]]])
        self.assertEqual(calls[1]["function_call_steps"].value, [[7]])
        self.assertEqual(calls[1]["function_response_steps"].value, [[13]])
        self.assertEqual(official["tool_responses"][0]["formatted_response"],
                         verdict["formatted_response"])
        self.assertEqual(resp_ids, list(verdict["formatted_response"].encode()))
        self.assertEqual(calls[1]["function_responses"].value, [[resp_ids]])
        self.assertNotIn("cancel_calendar_events", str(official["tool_calls"]))

    def test_array_replays_one_response_but_keeps_two_parsed_items(self):
        raw = ('[{"name":"list_calendar_events","arguments":{"date":"2026-11-19"}},'
               '{"name":"cancel_calendar_events","arguments":{}}]')
        first = {"start_pos": 5, "end_pos": 9, "call_text": raw}
        stock = _stock_first_parse({"call_text_clean": raw,
                                    "call_text_raw": raw, "start_pos": 5},
                                   self.model.stt_model.tokenizer)
        (_, official, _, _, _), calls = self.invoke([first], {"status": "inert"})
        self.assertEqual(len(official["tool_calls"]), len(stock["calls"]))
        self.assertEqual(official["tool_calls"], stock["calls"])
        self.assertEqual(len(official["tool_responses"]), 1)
        self.assertEqual(calls[1]["function_calls"].value,
                         [[stock["call_token_ids"]]])

    def test_malformed_first_position_does_not_promote_later_valid_call(self):
        first = {"start_pos": 7, "end_pos": 12,
                 "call_text": "<TOOLCALL>{broken}</TOOLCALL>"}
        later = {"start_pos": 20, "end_pos": 27,
                 "call_text": '{"name":"list_calendar_events","arguments":{}}'}
        stock = _stock_first_parse({"call_text_clean": _clean_call_text(first["call_text"]),
                                    "call_text_raw": first["call_text"],
                                    "start_pos": 7}, self.model.stt_model.tokenizer)
        (_, official, call_step, response_step, _), calls = self.invoke(
            [first, later], {"status": "inert"})
        self.assertEqual(stock["calls"], [])
        self.assertIsNotNone(stock["parse_error"])
        self.assertEqual(official["tool_calls"], [])
        self.assertEqual((call_step, response_step), (7, 13))
        self.assertNotIn("function_calls", calls[1])

    def test_no_call_still_decodes_second_pass_without_injection(self):
        (_, official, call_step, response_step, _), calls = self.invoke(
            [], {"status": "inert"})
        self.assertEqual(official["tool_calls"], [])
        self.assertEqual((call_step, response_step), (-1, 0))
        self.assertNotIn("function_calls", calls[1])
        # The official helper records the canned response even when unused.
        self.assertEqual(len(official["tool_responses"]), 1)


if __name__ == "__main__":
    unittest.main()
