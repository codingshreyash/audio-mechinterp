"""Fake-output checks for the stock/adapter capture and comparison boundary."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from .official_adapter_parity import capture_stock_helper, compare_stock_adapter
from .test_official_helper_static_parity_v1 import load_pinned_official_function
from .test_two_pass_adapter import FakeModel, FakeRunner, call_position
from .two_pass_adapter import run_two_pass


class Tensor:
    def __init__(self, value):
        self.value = value

    def unsqueeze(self, axis):
        if axis != 0:
            raise AssertionError("Official tensor axis changed")
        return Tensor([self.value])

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return Tensor(self.value)

    def tolist(self):
        return self.value


class Torch:
    long = "long"

    @staticmethod
    def tensor(value, *, dtype, device):
        if dtype != "long" or device != "cpu":
            raise AssertionError("Official tensor dtype/device changed")
        return Tensor(value)


class OfficialNamespace:
    """Expose the pinned AST function's globals as patchable module attributes."""

    def __init__(self):
        object.__setattr__(self, "namespace", load_pinned_official_function())
        self.namespace["torch"] = Torch

    def __getattr__(self, name):
        return self.namespace[name]

    def __setattr__(self, name, value):
        self.namespace[name] = value


class OfficialAdapterParityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.wav = Path(self.temp.name) / "source.wav"
        self.wav.write_bytes(b"source-only-fake-bytes")
        self.receipt = Path(self.temp.name) / "receipt.json"
        template = Path(__file__).with_name("qa_receipt_template.json")
        record = {**json.loads(template.read_text(encoding="ascii")), **{
            "fixture_version": "voicechat_calendar_fixture_v1",
            "case_id": "A", "script_exact": "Please list my meetings for tomorrow.",
            "final_wav_path": str(self.wav),
            "final_wav_sha256": hashlib.sha256(self.wav.read_bytes()).hexdigest(),
            "sample_rate_hz": 16000, "channels": 1, "sample_width_bytes": 2,
            "appended_zero_frames": 80000, "include_before_inference": True,
            "listener_clear": True, "case_semantics_preserved": True,
            "target_date_understood": True, "exclusion_reason": None,
            "voice_id": "fake", "listener_id": "test-listener",
            "source_kind": "fake", "raw_render_path": "fake-source.wav",
            "raw_render_sha256": "0" * 64,
            "renderer_version_or_recording_device": "fake-source-test",
            "final_sample_count": 81000, "final_exact_zero_tail_frames": 80000,
            "listener_heard_transcript": "Please list my meetings for tomorrow.",
            "independent_asr_model": "fake-asr",
            "independent_asr_transcript": "Please list my meetings for tomorrow.",
            "qa_completed_at_utc": "2026-11-18T12:00:00Z", "external_audit_id": "test-only",
            "external_reviewer_id": "test-reviewer",
        }}
        self.receipt.write_text(json.dumps(record, sort_keys=True), encoding="ascii")
        self.model = FakeModel()
        raw = ('<SPECIAL_20><TOOLCALL>{"name":"list_calendar_events",'
               '"arguments":{"date":"2026-11-19"}}</TOOLCALL><SPECIAL_21>')
        self.positions = [call_position(raw)]

    def _arms(self, *, positions=None):
        positions = self.positions if positions is None else positions
        adapter_runner = FakeRunner(positions, [])
        adapter_calls = []

        def observed_adapter(model, **kwargs):
            adapter_calls.append(kwargs)
            return adapter_runner(model, **kwargs)

        trace = run_two_pass(
            self.model, case_id="A", wav_path=self.wav,
            qa_receipt_path=self.receipt, input_signal="same-input",
            input_signal_lens=[10], prompt_tokens=[1], prompt_token_lens=[1],
            device="cpu", inference_runner=observed_adapter,
            tensor_builder=lambda values, device: Tensor(values),
        )
        if trace["replay"]:
            selector = trace["replay"]["response_selector"]
            response = {"tool_name": selector["tool_name"], "response": selector["response"]}
        else:
            response = {"tool_name": "none", "response": {"status": "unused"}}
        official = OfficialNamespace()
        original = official.run_offline_inference

        def stock_runner(model, **kwargs):
            return FakeRunner(positions, [])(model, **kwargs)

        official.run_offline_inference = stock_runner
        stock = []
        for wrapped in (True, True, False):
            stock.append(capture_stock_helper(
                official, self.model, api_response=response,
                input_signal="same-input", input_signal_lens=[10],
                prompt_tokens=[1], prompt_token_lens=[1], device="cpu",
                capture_tts=wrapped,
            ))
            self.assertIs(official.run_offline_inference, stock_runner)
        self.assertIsNot(official.run_offline_inference, original)
        return trace, adapter_calls, stock

    def test_pinned_official_function_and_real_adapter_fake_outputs_pass(self):
        trace, calls, stock = self._arms()
        result = compare_stock_adapter(*stock, trace, calls)
        self.assertEqual(result["status"], "source_comparison_passed_pending_live_provenance_review")
        self.assertEqual(result["stock_adapter_max_abs_pcm"], 0.0)
        self.assertEqual(len(calls), 2)
        self.assertIn("function_responses", calls[1])

    def test_no_call_has_no_injection_and_still_compares_raw_audio(self):
        trace, calls, stock = self._arms(positions=[])
        result = compare_stock_adapter(*stock, trace, calls)
        self.assertTrue(result["discrete_identity"])
        self.assertNotIn("function_calls", calls[1])

    def test_rejects_discrete_function_divergence(self):
        trace, calls, stock = self._arms()
        trace["pass1"]["raw_result"]["tokens_function_pred"] = {"positions": []}
        with self.assertRaisesRegex(ValueError, "adapter.pass1.tokens_function_pred"):
            compare_stock_adapter(*stock, trace, calls)

    def test_rejects_injection_argument_divergence(self):
        trace, calls, stock = self._arms()
        calls[1]["function_response_steps"] = Tensor([[99]])
        with self.assertRaisesRegex(ValueError, "adapter.pass2.injection.function_response_steps"):
            compare_stock_adapter(*stock, trace, calls)

    def test_rejects_unstable_stock_audio_even_when_adapter_matches(self):
        trace, calls, stock = self._arms()
        stock[1]["pass2"]["result"]["audio"][0] += 0.001
        with self.assertRaisesRegex(ValueError, "waveform tolerance"):
            compare_stock_adapter(*stock, trace, calls)

    def test_rejects_wrapper_effect(self):
        trace, calls, stock = self._arms()
        stock[2]["pass2"]["result"]["audio"][0] += 0.001
        with self.assertRaisesRegex(ValueError, "waveform tolerance"):
            compare_stock_adapter(*stock, trace, calls)

    def test_stock_wrapper_restores_original_after_failure(self):
        official = OfficialNamespace()

        def fail(_model, **_kwargs):
            raise RuntimeError("synthetic failure")

        official.run_offline_inference = fail
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            capture_stock_helper(
                official, self.model,
                api_response={"tool_name": "none", "response": {}},
                input_signal="same-input", input_signal_lens=[10],
                prompt_tokens=[1], prompt_token_lens=[1], device="cpu",
            )
        self.assertIs(official.run_offline_inference, fail)


if __name__ == "__main__":
    unittest.main()
