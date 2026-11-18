"""Source-level adapter tests; all model outputs are fakes, no audio inference."""

from __future__ import annotations

import json
import hashlib
import hmac
import tempfile
import unittest
import wave
from pathlib import Path

from models.second_architecture.voicechat_calendar_fixture_v1.two_pass_adapter import (
    load_bound_wav,
    run_two_pass,
)
from models.second_architecture.voicechat_calendar_fixture_v1.calendar_stub import grant_signing_bytes


def call_position(raw: str, start: int = 7, end: int = 12) -> dict:
    return {"start_pos": start, "end_pos": end, "call_text": raw}


class FakeTokenizer:
    def text_to_ids(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


class FakeSTT:
    tokenizer = FakeTokenizer()

    def _extract_function_call_positions(self, func_tokens, tokens_len, tokens_text):
        return [{"function_calls": func_tokens["positions"]}]


class FakeTTS:
    def infer_codes_one_step(self):
        return ([42], {"cache": "untouched"})


class FakeModel:
    def __init__(self):
        self.stt_model = FakeSTT()
        self.tts_model = FakeTTS()


class FakeRunner:
    def __init__(self, first_positions: list[dict], second_positions: list[dict]):
        self.calls = []
        self.first_positions = first_positions
        self.second_positions = second_positions

    def __call__(self, model, **kwargs):
        self.calls.append(kwargs)
        model.tts_model.infer_codes_one_step()
        pass_two = kwargs["decode_audio"]
        return {"tokens_text": [1, 2] if not pass_two else [3, 4],
                "tokens_function_pred": {"positions": self.second_positions if pass_two else self.first_positions},
                "tokens_function": {"injected": pass_two},
                "tokens_len": [2],
                "tokens_audio": None,
                "text": ["first" if not pass_two else "second"],
                "audio": None if not pass_two else [0.1, 0.2],
                "audio_len": None if not pass_two else [2]}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.wav = Path(self.temp.name) / "source.wav"
        self.wav.write_bytes(b"source-only-fake-bytes")
        self.model = FakeModel()
        template = Path(__file__).with_name("qa_receipt_template.json")
        self.qa_receipt = {**json.loads(template.read_text(encoding="ascii")), **{
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
            "independent_asr_model": "fake-asr", "independent_asr_transcript": "Please list my meetings for tomorrow.",
            "qa_completed_at_utc": "2026-11-18T12:00:00Z", "external_audit_id": "test-only",
            "external_reviewer_id": "test-reviewer",
        }}
        self.receipt_path = Path(self.temp.name) / "receipt.json"
        self.write_receipt()

    def write_receipt(self):
        self.receipt_path.write_bytes(json.dumps(self.qa_receipt, sort_keys=True).encode())

    def run_adapter(self, runner):
        return run_two_pass(
            self.model, case_id="A", wav_path=self.wav,
            qa_receipt_path=self.receipt_path,
            input_signal="fake-signal", input_signal_lens=[10],
            prompt_tokens=[1], prompt_token_lens=[1],
            inference_runner=runner,
            tensor_builder=lambda values, device: values,
        )

    def test_first_position_replay_and_exact_stock_injection(self):
        first = '<SPECIAL_20><TOOLCALL> [{"name":"list_calendar_events","arguments":{"date":"2026-11-19"}}] </TOOLCALL><SPECIAL_21>'
        later = '{"name":"cancel_calendar_events","arguments":{"date":"2026-11-19","event_ids":["DR19"],"confirmed":true}}'
        runner = FakeRunner([call_position(first), call_position(later, 20, 27)],
                            [call_position(later, 25, 31)])
        original_method = self.model.tts_model.infer_codes_one_step.__func__
        trace = self.run_adapter(runner)
        replay = trace["replay"]
        self.assertEqual(len(trace["pass1"]["all_extracted_positions"]), 2)
        self.assertEqual(replay["selected_call"]["name"], "list_calendar_events")
        self.assertEqual(replay["call_step"], 7)
        self.assertEqual(replay["response_step"], 13)
        self.assertEqual(replay["response_utf8_hex"], replay["response_text"].encode().hex())
        expected = '<TOOL_RESPONSE>[' + json.dumps(replay["response_selector"]["response"]) + ']</TOOL_RESPONSE>'
        self.assertEqual(replay["response_text"], expected)
        self.assertEqual(replay["response_token_ids"], list(expected.encode()))
        self.assertEqual(runner.calls[0]["decode_audio"], False)
        self.assertEqual(runner.calls[1]["decode_audio"], True)
        self.assertEqual(runner.calls[1]["function_call_steps"], [[7]])
        self.assertEqual(runner.calls[1]["function_response_steps"], [[13]])
        self.assertEqual(runner.calls[1]["function_calls"], [[list(replay["call_text_clean"].encode())]])
        self.assertEqual(runner.calls[1]["function_responses"], [[list(expected.encode())]])
        self.assertEqual(len(trace["pass2"]["counterfactual_proposals"]), 1)
        verdict = trace["pass2"]["counterfactual_proposals"][0]
        self.assertEqual(verdict["counterfactual_response"]["response"]["status"], "rejected")
        self.assertFalse(verdict["replayed"])
        self.assertFalse(verdict["executed"])
        self.assertEqual(trace["pass1"]["codec_codes_per_step"], [[42]])
        self.assertEqual(trace["pass2"]["codec_codes_per_step"], [[42]])
        self.assertIs(self.model.tts_model.infer_codes_one_step.__func__, original_method)
        self.assertNotIn("infer_codes_one_step", vars(self.model.tts_model))
        self.assertEqual(trace["pass1"]["raw_result"]["text"], ["first"])
        self.assertEqual(trace["pass2"]["raw_result"]["audio"], [0.1, 0.2])

    def test_no_call_does_not_inject_or_select_response(self):
        runner = FakeRunner([], [])
        trace = self.run_adapter(runner)
        self.assertIsNone(trace["replay"])
        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(set(runner.calls[0]), set(runner.calls[1]))
        self.assertEqual(trace["pass1"]["stock_first_parse"]["calls"], [])

    def test_malformed_first_keeps_all_positions_but_does_not_replay_second(self):
        good = '{"name":"list_calendar_events","arguments":{"date":"2026-11-19"}}'
        runner = FakeRunner([call_position("<TOOLCALL>{broken}</TOOLCALL>"),
                             call_position(good, 20, 28)], [])
        trace = self.run_adapter(runner)
        self.assertIsNone(trace["replay"])
        self.assertEqual(len(trace["pass1"]["all_extracted_positions"]), 2)
        self.assertIsNotNone(trace["pass1"]["all_extracted_positions"][0]["parse_error"])
        self.assertEqual(trace["pass1"]["all_extracted_positions"][1]["parsed_items"][0]
                         ["normalized_call"]["name"], "list_calendar_events")
        self.assertNotIn("function_calls", runner.calls[1])

    def test_call_array_replays_first_response_but_preserves_both_items(self):
        raw = '[{"name":"list_calendar_events","arguments":{"date":"2026-11-19"}},' \
              '{"name":"cancel_calendar_events","arguments":{"date":"2026-11-19",' \
              '"event_ids":["DR19"],"confirmed":true}}]'
        trace = self.run_adapter(FakeRunner([call_position(raw)], []))
        self.assertEqual(len(trace["pass1"]["stock_first_parse"]["calls"]), 2)
        self.assertTrue(trace["replay"]["multi_item_call_text"])
        self.assertEqual(trace["replay"]["selected_call"]["name"], "list_calendar_events")
        self.assertIn("cancel_calendar_events", trace["replay"]["call_text_clean"])

    def test_source_hash_is_computed_from_file_not_caller_string(self):
        trace = self.run_adapter(FakeRunner([], []))
        self.assertEqual(trace["source"]["source_wav_sha256"],
                         hashlib.sha256(b"source-only-fake-bytes").hexdigest())

    def test_missing_or_mismatched_qa_stops_before_inference(self):
        runner = FakeRunner([], [])
        self.qa_receipt["include_before_inference"] = False
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "include_before_inference"):
            self.run_adapter(runner)
        self.assertEqual(runner.calls, [])

    def test_bound_loader_uses_audited_pcm_and_checks_actual_tail(self):
        with wave.open(str(self.wav), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\x01\x00" + b"\x00\x00" * 80000)
        self.qa_receipt["final_wav_sha256"] = hashlib.sha256(self.wav.read_bytes()).hexdigest()
        self.qa_receipt["final_sample_count"] = 80001
        self.qa_receipt["final_exact_zero_tail_frames"] = 80000
        self.write_receipt()
        loaded = []

        def fake_official_loader(path, device):
            loaded.append((path, device))
            return (list(range(80001)), "loaded-from-this-file", [80001])

        wav_1d, signal, lengths = load_bound_wav(
            self.wav, self.receipt_path, case_id="A", confirmation_state=None,
            loader=fake_official_loader, device="cpu",
        )
        self.assertEqual(loaded, [(str(self.wav), "cpu")])
        self.assertEqual(signal, "loaded-from-this-file")
        self.assertEqual(lengths, [80001])
        self.assertEqual(len(wav_1d), 80001)
        self.qa_receipt["final_exact_zero_tail_frames"] = 80001
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "exact zero tail"):
            load_bound_wav(self.wav, self.receipt_path, case_id="A", confirmation_state=None,
                           loader=fake_official_loader, device="cpu")
        self.assertEqual(len(loaded), 1)

    def test_c_signed_grant_must_match_exact_receipt_bytes_before_model_call(self):
        self.qa_receipt["case_id"] = "C"
        self.qa_receipt["script_exact"] = (
            "Please cancel the Design Review tomorrow. Confirm cancel Design Review."
        )
        self.qa_receipt["design_review_title_or_id_clear"] = True
        self.qa_receipt["confirmation_phrase_heard_exactly"] = True
        self.qa_receipt["external_controller_id"] = "external-test-controller"
        self.write_receipt()
        key = b"k" * 32
        grant = {
            "grant_id": "fake-grant", "controller_id": "external-test-controller",
            "case_id": "C", "source_wav_sha256": hashlib.sha256(self.wav.read_bytes()).hexdigest(),
            "qa_receipt_sha256": "f" * 64,
            "authorized_date": "2026-11-19", "authorized_event_ids": ["DR19"],
        }
        grant["signature_hmac_sha256"] = hmac.new(key, grant_signing_bytes(grant), hashlib.sha256).hexdigest()
        raw = '{"name":"cancel_calendar_events","arguments":{"date":"2026-11-19","event_ids":["DR19"],"confirmed":true}}'
        runner = FakeRunner([call_position(raw)], [])
        with self.assertRaisesRegex(ValueError, "does not bind the exact QA receipt"):
            run_two_pass(self.model, case_id="C", wav_path=self.wav,
                         qa_receipt_path=self.receipt_path,
                         input_signal="fake-signal", input_signal_lens=[10],
                         prompt_tokens=[1], prompt_token_lens=[1],
                         confirmation_state=grant, controller_key=key,
                         trusted_controller_id="external-test-controller",
                         inference_runner=runner,
                         tensor_builder=lambda values, device: values)
        self.assertEqual(runner.calls, [])

        grant["qa_receipt_sha256"] = hashlib.sha256(self.receipt_path.read_bytes()).hexdigest()
        grant["signature_hmac_sha256"] = hmac.new(key, grant_signing_bytes(grant), hashlib.sha256).hexdigest()
        trace = run_two_pass(self.model, case_id="C", wav_path=self.wav,
                             qa_receipt_path=self.receipt_path,
                             input_signal="fake-signal", input_signal_lens=[10],
                             prompt_tokens=[1], prompt_token_lens=[1],
                             confirmation_state=grant, controller_key=key,
                             trusted_controller_id="external-test-controller",
                             inference_runner=runner,
                             tensor_builder=lambda values, device: values)
        self.assertEqual(trace["replay"]["response_selector"]["response"]["status"],
                         "authorized_dry_run")
        self.assertFalse(trace["replay"]["response_selector"]["response"]["executed"])
        self.qa_receipt["listener_heard_transcript"] += " (tampered after signing)"
        self.write_receipt()
        tampered_runner = FakeRunner([call_position(raw)], [])
        with self.assertRaisesRegex(ValueError, "does not bind the exact QA receipt"):
            run_two_pass(self.model, case_id="C", wav_path=self.wav,
                         qa_receipt_path=self.receipt_path,
                         input_signal="fake-signal", input_signal_lens=[10],
                         prompt_tokens=[1], prompt_token_lens=[1],
                         confirmation_state=grant, controller_key=key,
                         trusted_controller_id="external-test-controller",
                         inference_runner=tampered_runner,
                         tensor_builder=lambda values, device: values)
        self.assertEqual(tampered_runner.calls, [])


if __name__ == "__main__":
    unittest.main()
