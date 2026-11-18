"""Fake-model identity and fail-closed tests; no VoiceChat checkpoint is loaded."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from models.second_architecture.voicechat_calendar_fixture_v1.passive_dual_head_recorder import (
    PassiveDualHeadRecorder,
)
from models.second_architecture.voicechat_calendar_fixture_v1.two_pass_adapter import run_two_pass


CALL = '{"name":"list_calendar_events","arguments":{"date":"2026-11-19"}}'


class FakeTokenizer:
    def text_to_ids(self, text):
        return list(text.encode("utf-8"))


class FakeLLM(torch.nn.Module):
    def forward(self, x):
        return {"last_hidden_state": x + 0.25}


class FakeHead(torch.nn.Module):
    def __init__(self, multiplier):
        super().__init__()
        self.multiplier = multiplier

    def forward(self, x):
        return torch.cat([x[:, :, :1] * self.multiplier + i for i in range(5)], dim=-1)


class FakeSTT(torch.nn.Module):
    use_function_head = True
    text_pad_id = 0
    tokenizer = FakeTokenizer()

    def __init__(self):
        super().__init__()
        self.llm = FakeLLM()
        self.lm_head = FakeHead(1)
        self.function_head = FakeHead(2)
        self.positions = []
        self.force_one_position = False

    def forward(self, input_embeds):
        output = self.llm(input_embeds)
        text = self.lm_head(output["last_hidden_state"])
        function = self.function_head(output["last_hidden_state"])
        text[:, :, 0] += 2  # Same in-place post-head boost shape as pinned NeMo.
        return {"text_logits": text, "function_logits": function}

    def _step_zero(self, state):
        ans = self(torch.ones((1, 1, 3)))
        state["gen_text"][:, 0] = ans["text_logits"][:, -1].argmax(dim=-1)
        state["gen_function"][:, 0] = ans["function_logits"][:, -1].argmax(dim=-1)
        return ans, state

    def _step_inference(self, t, state, ans, force_bos_positions=None):
        # Pinned Nemotron uses no KV cache and recomputes the growing prefix.
        prefix = 1 if self.force_one_position or state["use_cache"] else t + 1
        ans = self(torch.ones((1, prefix, 3)) * (t + 1))
        state["gen_text"][:, t] = ans["text_logits"][:, -1].argmax(dim=-1)
        if state["gen_function"][0, t] == self.text_pad_id:
            state["gen_function"][:, t] = ans["function_logits"][:, -1].argmax(dim=-1)
        return ans

    def _post_inference(self, state, prompt_token_lens):
        return {
            "tokens_text": state["gen_text"].clone(),
            "tokens_function": state["gen_function"].clone(),
            "tokens_function_pred": state["gen_function"].clone(),
            "tokens_len": torch.tensor([4]),
            "tokens_audio": None,
            "text": ["I will check tomorrow's meetings."],
            "audio": torch.tensor([[0.1, 0.2, 0.3]]),
            "audio_len": torch.tensor([3]),
        }

    def _extract_function_call_positions(self, func_tokens, tokens_len, tokens_text):
        return [{"function_calls": self.positions}]


class FakeTTS:
    def infer_codes_one_step(self, **kwargs):
        return torch.tensor([[[11, 12]]]), {"cache": "unchanged"}


class FakeModel:
    def __init__(self):
        self.stt_model = FakeSTT()
        self.tts_model = FakeTTS()


class FakeRunner:
    def __init__(self, *, skip_stt_at=None, duplicate_stt_at=None,
                 skip_tts_at=None, use_cache=False):
        self.calls = []
        self.skip_stt_at = skip_stt_at
        self.duplicate_stt_at = duplicate_stt_at
        self.skip_tts_at = skip_tts_at
        self.use_cache = use_cache

    def __call__(self, model, **kwargs):
        self.calls.append(kwargs)
        model.stt_model.positions = (
            [{"start_pos": 1, "end_pos": 3, "call_text": CALL}]
            if not kwargs["decode_audio"] else []
        )
        state = {"gen_text": torch.zeros((1, 4), dtype=torch.long),
                 "gen_function": torch.zeros((1, 4), dtype=torch.long),
                 "is_prompt_position_mask": torch.tensor([[False, False, False, False]]),
                 "start_gen_pos": 0, "prompt_token_lens": torch.tensor([0]),
                 "function_response_spans": [[(2, 3)]] if kwargs["decode_audio"] else [[]],
                 "T": 4, "use_cache": self.use_cache}
        if "function_calls" in kwargs:
            state["gen_function"][0, 2] = 3  # Pass-2 injection, not a selected head token.
        ans, _ = model.stt_model._step_zero(state)
        for t in range(1, 4):
            if t == self.skip_stt_at:
                continue
            ans = model.stt_model._step_inference(t, state, ans, None)
            if t == self.duplicate_stt_at:
                ans = model.stt_model._step_inference(t, state, ans, None)
            if t != self.skip_tts_at:
                model.tts_model.infer_codes_one_step(
                    current_subword_id=state["gen_text"][:, t:t + 1],
                    prev_subword_id=state["gen_text"][:, t - 1:t],
                )
        return model.stt_model._post_inference(state, kwargs["prompt_token_lens"])


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.wav = Path(self.temp.name) / "case.wav"
        self.wav.write_bytes(b"fake-audio-for-source-test")
        template = Path(__file__).with_name("qa_receipt_template.json")
        receipt = json.loads(template.read_text())
        receipt.update({
            "fixture_version": "voicechat_calendar_fixture_v1",
            "case_id": "A", "script_exact": "Please list my meetings for tomorrow.",
            "final_wav_path": str(self.wav),
            "final_wav_sha256": hashlib.sha256(self.wav.read_bytes()).hexdigest(),
            "sample_rate_hz": 16000, "channels": 1, "sample_width_bytes": 2,
            "appended_zero_frames": 80000, "include_before_inference": True,
            "listener_clear": True, "case_semantics_preserved": True,
            "target_date_understood": True, "exclusion_reason": None,
            "voice_id": "fake", "listener_id": "fake-listener",
            "source_kind": "fake", "raw_render_path": "fake.wav",
            "raw_render_sha256": "0" * 64,
            "renderer_version_or_recording_device": "fake-test",
            "final_sample_count": 81000, "final_exact_zero_tail_frames": 80000,
            "listener_heard_transcript": "Please list my meetings for tomorrow.",
            "independent_asr_model": "fake-asr",
            "independent_asr_transcript": "Please list my meetings for tomorrow.",
            "qa_completed_at_utc": "2026-11-18T12:00:00Z",
            "external_audit_id": "test-only", "external_reviewer_id": "fake-reviewer",
        })
        self.receipt = Path(self.temp.name) / "qa.json"
        self.receipt.write_text(json.dumps(receipt))

    def run_adapter(self, model, runner):
        return run_two_pass(
            model, case_id="A", wav_path=self.wav,
            qa_receipt_path=self.receipt,
            input_signal="fake-signal", input_signal_lens=[4],
            prompt_tokens=[1], prompt_token_lens=[1],
            inference_runner=runner, tensor_builder=lambda values, device: values,
        )

    def test_identity_channels_and_parser_with_passive_recorder(self):
        stock_model, stock_runner = FakeModel(), FakeRunner()
        stock = self.run_adapter(stock_model, stock_runner)
        model, runner = FakeModel(), FakeRunner()
        recorder = PassiveDualHeadRecorder(model, source_verifier=lambda _: None)
        traced = self.run_adapter(model, recorder.wrap_runner(runner))
        recorder.bind_adapter_trace(traced)
        self.assertEqual(stock["replay"], traced["replay"])
        self.assertEqual(stock["pass1"]["all_extracted_positions"],
                         traced["pass1"]["all_extracted_positions"])
        self.assertEqual(stock["pass2"]["counterfactual_proposals"],
                         traced["pass2"]["counterfactual_proposals"])
        for label in ("pass1", "pass2"):
            for channel in ("tokens_text", "tokens_function", "tokens_function_pred", "audio"):
                self.assertTrue(torch.equal(stock[label]["raw_result"][channel],
                                            traced[label]["raw_result"][channel]))
            frames = recorder.passes[label]["frames"]
            self.assertEqual([row["frame_index_untrimmed"] for row in frames], [0, 1, 2, 3])
            self.assertEqual([row["observed_prefix_length"] for row in frames],
                             [1, 2, 3, 4])
            self.assertTrue(all(row["raw_lm_head_logits"].shape == (1, 1, 5)
                                for row in frames))
            self.assertEqual(len(recorder.passes[label]["tts_steps"]), 3)
            self.assertEqual(frames[0]["raw_lm_head_logits"][0, 0, 0].item() + 2,
                             frames[0]["returned_text_logits"][0, 0, 0].item())
        self.assertEqual(recorder.passes["pass2"]["frames"][2]["function_token_source"],
                         "injected")
        self.assertEqual(recorder.passes["pass1"]["frames"][2]["function_token_source"],
                         "model")
        self.assertNotIn("_step_zero", vars(model.stt_model))
        self.assertNotIn("infer_codes_one_step", vars(model.tts_model))
        paths = recorder.save(Path(self.temp.name) / "recorded")
        saved = torch.load(paths["pass2"], weights_only=True)
        self.assertEqual(saved["parser"]["external_tool_execution"], False)
        self.assertEqual(saved["result_channels_unspliced"]["audio"].shape, (1, 3))
        self.assertEqual(saved["untrimmed_selected"]["function_response_spans"], [[(2, 3)]])

    def test_cache_mode_records_one_position_per_step(self):
        model, runner = FakeModel(), FakeRunner(use_cache=True)
        recorder = PassiveDualHeadRecorder(model, source_verifier=lambda _: None)
        trace = self.run_adapter(model, recorder.wrap_runner(runner))
        recorder.bind_adapter_trace(trace)
        self.assertEqual([row["observed_prefix_length"]
                          for row in recorder.passes["pass2"]["frames"]], [1, 1, 1, 1])

    def test_skipped_or_duplicate_stt_and_missing_tts_fail_closed(self):
        for runner, error in ((FakeRunner(skip_stt_at=2), "frame sequence"),
                              (FakeRunner(duplicate_stt_at=1), "frame sequence"),
                              (FakeRunner(skip_tts_at=3), "frame count")):
            with self.subTest(error=error):
                model = FakeModel()
                recorder = PassiveDualHeadRecorder(model, source_verifier=lambda _: None)
                with self.assertRaisesRegex(ValueError, error):
                    self.run_adapter(model, recorder.wrap_runner(runner))
                self.assertNotIn("_step_zero", vars(model.stt_model))
                self.assertNotIn("infer_codes_one_step", vars(model.tts_model))

    def test_no_cache_wrong_prefix_fails_at_first_bad_frame(self):
        model = FakeModel()
        model.stt_model.force_one_position = True
        recorder = PassiveDualHeadRecorder(model, source_verifier=lambda _: None)
        with self.assertRaisesRegex(ValueError, "prefix differs"):
            self.run_adapter(model, recorder.wrap_runner(FakeRunner(use_cache=False)))
        self.assertNotIn("_step_inference", vars(model.stt_model))

    def test_source_verifier_and_frame_cap_fail_before_or_during_model(self):
        model, runner = FakeModel(), FakeRunner()
        recorder = PassiveDualHeadRecorder(model, source_verifier=lambda _: (_ for _ in ()).throw(RuntimeError("unpinned")))
        with self.assertRaisesRegex(RuntimeError, "unpinned"):
            self.run_adapter(model, recorder.wrap_runner(runner))
        self.assertEqual(runner.calls, [])
        capped = PassiveDualHeadRecorder(model, source_verifier=lambda _: None,
                                         max_frames_per_pass=1)
        with self.assertRaisesRegex(ValueError, "frame sequence or cap"):
            self.run_adapter(model, capped.wrap_runner(runner))
        self.assertNotIn("_step_zero", vars(model.stt_model))
        self.assertNotIn("infer_codes_one_step", vars(model.tts_model))

        byte_capped = PassiveDualHeadRecorder(model, source_verifier=lambda _: None,
                                               max_capture_bytes_per_pass=1)
        with self.assertRaisesRegex(MemoryError, "tensor budget"):
            self.run_adapter(model, byte_capped.wrap_runner(FakeRunner()))
        self.assertNotIn("_step_inference", vars(model.stt_model))
        self.assertNotIn("infer_codes_one_step", vars(model.tts_model))

    def test_missing_function_head_fails_before_inference(self):
        model, runner = FakeModel(), FakeRunner()
        model.stt_model.use_function_head = False
        recorder = PassiveDualHeadRecorder(model, source_verifier=lambda _: None)
        with self.assertRaisesRegex(ValueError, "function head"):
            self.run_adapter(model, recorder.wrap_runner(runner))
        self.assertEqual(runner.calls, [])

    def test_binding_rejects_other_result_and_save_requires_parser(self):
        model, runner = FakeModel(), FakeRunner()
        recorder = PassiveDualHeadRecorder(model, source_verifier=lambda _: None)
        trace = self.run_adapter(model, recorder.wrap_runner(runner))
        with self.assertRaisesRegex(ValueError, "parser trace"):
            recorder.save(Path(self.temp.name) / "unbound")
        trace["pass2"]["raw_result"] = {}
        with self.assertRaisesRegex(ValueError, "does not contain recorded raw result"):
            recorder.bind_adapter_trace(trace)


if __name__ == "__main__":
    unittest.main()
