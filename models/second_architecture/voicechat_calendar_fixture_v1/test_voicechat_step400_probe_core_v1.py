"""No-model tests for the VoiceChat memory-probe core."""
from __future__ import annotations

import json
import gc
import tempfile
import time
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace

import torch

from .voicechat_step400_probe_core_v1 import (
    CHECKPOINTS, CheckpointLedger, DiagnosticStop, Step400Probe,
    inventory_carried_state, run_pinned_pass1_only,
)


class FakeCuda:
    def __init__(self) -> None:
        self.resets = 0
        self.syncs = 0

    def synchronize(self, device):
        self.syncs += 1

    def memory_allocated(self, device):
        return 1000 + self.syncs

    def memory_reserved(self, device):
        return 2000 + self.syncs

    def max_memory_allocated(self, device):
        return 3000 + self.syncs

    def max_memory_reserved(self, device):
        return 4000 + self.syncs

    def reset_peak_memory_stats(self, device):
        self.resets += 1


class FakeSTT:
    def __init__(self) -> None:
        self.calls = []

    def _step_inference(self, t, state, ans, force_bos_positions=None):
        self.calls.append((t, state, ans, force_bos_positions))
        return ans


class FakeTTS:
    def __init__(self, cache) -> None:
        self.calls = []
        self.cache = cache

    def infer_codes_one_step(self, **kwargs):
        self.calls.append(kwargs)
        return (kwargs["prev_audio_tokens"], self.cache)


def tts_kwargs(token, cache):
    return {"current_subword_id": token, "prev_subword_id": token,
            "current_subword_mask": token, "prev_audio_tokens": token,
            "past_key_values": cache, "guidance_enabled": True,
            "generation_config": token, "ignore_eos_flag_stop": True}


class ProbeCoreTests(unittest.TestCase):
    def test_shared_tensor_views_deduplicate_across_families(self) -> None:
        tensor = torch.arange(16, dtype=torch.float32)
        state = {"a": tensor[:8], "b": [tensor[8:]]}
        cache = (tensor[:4],)
        result = inventory_carried_state(state, cache)
        self.assertTrue(result["complete"])
        self.assertEqual(result["families"]["stt"]["logical_bytes"], 64)
        self.assertEqual(result["families"]["stt"]["unique_storage_bytes"], 64)
        self.assertEqual(result["families"]["tts"]["unique_storage_bytes"], 64)
        self.assertEqual(result["shared_storage_bytes"], 64)
        self.assertEqual(result["global_unique_storage_bytes"], 64)
        self.assertEqual(result["global_unique_cuda_storage_bytes"], 0)
        raw = json.dumps(result)
        self.assertNotIn(str(tensor.untyped_storage().data_ptr()), raw)

    def test_unknown_cache_marks_incomplete_without_traversal(self) -> None:
        class Unknown:
            def __init__(self):
                self.tensor = torch.ones(100)
        result = inventory_carried_state({}, Unknown())
        self.assertFalse(result["complete"])
        self.assertEqual(result["unknown_count"], 1)
        self.assertEqual(result["families"]["tts"]["tensor_count"], 0)

    def test_dict_keys_never_enter_serialized_tensor_paths(self) -> None:
        secret_prompt = "SECRET_PROMPT_and_output_token_12345"
        result = inventory_carried_state(
            {secret_prompt: torch.ones(2), 12345: {secret_prompt: torch.ones(1)}},
            {secret_prompt: torch.zeros(1)})
        serialized = json.dumps(result, sort_keys=True)
        self.assertTrue(result["complete"])
        self.assertNotIn(secret_prompt, serialized)
        self.assertNotIn("12345", serialized)
        self.assertTrue(all("[entry:" in row["path"]
                            for row in result["tensor_rows"]))

    def test_dynamic_cache_shape_is_metadata_only(self) -> None:
        class DynamicLayer:
            def __init__(self):
                self.keys = torch.ones((1, 2, 3, 4))
                self.values = torch.zeros((1, 2, 3, 4))
        class DynamicCache:
            def __init__(self):
                self.layers = [DynamicLayer()]
        DynamicLayer.__module__ = "transformers.cache_utils"
        DynamicCache.__module__ = "transformers.cache_utils"
        result = inventory_carried_state({}, DynamicCache())
        self.assertTrue(result["complete"])
        self.assertEqual(result["families"]["tts"]["tensor_count"], 2)
        self.assertEqual(result["families"]["tts"]["logical_bytes"], 192)

    def test_six_checkpoints_planned_stop_and_restoration(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            ledger = CheckpointLedger(Path(folder) / "checkpoints.jsonl")
            tensor = torch.ones(4)
            model = SimpleNamespace(stt_model=FakeSTT(),
                                    tts_model=FakeTTS({"view": tensor[:2]}))
            original_stt = model.stt_model._step_inference.__func__
            original_tts = model.tts_model.infer_codes_one_step.__func__
            cuda = FakeCuda()
            now = [0.0]
            def clock():
                now[0] += 0.001
                return now[0]
            def nvml(pid):
                return {"used_bytes": 1024, "model_pid_present": True,
                        "sample_monotonic": clock()}
            probe = Step400Probe(model, ledger, cuda, nvml, pid=123,
                                 clock=clock)
            ans = object()
            token = object()
            state = {"T": 859, "tensor": tensor}
            with self.assertRaises(DiagnosticStop):
                with probe.install():
                    for t in range(1, 401):
                        self.assertIs(model.stt_model._step_inference(
                            t, state, ans, None), ans)
                        result = model.tts_model.infer_codes_one_step(
                            **tts_kwargs(token, None))
                        if t < 400:
                            self.assertIs(result[0], token)
                            self.assertIs(result[1], model.tts_model.cache)
                        self.assertIsNone(probe.pending_state)
            self.assertNotIn("_step_inference", vars(model.stt_model))
            self.assertNotIn("infer_codes_one_step", vars(model.tts_model))
            self.assertIs(model.stt_model._step_inference.__func__, original_stt)
            self.assertIs(model.tts_model.infer_codes_one_step.__func__, original_tts)
            self.assertEqual(len(model.stt_model.calls), 400)
            self.assertEqual(len(model.tts_model.calls), 400)
            self.assertEqual(cuda.resets, 6)
            result = probe.completion()
            self.assertEqual(result["checkpoint_steps"], list(CHECKPOINTS))
            records = [json.loads(line) for line in ledger.path.read_text().splitlines()]
            self.assertEqual([row["t"] for row in records], list(CHECKPOINTS))
            self.assertIn("pre_step_1_baseline", records[0])
            self.assertNotIn("pre_step_1_baseline", records[1])
            for row in records:
                self.assertLess(row["interval_start_monotonic"], row["snapshot_monotonic"])
                self.assertGreaterEqual(row["nvml_lag_seconds"], 0)
            baseline = records[0]["pre_step_1_baseline"]
            self.assertLess(baseline["snapshot_monotonic"], baseline["reset_monotonic"])
            self.assertEqual(records[0]["interval_start_monotonic"],
                             baseline["reset_monotonic"])
            self.assertEqual(records[-1]["carried"]["shared_storage_bytes"], 16)

    def test_probe_does_not_retain_per_step_result(self) -> None:
        class Payload:
            pass
        class NoRetentionSTT:
            def _step_inference(self, t, state, ans, force_bos_positions=None):
                return ans
        with tempfile.TemporaryDirectory() as folder:
            ledger = CheckpointLedger(Path(folder) / "checkpoints.jsonl")
            model = SimpleNamespace(stt_model=NoRetentionSTT(), tts_model=FakeTTS(None))
            probe = Step400Probe(model, ledger, FakeCuda(),
                                 lambda pid: {"used_bytes": 1,
                                              "model_pid_present": True,
                                              "sample_monotonic": time.monotonic()})
            with probe.install():
                payload = Payload()
                ref = weakref.ref(payload)
                state = {"T": 859}
                self.assertIs(model.stt_model._step_inference(1, state, payload), payload)
                model.tts_model.infer_codes_one_step(**tts_kwargs(object(), None))
                del payload
                gc.collect()
                self.assertIsNone(ref())
                self.assertIsNone(probe.pending_state)
            ledger.close()

    def test_bad_step_and_bad_tts_args_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            ledger = CheckpointLedger(Path(folder) / "checkpoints.jsonl")
            model = SimpleNamespace(stt_model=FakeSTT(), tts_model=FakeTTS(None))
            cuda = FakeCuda()
            probe = Step400Probe(model, ledger, cuda,
                                 lambda pid: {"used_bytes": 1,
                                              "model_pid_present": True,
                                              "sample_monotonic": time.monotonic()})
            with probe.install():
                with self.assertRaisesRegex(ValueError, "STT step order"):
                    model.stt_model._step_inference(2, {"T": 859}, object())
                state = {"T": 859}
                model.stt_model._step_inference(1, state, object())
                with self.assertRaisesRegex(ValueError, "TTS call order"):
                    model.tts_model.infer_codes_one_step()
            ledger.close()

    def test_pass1_uses_official_helper_arguments_and_never_reaches_pass2(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            ledger = CheckpointLedger(Path(folder) / "checkpoints.jsonl")
            model = SimpleNamespace(stt_model=FakeSTT(), tts_model=FakeTTS(None))
            audio, lengths, prompt, prompt_lengths = (object() for _ in range(4))
            calls = []
            def official_pass1(given_model, **kwargs):
                calls.append((given_model, kwargs))
                state = {"T": 859}
                for t in range(1, 859):
                    given_model.stt_model._step_inference(t, state, None, None)
                    given_model.tts_model.infer_codes_one_step(
                        **tts_kwargs(None, None))
                raise AssertionError("diagnostic failed to stop at 400")
            result = run_pinned_pass1_only(
                model=model, input_signal=audio, input_signal_lens=lengths,
                prompt_tokens=prompt, prompt_token_lens=prompt_lengths,
                offline_inference=official_pass1, ledger=ledger, cuda=FakeCuda(),
                nvml=lambda pid: {"used_bytes": 1, "model_pid_present": True,
                                  "sample_monotonic": time.monotonic()})
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0][0], model)
            self.assertEqual(set(calls[0][1]), {
                "input_signal", "input_signal_lens", "prompt_tokens",
                "prompt_token_lens", "decode_audio"})
            self.assertIs(calls[0][1]["input_signal"], audio)
            self.assertIs(calls[0][1]["input_signal_lens"], lengths)
            self.assertIs(calls[0][1]["prompt_tokens"], prompt)
            self.assertIs(calls[0][1]["prompt_token_lens"], prompt_lengths)
            self.assertIs(calls[0][1]["decode_audio"], False)
            self.assertEqual(len(model.stt_model.calls), 400)
            self.assertEqual(len(model.tts_model.calls), 400)
            self.assertEqual(result["status"], "diagnostic_early_stop_complete")

    def test_ledger_refuses_reuse_and_wrong_order(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.jsonl"
            ledger = CheckpointLedger(path)
            with self.assertRaisesRegex(ValueError, "checkpoint order"):
                ledger.write({"t": 50})
            ledger.close()
            with self.assertRaisesRegex(ValueError, "consumed"):
                CheckpointLedger(path)


if __name__ == "__main__":
    unittest.main()
