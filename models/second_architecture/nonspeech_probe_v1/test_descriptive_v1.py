"""No candidate-model load. Fake tensors and subprocesses exercise source contracts."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from . import descriptive_audit_v1 as audit
from . import descriptive_parent_v1 as parent
from . import descriptive_worker_v1 as worker


class FakeTokenizer:
    eos_token_id = 50257
    pad_token_id = 50257

    def decode(self, ids, *, skip_special_tokens):
        return "hello" if skip_special_tokens else "<|startoftranscript|>hello<|endoftext|>"


class FakeProcessor:
    tokenizer = SimpleNamespace(pad_token_id=0)

    def batch_decode(self, ids):
        return ["A"]


def logits(token: int) -> torch.Tensor:
    values = torch.full((1, 51865), -10.0, dtype=torch.float32)
    values[0, token] = 5.0
    return values


class DescriptiveSourceTests(unittest.TestCase):
    def test_pinned_local_assets_and_source_only_preflight(self) -> None:
        for name in worker.MODEL_NAMES:
            actual = worker.verify_local_assets(name)
            self.assertEqual(actual["files_sha256"]["model.safetensors"],
                             json.loads(worker.PINS.read_text())["models"][name]["weight_sha256"])
        self.assertEqual(len(worker.trusted_cases()), 20)
        self.assertEqual(worker.verify_installed_sources()["versions"],
                         {"numpy": "2.3.5", "torch": "2.15.0", "transformers": "5.20.2"})

    def test_whisper_detect_prefix_separate_from_generated_steps(self) -> None:
        prefix = [50258, 50263, 50359, 50363]
        generated = [123, 50257]
        result = SimpleNamespace(sequences=torch.tensor([prefix + generated]),
                                 scores=(logits(123), logits(50257)))
        detection = {"chosen_language_ids": [50263], "candidate_count": 99,
                     "language_logits": {str(x): float(x) for x in range(99)}}
        scored = worker.score_whisper(result, FakeTokenizer(), detection)
        self.assertEqual(scored["actual_generation_steps"], 2)
        self.assertEqual(scored["prefix_ids"], prefix)
        self.assertEqual(scored["generated_ids"], generated)
        self.assertEqual(scored["termination"], "eos")
        audit.check_whisper(scored, True)

    def test_whisper_length_censor_uses_actual_scores(self) -> None:
        prefix = [50258, 50359, 50363]
        result = SimpleNamespace(sequences=torch.tensor([prefix + [123, 124]]),
                                 scores=(logits(123), logits(124)))
        scored = worker.score_whisper(result, FakeTokenizer(), None, max_new_tokens=2)
        self.assertEqual(scored["termination"], "length_censored")
        self.assertFalse(scored["eos_seen"])
        with self.assertRaisesRegex(ValueError, "tokens after first generated EOS"):
            bad = SimpleNamespace(sequences=torch.tensor([prefix + [50257, 123]]),
                                  scores=(logits(50257), logits(123)))
            worker.score_whisper(bad, FakeTokenizer(), None, max_new_tokens=2)

    def test_ctc_full_path_and_blank_collapse(self) -> None:
        values = torch.zeros((1, 7, 32), dtype=torch.float32)
        for index, token in enumerate([0, 1, 1, 0, 1, 2, 2]):
            values[0, index, token] = 10.0
        scored = worker.score_ctc(values, FakeProcessor())
        self.assertEqual(scored["frame_ids"], [0, 1, 1, 0, 1, 2, 2])
        self.assertEqual(scored["collapsed_nonblank_ids"], [1, 1, 2])

    def test_fake_model_failure_keeps_all_planned_case_rows(self) -> None:
        def failing_load(*_args):
            raise RuntimeError("fake load failure")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "attempt"
            output.mkdir()
            for name in ("parent_start.json", "host_before.json",
                         "host_before_power_log.txt.gz", "worker_stdout.txt",
                         "worker_stderr.txt"):
                (output / name).write_bytes(b"")
            with mock.patch.object(worker, "require_independent_go", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "fake load failure"):
                    worker.run_model("whisper_multilingual", output, Path(temp), Path(temp),
                                     load=failing_load)
            records = audit.rows(output / "records.jsonl")
            cases = [row for row in records if row.get("type") == "case"]
            self.assertEqual(len(cases), 22)
            self.assertTrue(all(row["status"] == "not_run_after_failure" for row in cases))
            self.assertEqual(records[-1]["status"], "invalid_or_partial")

    def test_fake_model_success_covers_cases_repeats_and_all_differences(self) -> None:
        def fake_infer(_model_name, _processor, _model, pcm):
            sample = np.array([float(pcm[0]), float(pcm[-1])], dtype=np.float32)
            array = sample.reshape(1, 2, 1)
            output = {"transcript": "", "termination": "complete_ctc_frames"}
            return {"output": output, "capture_counts": {"ctc": {"input_values": 1}},
                    "traces": {"ctc": {"input_values": worker.trace_metric(
                        "input_values", array)}}}, {"ctc": {"input_values": array}}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "attempt"
            output.mkdir()
            for name in ("parent_start.json", "host_before.json",
                         "host_before_power_log.txt.gz", "worker_stdout.txt",
                         "worker_stderr.txt"):
                (output / name).write_bytes(b"")
            with mock.patch.object(worker, "require_independent_go", return_value={}):
                worker.run_model("wav2vec2", output, Path(temp), Path(temp),
                                 load=lambda *_: (object(), object()), infer=fake_infer)
            records = audit.rows(output / "records.jsonl")
            self.assertEqual(len([row for row in records if row.get("type") == "case"]), 22)
            self.assertEqual(len([row for row in records
                                  if row.get("type") == "silence_difference"]), 18)
            self.assertEqual(records[-1]["status"], "complete")

    def test_parent_wall_cap_kills_fake_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = parent.run_bounded(["/bin/sleep", "2"], Path(temp), seconds=0.2)
            self.assertEqual(result["limit_reason"], "wall_cap")
            self.assertNotEqual(result["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
