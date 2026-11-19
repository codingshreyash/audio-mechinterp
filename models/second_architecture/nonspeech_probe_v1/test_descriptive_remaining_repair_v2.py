"""No-weight tests for English generate and one-dimensional Wav2Vec2 traces."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from . import descriptive_remaining_audit_v2 as audit
from . import descriptive_worker_v2 as worker


class RemainingRepairTests(unittest.TestCase):
    def test_installed_english_generate_contract_without_weights(self) -> None:
        from transformers import GenerationConfig
        from transformers.models.whisper.generation_whisper import WhisperGenerationMixin

        spec = json.loads(worker.PINS.read_text())["models"]["whisper_english"]
        snapshot = worker._snapshot("whisper_english", spec)
        config = GenerationConfig.from_pretrained(snapshot, local_files_only=True)
        self.assertIs(config.is_multilingual, False)
        with self.assertRaisesRegex(ValueError, "Cannot specify `task` or `language`"):
            WhisperGenerationMixin._set_language_and_task(
                None, "transcribe", None, copy.deepcopy(config))
        WhisperGenerationMixin._set_language_and_task(
            None, None, None, copy.deepcopy(config))
        kwargs = worker.english_generation_kwargs(object(), object())
        self.assertNotIn("task", kwargs)
        self.assertNotIn("language", kwargs)
        self.assertEqual(kwargs["max_new_tokens"], 128)
        self.assertIs(kwargs["do_sample"], False)
        self.assertEqual(kwargs["num_beams"], 1)

    def test_one_dimensional_input_trace_and_silence_difference(self) -> None:
        trace = np.asarray([[0.0, -2.0, 3.0, -4.0]], dtype=np.float32)
        donor = np.zeros_like(trace)
        metric = worker.trace_metric("input_values", trace)
        difference = worker.trace_difference("input_values", trace, donor)
        self.assertEqual(metric["shape"], [1, 4])
        self.assertEqual(metric["time_axis"], 1)
        self.assertEqual(metric["row_l2"], [0.0, 2.0, 3.0, 4.0])
        self.assertEqual(difference["row_l2"], [0.0, 2.0, 3.0, 4.0])
        self.assertEqual(difference["nonzero_rows"], 3)
        self.assertAlmostEqual(difference["whole_l2"], np.sqrt(29))
        high_rank = np.asarray([[[3.0, 4.0], [5.0, 12.0]]], dtype=np.float32)
        self.assertEqual(worker.trace_metric("post_block0", high_rank)["row_l2"], [5.0, 13.0])

    def test_model_specific_source_only_preflight_and_cross_arm_go(self) -> None:
        english = worker.preflight_receipt("whisper_english")
        wav = worker.preflight_receipt("wav2vec2")
        self.assertEqual(english["case_ids"], wav["case_ids"])
        self.assertEqual(len(english["case_ids"]), 20)
        self.assertNotEqual(english["attempt_path"], wav["attempt_path"])
        self.assertNotEqual(english["review_path"], wav["review_path"])
        self.assertEqual(english["prior_failed_attempt"]["files_sha256"],
                         worker.MODEL_PATHS["whisper_english"]["prior_sha256"])
        review = {"schema": "nonspeech_descriptive_remaining_repair_review_v2",
                  "status": "GO", "model": "whisper_english",
                  "attempt_path": english["attempt_path"],
                  "preflight_sha256": "a" * 64,
                  "source_identity": english["source_identity"],
                  "prior_failed_attempt": english["prior_failed_attempt"],
                  "manifest_sha256": worker.MANIFEST_SHA256}
        worker.validate_review_fields("whisper_english", review, english, "a" * 64)
        with self.assertRaisesRegex(ValueError, "does not bind"):
            worker.validate_review_fields("wav2vec2", review, wav, "b" * 64)

    def test_fake_wav_worker_writes_complete_1d_trace_differences(self) -> None:
        def fake_infer(_model_name, _processor, _model, pcm):
            array = np.asarray([[float(pcm[0]), float(pcm[-1])]], dtype=np.float32)
            return {"output": {"transcript": "", "termination": "complete_ctc_frames"},
                    "input_geometry": worker.wav_geometry(len(pcm),
                        (len(pcm) - 400) // 320 + 1),
                    "capture_counts": {"ctc": {"input_values": 1}},
                    "traces": {"ctc": {"input_values": worker.trace_metric(
                        "input_values", array)}}}, {"ctc": {"input_values": array}}

        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp) / "attempt"
            attempt.mkdir()
            for name in ("parent_start.json", "host_before.json",
                         "host_before_power_log.txt.gz", "worker_stdout.txt",
                         "worker_stderr.txt"):
                (attempt / name).write_bytes(b"")
            with mock.patch.object(worker, "require_independent_go", return_value={}):
                worker.run_model("wav2vec2", attempt, Path(temp), Path(temp),
                                 load=lambda *_: (object(), object()), infer=fake_infer)
            records = audit.rows(attempt / "records.jsonl")
            self.assertEqual(records[0]["schema"], "nonspeech_descriptive_remaining_v2")
            self.assertEqual(len([x for x in records if x.get("type") == "case"]), 22)
            differences = [x for x in records if x.get("type") == "silence_difference"]
            self.assertEqual(len(differences), 18)
            self.assertEqual(len(differences[0]["differences"]["ctc"]["input_values"]["row_l2"]),
                             2)
            self.assertEqual(records[-1]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
