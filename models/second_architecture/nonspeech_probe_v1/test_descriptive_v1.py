"""No candidate-model load. Fake tensors and subprocesses exercise source contracts."""
from __future__ import annotations

import json
import gzip
import copy
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


def fake_whisper_result(prefix: list[int], generated: list[int],
                        *, vocab: int = 51865) -> SimpleNamespace:
    raw = []
    processed = []
    states = []
    for index, token in enumerate(generated):
        vector = torch.full((1, vocab), -10.0, dtype=torch.float32)
        vector[0, token] = 5.0
        raw.append(vector)
        filtered = vector.clone()
        if index == 0:
            filtered[0, 321] = -float("inf")
        processed.append(filtered)
        states.append((torch.full((1, len(prefix) if index == 0 else 1, 384),
                                  float(index + 1)),))
    return SimpleNamespace(sequences=torch.tensor([prefix + generated]),
                           scores=tuple(processed), logits=tuple(raw),
                           decoder_hidden_states=tuple(states))


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
        result = fake_whisper_result(prefix, generated)
        config = json.loads((worker._snapshot("whisper_multilingual",
            json.loads(worker.PINS.read_text())["models"]["whisper_multilingual"])
            / "generation_config.json").read_text())
        detection = {"chosen_language_ids": [50263], "candidate_count": 99,
                     "raw_logits_sha256": "0" * 64,
                     "language_logits": {name: (10.0 if token == 50263 else 0.0)
                                         for name, token in config["lang_to_id"].items()}}
        scored = worker.score_whisper(result, FakeTokenizer(), detection)
        self.assertEqual(scored["actual_generation_steps"], 2)
        self.assertEqual(scored["prefix_ids"], prefix)
        self.assertEqual(scored["generated_ids"], generated)
        self.assertEqual(scored["termination"], "eos")
        self.assertEqual(scored["score_rows"][0]["raw_selected_logit"], 5.0)
        self.assertNotEqual(scored["score_rows"][0]["raw_logits_sha256"],
                            scored["score_rows"][0]["processed_logits_sha256"])
        audit.check_whisper(scored, True)
        changed = copy.deepcopy(scored)
        changed["language_detection"]["language_logits"]["<|en|>"] = 11.0
        with self.assertRaisesRegex(ValueError, "not the argmax"):
            audit.check_whisper(changed, True)
        changed = copy.deepcopy(scored)
        changed["first_raw_logits"]["values"][123] = 6.0
        with self.assertRaisesRegex(ValueError, "hash or value"):
            audit.check_whisper(changed, True)

    def test_whisper_length_censor_uses_actual_scores(self) -> None:
        prefix = [50258, 50359, 50363]
        result = fake_whisper_result(prefix, [123, 124])
        scored = worker.score_whisper(result, FakeTokenizer(), None, max_new_tokens=2)
        self.assertEqual(scored["termination"], "length_censored")
        self.assertFalse(scored["eos_seen"])
        with self.assertRaisesRegex(ValueError, "tokens after first generated EOS"):
            bad = fake_whisper_result(prefix, [50257, 123])
            worker.score_whisper(bad, FakeTokenizer(), None, max_new_tokens=2)

    def test_whisper_processor_masks_two_and_eight_seconds(self) -> None:
        from transformers import WhisperProcessor
        spec = json.loads(worker.PINS.read_text())["models"]["whisper_multilingual"]
        snapshot = worker._snapshot("whisper_multilingual", spec)
        processor = WhisperProcessor.from_pretrained(snapshot, local_files_only=True)
        for seconds, active in ((2, 200), (8, 800)):
            pcm = np.zeros(seconds * 16000, dtype=np.float32)
            processed = processor(pcm, sampling_rate=16000, return_tensors="pt",
                                  return_attention_mask=True)
            geometry = worker.whisper_geometry(processed.attention_mask, len(pcm),
                                                processed.input_features.shape[-1])
            self.assertEqual(geometry["active_mel_frames"], active)
            self.assertEqual(geometry["nominal_active_encoder_rows"], active // 2)
            audit.check_geometry(geometry, "whisper_multilingual", len(pcm))
            changed = copy.deepcopy(geometry)
            changed["attention_mask"][active] = 1
            with self.assertRaisesRegex(ValueError, "mask"):
                audit.check_geometry(changed, "whisper_multilingual", len(pcm))

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

    def test_saved_book_accepts_complete_fake_and_rejects_tampering(self) -> None:
        cases = worker.trusted_cases()
        stages = worker.WAV_STAGES

        def fake_case(case, repeat=False):
            name, frames = case["case_id"], case["frames"]
            length = audit.ctc_length(frames)
            traces = {stage: {
                "shape": shape, "time_axis": axis, "dtype": "float32",
                "sha256": "0" * 64, "row_l2": [0.0] * shape[axis]}
                for stage in stages
                for shape, axis in [audit.expected_stage_shape(stage, "wav2vec2", frames)]}
            return {"type": "case", "run_id": name + ("__repeat" if repeat else ""),
                    "case_id": name, "repeat": repeat, "status": "complete",
                    "frames": frames, "wav_sha256": case["wav_sha256"],
                    "pcm_sha256": case["pcm_sha256"], "elapsed_seconds": 0.01,
                    "input_geometry": worker.wav_geometry(frames, length),
                    "capture_counts": {"ctc": {stage: 1 for stage in stages}},
                    "traces": {"ctc": traces},
                    "output": {"frame_ids": [0] * length,
                               "collapsed_nonblank_ids": [], "blank_id": 0,
                               "logits_sha256": "0" * 64,
                               "transcript": "", "lexical_nonempty": False,
                               "unicode_scripts": [],
                               "termination": "complete_ctc_frames"}}

        def fake_diff(case):
            frames = case["frames"]
            differences = {stage: {"whole_l2": 0.0, "nonzero_rows": 0,
                                   "row_l2": [0.0] * audit.expected_stage_shape(
                                       stage, "wav2vec2", frames)[0][
                                           audit.expected_stage_shape(stage, "wav2vec2", frames)[1]]}
                           for stage in stages}
            return {"type": "silence_difference", "case_id": case["case_id"],
                    "donor_case_id": case["case_id"][:3] + "_silence",
                    "differences": {"ctc": differences}}

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            attempt = root / "attempt"
            attempt.mkdir()
            receipt, review = root / "preflight.json", root / "review.json"
            receipt.write_text("{}")
            review.write_text("{}")
            (attempt / "parent_result.json").write_text(json.dumps({
                "schema": "nonspeech_descriptive_parent_result_v1",
                "status": "worker_exited_pending_audit", "model": "wav2vec2",
                "exit_code": 0, "limit_reason": None, "host_after_ok": True}))
            (attempt / "parent_start.json").write_text(json.dumps({
                "model": "wav2vec2", "review_sha256": worker.sha_file(review),
                "preflight_sha256": worker.sha_file(receipt)}))
            for label in ("before", "after"):
                packed = gzip.compress(b"fake power log", mtime=0)
                filename = f"host_{label}_power_log.txt.gz"
                (attempt / filename).write_bytes(packed)
                (attempt / f"host_{label}.json").write_text(json.dumps({
                    "power_log_archive": {"file": filename,
                        "compressed_sha256": worker.sha_bytes(packed),
                        "raw_sha256": worker.sha_bytes(b"fake power log")}}))
            book = [{"type": "start", "schema": "nonspeech_descriptive_v1",
                     "model": "wav2vec2", "source_identity": worker.source_identity(),
                     "manifest_sha256": worker.MANIFEST_SHA256}]
            pending = []
            ready = set()
            for case in cases:
                book.append(fake_case(case))
                name = case["case_id"]
                if name.endswith("_silence"):
                    ready.add(name[:3])
                    for earlier in pending[:]:
                        if earlier["case_id"][:3] == name[:3]:
                            book.append(fake_diff(earlier))
                            pending.remove(earlier)
                elif name[:3] in ready:
                    book.append(fake_diff(case))
                else:
                    pending.append(case)
            self.assertFalse(pending)
            for case in cases:
                if case["case_id"].endswith("_silence"):
                    book.append(fake_case(case, repeat=True))
            book.append({"type": "complete", "status": "complete",
                         "planned_cases": 20, "planned_repeats": 2})
            self.assertEqual(len(book), 42)

            def write_book(value):
                (attempt / "records.jsonl").write_text(
                    "".join(worker.canonical(row) + "\n" for row in value))

            with mock.patch.object(worker, "require_independent_go", return_value={}):
                write_book(book)
                self.assertEqual(audit.audit_attempt(attempt, receipt, review)["status"],
                                 "PASS_STRUCTURAL")
                mutations = [
                    ("capture stage/count", lambda b: b[1]["capture_counts"]["ctc"].__setitem__(
                        "logits", 0)),
                    ("trace metric invalid", lambda b: b[1]["traces"]["ctc"][
                        "logits"].__setitem__("sha256", "z" * 64)),
                    ("trace metric invalid", lambda b: b[1]["traces"]["ctc"][
                        "logits"].__setitem__("shape", [1, 2, 3])),
                    ("trace metric invalid", lambda b: b[1]["traces"]["ctc"][
                        "logits"].__setitem__("time_axis", 2)),
                    ("all-active/CTC geometry", lambda b: b[1]["input_geometry"].__setitem__(
                        "active_ctc_frames", 1)),
                    ("inventory", lambda b: b.insert(-1, {"type": "unknown"})),
                ]
                for expected_error, mutate in mutations:
                    altered = copy.deepcopy(book)
                    mutate(altered)
                    write_book(altered)
                    with self.subTest(expected_error=expected_error):
                        with self.assertRaisesRegex(ValueError, expected_error):
                            audit.audit_attempt(attempt, receipt, review)

    def test_parent_wall_cap_kills_fake_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = parent.run_bounded(["/bin/sleep", "2"], Path(temp), seconds=0.2)
            self.assertEqual(result["limit_reason"], "wall_cap")
            self.assertNotEqual(result["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
