"""Independent no-model checks for Wav2Vec2 saved-baseline auditing."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from models.second_architecture import wav2vec2_fit_baseline_audit_v1 as audit


class Wav2Vec2BaselineAuditTests(unittest.TestCase):
    def test_trusted_fit_inputs_and_normalization_locality(self) -> None:
        cases = audit.trusted_inputs()
        self.assertEqual(len(cases), 8)
        for pair_id, words in audit.PAIRS.items():
            for engine in audit.ENGINES:
                a = cases[audit.case_key(pair_id, engine, words[0])]
                b = cases[audit.case_key(pair_id, engine, words[1])]
                start, end = a["slot"]
                outside = np.r_[0:start, end:a["frames"]]
                self.assertTrue(np.array_equal(a["pcm"][outside], b["pcm"][outside]))
                stock_a, stock_b = (audit.expected_input(c, "S", cases) for c in (a, b))
                common_a, common_b = (audit.expected_input(c, "C", cases) for c in (a, b))
                self.assertGreater(float(np.max(np.abs(stock_a[outside] - stock_b[outside]))), 0)
                self.assertTrue(np.array_equal(common_a[outside], common_b[outside]))

    def test_stage_geometry_and_windows(self) -> None:
        self.assertEqual(audit.conv_length(86876), 271)
        self.assertEqual(audit.conv1_length(86876), 17374)
        self.assertEqual(audit.stage_shape("first_conv_raw", 86876), (1, 512, 17374))
        self.assertEqual(audit.stage_shape("logits", 86876), (1, 271, 32))
        slot = (30527, 49847)
        self.assertEqual(audit.nominal_window(slot, "input_values"), slot)
        self.assertEqual(audit.nominal_window(slot, "first_conv_raw"), (6104, 9970))
        self.assertEqual(audit.nominal_window(slot, "post_block11"), (95, 156))

    def test_trace_vector_axes(self) -> None:
        a = np.zeros((1, 2, 3), dtype=np.float32)
        b = np.zeros_like(a)
        b[0, :, 1] = (3, 4)
        l2, mx = audit.trace_vectors(a, b, "first_conv_raw")
        self.assertEqual(l2.tolist(), [0.0, 5.0, 0.0])
        self.assertEqual(mx.tolist(), [0.0, 4.0, 0.0])
        x = np.zeros((1, 3, 2), dtype=np.float32)
        y = np.zeros_like(x)
        y[0, 1] = (3, 4)
        l2, mx = audit.trace_vectors(x, y, "post_block0")
        self.assertEqual(l2.tolist(), [0.0, 5.0, 0.0])
        self.assertEqual(mx.tolist(), [0.0, 4.0, 0.0])

    def test_checked_array_rejects_tamper_shape_dtype_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture_dir = root / "captures"
            capture_dir.mkdir()
            path = capture_dir / "ok.npy"
            np.save(path, np.ones((1, 2), dtype=np.float32), allow_pickle=False)
            digest = audit.sha(path)
            self.assertEqual(audit.checked_array(root, "captures/ok.npy", digest, (1, 2)).shape,
                             (1, 2))
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                audit.checked_array(root, "captures/ok.npy", digest, (2, 1))
            with self.assertRaisesRegex(ValueError, "SHA"):
                audit.checked_array(root, "captures/ok.npy", "0" * 64, (1, 2))
            link = capture_dir / "link.npy"
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "path invalid"):
                audit.checked_array(root, "captures/link.npy", digest, (1, 2))
            np.save(path, np.ones((1, 2), dtype=np.float64), allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                audit.checked_array(root, "captures/ok.npy", audit.sha(path), (1, 2))

    def test_records_reject_duplicate_keys_and_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "records.jsonl"
            path.write_text('{"type":"preflight","type":"complete"}\n')
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                audit.read_records(path)
            path.write_text('{"type":"preflight"}')
            with self.assertRaisesRegex(ValueError, "truncated"):
                audit.read_records(path)

    def test_normalized_word_eligibility_semantics(self) -> None:
        self.assertEqual(audit.normalized_words("The word is... SILVER!"),
                         ["the", "word", "is", "silver"])
        self.assertNotEqual(audit.normalized_words("plate plate" ).count("plate"), 1)

    def test_fake_baseline_detects_source_ctc_and_capture_corruption(self) -> None:
        class Decoder:
            def batch_decode(self, batches):  # type: ignore[no-untyped-def]
                return ["SILVER" if batches == [[1, 2]] else "WRONG"]

        case_id = "color_05_piper_silver"
        case = {"pair_id": "color_05", "engine": "piper", "word": "silver",
                "wav_sha256": "a" * 64, "pcm_sha256": "b" * 64,
                "frames": 3, "slot": (1, 2), "pcm": np.zeros(3, dtype=np.int16)}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "captures").mkdir()
            hashes, shapes = {}, {}
            def fake_shape(stage, _frames):  # type: ignore[no-untyped-def]
                if stage == "input_values":
                    return (1, 3)
                if stage == "logits":
                    return (1, 2, 32)
                return (1, 2, 2)

            for stage in audit.STAGES:
                value = np.zeros(fake_shape(stage, 3), dtype=np.float32)
                if stage == "logits":
                    value[0, 0, 1] = 1.0
                    value[0, 1, 2] = 1.0
                path = output / "captures" / f"S_{case_id}_{stage}.npy"
                np.save(path, value, allow_pickle=False)
                hashes[stage], shapes[stage] = audit.sha(path), list(value.shape)
            row = {"type": "baseline", "schema": "wav2vec2_fit_baseline_run_v1",
                   "case_key": case_id,
                   "case": {"pair_id": "color_05", "engine": "piper", "word": "silver"},
                   "pair_id": "color_05", "engine": "piper", "word": "silver",
                   "wav_sha256": case["wav_sha256"], "pcm_sha256": case["pcm_sha256"],
                   "mode": "S", "common_mean": None, "common_variance": None,
                   "capture_sha256": hashes, "capture_shapes": shapes,
                   "input_sha256": hashes["input_values"],
                   "logits_sha256": hashes["logits"], "logits_shape": shapes["logits"],
                   "argmax_ids": [1, 2], "transcript": "SILVER",
                   "normalized_words": ["silver"], "eligible": True,
                   "hook_counts": {stage: 1 for stage in audit.STAGES},
                   "elapsed_seconds": 1.0, "rss_bytes": 1000}
            with patch.object(audit, "stage_shape", fake_shape), \
                 patch.object(audit, "expected_input", return_value=np.zeros(3, dtype=np.float32)):
                self.assertTrue(audit.validate_baseline(row, output, {case_id: case}, Decoder())["eligible"])
                for field, bad_value, match in (
                    ("pcm_sha256", "0" * 64, "source identity"),
                    ("transcript", "TEAL", "CTC transcript"),
                    ("argmax_ids", [2, 1], "CTC argmax"),
                    ("eligible", False, "eligibility"),
                ):
                    changed = copy.deepcopy(row)
                    changed[field] = bad_value
                    with self.assertRaisesRegex(ValueError, match):
                        audit.validate_baseline(changed, output, {case_id: case}, Decoder())
                changed = copy.deepcopy(row)
                changed["capture_sha256"]["logits"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                    audit.validate_baseline(changed, output, {case_id: case}, Decoder())

    def test_incomplete_or_failed_run_cannot_pass_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "records.jsonl").write_text("{}\n")
            for ending, match in (({"type": "complete"}, "26-row"),
                                  ({"type": "failure", "schema": "wav2vec2_fit_baseline_run_v1",
                                    "error_type": "RuntimeError", "error": "budget",
                                    "computed_output": None}, "failed attempt")):
                rows = [{"type": "preflight"}, ending]
                with patch.object(audit, "trusted_inputs", return_value={}), \
                     patch.object(audit, "read_records", return_value=rows), \
                     patch.object(audit, "audit_preflight", return_value={}):
                    with self.assertRaisesRegex(ValueError, match):
                        audit.audit(root, root / "preflight.json", "0" * 64)

    def test_fake_trace_rejects_rehashed_numeric_corruption(self) -> None:
        pair_id, engine, mode = "color_05", "piper", "C"
        words = audit.PAIRS[pair_id]
        keys = [audit.case_key(pair_id, engine, word) for word in words]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "captures").mkdir()
            (output / "traces").mkdir()
            hashes = [{}, {}]
            stages = {}
            for stage in audit.STAGES:
                shape = ((1, 3) if stage == "input_values" else
                         (1, 2, 3) if stage in {"first_conv_raw", "first_conv_norm", "final_feature_conv"}
                         else (1, 3, 2))
                values = [np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)]
                if stage == "input_values":
                    values[1][0, 1] = 1.0
                elif stage in {"first_conv_raw", "first_conv_norm", "final_feature_conv"}:
                    values[1][0, 0, 1] = 1.0
                else:
                    values[1][0, 1, 0] = 1.0
                for i, key in enumerate(keys):
                    path = output / "captures" / f"{mode}_{key}_{stage}.npy"
                    np.save(path, values[i], allow_pickle=False)
                    hashes[i][stage] = audit.sha(path)
                l2, maxabs = audit.trace_vectors(*values, stage)
                stage_row = {"nominal_window": [1, 2], "first_nonzero_index": 1,
                             "offslot_maxabs": 0.0}
                for kind, trace in (("l2", l2), ("maxabs", maxabs)):
                    path = output / "traces" / f"{mode}_{pair_id}_{engine}_{stage}_{kind}.npy"
                    np.save(path, trace, allow_pickle=False)
                    stage_row[f"{kind}_sha256"] = audit.sha(path)
                stages[stage] = stage_row
            baselines = {(mode, keys[i]): {"capture_sha256": hashes[i]} for i in (0, 1)}
            cases = {keys[0]: {"frames": 3, "slot": (1, 2)}}
            row = {"type": "trace", "schema": "wav2vec2_fit_baseline_run_v1",
                   "pair_id": pair_id, "engine": engine, "mode": mode,
                   "words": list(words), "stages": stages}
            def fake_shape(stage, _frames):  # type: ignore[no-untyped-def]
                if stage == "input_values":
                    return (1, 3)
                if stage in {"first_conv_raw", "first_conv_norm", "final_feature_conv"}:
                    return (1, 2, 3)
                return (1, 3, 2)
            with patch.object(audit, "stage_shape", fake_shape), \
                 patch.object(audit, "nominal_window", return_value=(1, 2)):
                self.assertEqual(audit.validate_trace(row, output, baselines, cases)["mode"], "C")
                path = output / "traces" / f"{mode}_{pair_id}_{engine}_logits_l2.npy"
                altered = np.load(path)
                altered[0] = np.float32(1e-7)
                np.save(path, altered, allow_pickle=False)
                changed = copy.deepcopy(row)
                changed["stages"]["logits"]["l2_sha256"] = audit.sha(path)
                with self.assertRaisesRegex(ValueError, "trace values differ"):
                    audit.validate_trace(changed, output, baselines, cases)


if __name__ == "__main__":
    unittest.main()
