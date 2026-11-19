"""No-weight checks for the source-only Qwen spoken-input candidate."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import audio_candidate_v1 as candidate


class CandidateTests(unittest.TestCase):
    def test_real_source_plan_is_six_development_only_cases(self) -> None:
        rows = candidate.source_plan()
        self.assertEqual(len(rows), 6)
        self.assertEqual([r["case_id"] for r in rows],
                         ["D_L1__piper_en_US_amy_medium",
                          "D_L2__piper_en_US_amy_medium",
                          "D_A1__piper_en_US_amy_medium",
                          "D_A2__piper_en_US_amy_medium",
                          "D_C1__piper_en_US_amy_medium",
                          "D_C2__piper_en_US_amy_medium"])
        self.assertFalse(any(r["case_id"].startswith("H_") for r in rows))

    def test_local_processor_snapshot_assets_match_pinned_manifest(self) -> None:
        snapshot = (Path.home() /
                    ".cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B"
                    / "snapshots" / candidate.load_spec()["model_revision"])
        self.assertEqual(set(candidate.verify_processor_snapshot(snapshot)),
                         set(candidate.PROCESSOR_FILES))
        with self.assertRaisesRegex(ValueError, "snapshot identity"):
            candidate.verify_processor_snapshot(snapshot.parent)

    def test_missing_real_qa_stops_before_processor_or_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            absent = Path(folder) / "absent.json"
            files = candidate.QAFiles(absent, absent, absent, absent, absent)
            with self.assertRaisesRegex(ValueError, "missing or linked"):
                candidate.prepare_development_case(
                    "D_L1__piper_en_US_amy_medium", files)

    def test_processor_provenance_cannot_be_supplied_by_caller(self) -> None:
        # The only preparation API takes a snapshot path and constructs the
        # processor internally after QA; it cannot accept an unrelated object.
        import inspect
        names = inspect.signature(candidate.prepare_development_case).parameters
        self.assertEqual(tuple(names), ("case_id", "files", "snapshot"))
        self.assertNotIn("processor", names)

    def test_source_drift_blocks_processor_construction(self) -> None:
        class ForbiddenProcessor:
            constructed = False

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.constructed = True
                raise AssertionError("processor constructed before source check")

        absent = Path("/does/not/matter.json")
        files = candidate.QAFiles(absent, absent, absent, absent, absent)
        with patch.object(candidate, "reviewed_qa", return_value={"sha256": {}}), \
             patch.object(candidate, "validate_local_source",
                          side_effect=ValueError("source drift")), \
             patch.dict(sys.modules, {"transformers": SimpleNamespace(
                 Qwen2_5OmniProcessor=ForbiddenProcessor)}):
            with self.assertRaisesRegex(ValueError, "source drift"):
                candidate.prepare_development_case("D_L1__piper_en_US_amy_medium",
                                                   files, Path("/unused"))
        self.assertFalse(ForbiddenProcessor.constructed)

    def test_wrong_runtime_blocks_before_processor_import(self) -> None:
        absent = Path("/does/not/matter.json")
        files = candidate.QAFiles(absent, absent, absent, absent, absent)
        with patch.object(candidate, "reviewed_qa", return_value={"sha256": {}}), \
             patch.object(candidate, "QWEN_PYTHON", Path("/different/python")):
            with self.assertRaisesRegex(ValueError, "pinned Qwen runtime"):
                candidate.prepare_development_case("D_L1__piper_en_US_amy_medium",
                                                   files, Path("/unused"))

    def test_cli_missing_qa_stops_before_transformers_import(self) -> None:
        result = subprocess.run(
            (sys.executable, "-m",
             "models.second_architecture.qwen_action_method_v1.audio_candidate_v1",
             "--prepare-development", "D_L1__piper_en_US_amy_medium"),
            cwd=candidate.ROOT, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("real QA exports", result.stderr)

    @staticmethod
    def fake_batch() -> tuple[dict, SimpleNamespace]:
        audio_id = 99
        batch = {"input_ids": np.array([[7] + [audio_id] * 100 + [8]], dtype=np.int64),
                 "attention_mask": np.ones((1, 102), dtype=np.int64),
                 "input_features": np.zeros((1, 128, 500), dtype=np.float32),
                 "feature_attention_mask": np.array([[1] * 400 + [0] * 100,
                                                       ], dtype=np.int64)}
        processor = SimpleNamespace(audio_token_id=audio_id,
                                    feature_extractor=SimpleNamespace(
                                        sampling_rate=16000, hop_length=160))
        return batch, processor

    def test_processor_geometry_and_exact_prompt_last(self) -> None:
        batch, processor = self.fake_batch()
        record = candidate.validate_audio_batch(
            batch, processor, wav_sha256="a" * 64, case_id="D_L1__piper_en_US_amy_medium")
        self.assertEqual(record["projected_row_count"], 100)
        self.assertEqual(record["encoder_row_count"], 200)
        self.assertEqual(record["audio_placeholder_positions"], list(range(1, 101)))
        self.assertEqual(record["prompt_last_position"], 101)
        self.assertEqual(record["mel_frames"], 400)

    def test_processor_rejects_placeholder_and_mask_tampering(self) -> None:
        batch, processor = self.fake_batch()
        batch["input_ids"][0, 50] = 12
        with self.assertRaisesRegex(ValueError, "split audio placeholders"):
            candidate.validate_audio_batch(batch, processor,
                                           wav_sha256="a" * 64, case_id="D_L1")
        batch, processor = self.fake_batch()
        batch["feature_attention_mask"][0, 399] = 0
        batch["feature_attention_mask"][0, 400] = 1
        with self.assertRaisesRegex(ValueError, "feature mask invalid"):
            candidate.validate_audio_batch(batch, processor,
                                           wav_sha256="a" * 64, case_id="D_L1")

    def test_review_receipt_is_exact_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            paths = [base / f"{name}.json" for name in
                     ("A", "B", "attestation", "report", "independent")]
            for path in paths[:3]:
                path.write_text("{}\n")
            report = {"status": candidate.QA_PASS, "case_census": 18,
                      "all_18_initial_pairs_pass": True,
                      "critical_disagreement_count": 0,
                      "third_review_received": False,
                      "cases": [{"initial_pair_pass": True} for _ in range(18)]}
            paths[3].write_text(json.dumps(report) + "\n")
            hashes = {"first_a": candidate.sha(paths[0]),
                      "first_b": candidate.sha(paths[1]),
                      "attestation": candidate.sha(paths[2]),
                      "report": candidate.sha(paths[3]), "third": None,
                      "source_manifest": candidate.sha(candidate.SOURCE / "manifest.json"),
                      "bundle_mapping": candidate.sha(candidate.human_qa_v2.BUNDLE /
                                                      "TECHNICAL_MAPPING.json")}
            review = {"schema": candidate.REVIEW_SCHEMA,
                      "status": "PASS_REAL_HUMAN_QA", "sha256": hashes,
                      "reviewer_distinct_from_first_pair": True,
                      "original_exports_inspected": True}
            paths[4].write_text(json.dumps(review) + "\n")
            files = candidate.QAFiles(*paths)
            with patch.object(candidate, "_tts_payload", return_value=report), \
                 patch.object(candidate, "source_plan", return_value=[
                     {"case_id": "D_L1__piper_en_US_amy_medium"}]):
                admission = candidate.reviewed_qa(files)
                self.assertEqual(admission["status"],
                                 "human_qa_reviewed_source_only_no_model_go")
                self.assertFalse(admission["candidate_model_inference_authorized"])
                paths[4].write_text(json.dumps({**review,
                                                 "original_exports_inspected": False}) + "\n")
                with self.assertRaisesRegex(ValueError, "review receipt"):
                    candidate.reviewed_qa(files)
                paths[4].write_text(json.dumps(review) + "\n")
                report["cases"][0]["initial_pair_pass"] = False
                with self.assertRaisesRegex(ValueError, "QA has not passed"):
                    candidate.reviewed_qa(files)


if __name__ == "__main__":
    unittest.main()
