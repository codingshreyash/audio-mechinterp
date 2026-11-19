"""No-candidate-model tests for human speech ASR and listening source contracts."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from . import asr_proxy_saved_audit_v1 as saved
from . import asr_proxy_worker_v1 as worker
from . import blinded_listening_v1 as blind


class HumanSourceContracts(unittest.TestCase):
    def test_asr_denominator_and_model_pin(self) -> None:
        admission = worker.admission()
        self.assertEqual(len(admission["plan"]), 16)
        self.assertEqual(admission["model_revision"],
                         "22aad52d435eb6dbaf354bdad9b0da84ce7d6156")
        self.assertEqual([p["condition"] for p in admission["plan"][:4]],
                         ["base", "z32000", "n32000_s0", "n32000_s1"])

    def test_saved_sidecar_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "01_logits.npy"
            ref = worker.save_npy(path, np.arange(12, dtype=np.float32).reshape(3, 4))
            self.assertTrue(np.array_equal(saved.load_ref(directory, ref, (3, 4)),
                                           np.arange(12, dtype=np.float32).reshape(3, 4)))
            with path.open("wb") as out:
                np.save(out, np.zeros((3, 4), dtype=np.float32), allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "sidecar hash mismatch"):
                saved.load_ref(directory, ref, (3, 4))

    def test_blind_bundle_and_third_only_on_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            mapping = blind.build(bundle)
            self.assertEqual(blind.audit_bundle(bundle)["copied_wavs"], 32)
            reviews = {}
            for label in ("A", "B"):
                rows = mapping["queues"][label]["mapping"]
                records = [{"opaque_id": r["opaque_id"], "wav_sha256": r["wav_sha256"],
                            "reviewer_id": label + "_fake", "submitted_at_utc": "fake_fixture",
                            "verbatim_transcript": "the same words", "clarity": "clear",
                            "tail_audible": "no", "notes": ""} for r in rows]
                if label == "B":
                    records[0]["verbatim_transcript"] = "different words"
                path = root / f"{label}.json"
                path.write_text(json.dumps({"schema": "human_audio_blind_review_v1",
                                            "queue_sha256": mapping["queues"][label]["queue_sha256"],
                                            "bundle_binding": mapping["bundle_binding"],
                                            "planned_clips": 16, "reviewer_id": label + "_fake",
                                            "records": records}))
                reviews[label] = path
            third = blind.build_third(bundle, reviews["A"], reviews["B"], root / "third")
            self.assertEqual(third["disagreements"], 1)
            self.assertEqual(len(third["reviewer_C"]["mapping"]), 1)
            changed = json.loads(reviews["A"].read_text())
            changed["records"].pop()
            reviews["A"].write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "schema/census"):
                blind.validate_review(reviews["A"], "A", mapping)


if __name__ == "__main__":
    unittest.main()
