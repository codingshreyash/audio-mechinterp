"""No-checkpoint tests for the held causal worker and saved-output auditor."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np

import acoustic_fit_noise_causal_followup_v1 as causal
import acoustic_fit_noise_causal_parent_v1 as parent
import acoustic_fit_noise_causal_worker_v1 as worker
import wav2vec2_fit_baseline_stage_capture_v1 as source


class FakeProcessor:
    def batch_decode(self, paths):
        return [{1: "ONE", 2: "TWO"}.get(path[0], "OTHER") for path in paths]


def logits_for(code: int) -> np.ndarray:
    result = np.zeros((1, 286, 32), dtype=np.float32)
    result[:, :, code] = 1.0
    return result


class CausalLiveSourceOnlyTest(unittest.TestCase):
    def _minimal_saved_parent(self, root: Path, *, status: str,
                              worker_seconds: float = 1.0,
                              parent_seconds: float = 2.0):
        attempt = root / "attempt"
        attempt.mkdir()
        (attempt / "filler.bin").write_bytes(b"saved evidence")
        worker_go = root / "worker_go.json"
        parent_go = root / "parent_go.json"
        worker_go.write_text("{}\n")
        parent_go.write_text("{}\n")
        binding = {"schema": "fake_source_binding"}
        inventory = parent.base.evidence_inventory(attempt)
        total = sum(item["bytes"] for item in inventory.values())
        result = {
            "schema": "acoustic_fit_noise_causal_parent_result_v1",
            "attempt_path": str(attempt.resolve()), "source_binding": binding,
            "preflight_sha256": parent.pilot.sha_file(worker.PREFLIGHT),
            "worker_go_sha256": parent.pilot.sha_file(worker_go),
            "parent_go_sha256": parent.pilot.sha_file(parent_go),
            "evidence_sha256": inventory,
            "total_output_bytes_before_result": total,
            "status": status, "errors": [], "worker_started": True,
            "worker_returncode": 0, "worker_timed_out": False,
            "worker_elapsed_seconds": worker_seconds,
            "monotonic_seconds": parent_seconds,
        }
        result_file = attempt / "launch_result.json"
        result_file.write_text(parent.pilot.canonical(result) + "\n")
        return attempt, worker_go, parent_go, binding, total, result_file

    def test_total_cap_counts_result_newline_at_boundary(self):
        encoded = (parent.pilot.canonical({"status": "invalid_or_partial"}) + "\n").encode("utf-8")
        with mock.patch.object(parent, "MAX_OUTPUT_BYTES", 100):
            parent._check_total_output_cap(100 - len(encoded), len(encoded))
            with self.assertRaisesRegex(ValueError, "including result"):
                parent._check_total_output_cap(101 - len(encoded), len(encoded))
            with self.assertRaisesRegex(ValueError, "including result"):
                parent._check_total_output_cap(100, len(encoded))

    def test_saved_worker_start_is_contained_by_parent_utc(self):
        result = {"start_utc": datetime.fromtimestamp(100, timezone.utc).isoformat(),
                  "end_utc": datetime.fromtimestamp(102, timezone.utc).isoformat()}
        parent._check_worker_start_containment(result,
                                               {"type": "start", "start_utc": 101.0})
        with self.assertRaisesRegex(ValueError, "outside parent"):
            parent._check_worker_start_containment(result,
                                                   {"type": "start", "start_utc": 102.01})
        with self.assertRaisesRegex(ValueError, "start time invalid"):
            parent._check_worker_start_containment(result,
                                                   {"type": "start", "start_utc": float("nan")})

    def test_worker_audit_rejects_oversize_before_parsing(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "records.jsonl").write_bytes(b"not even json")
            with mock.patch.object(parent, "MAX_OUTPUT_BYTES", 12):
                with self.assertRaisesRegex(ValueError, "exceeds cap"):
                    parent.audit_worker_output(output)

    def test_saved_parent_rejects_result_over_cap_and_worker_over_parent_time(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (attempt, worker_go, parent_go, binding,
             total, result_file) = self._minimal_saved_parent(
                root, status="invalid_or_partial")
            patches = [
                mock.patch.object(parent, "ATTEMPT", attempt),
                mock.patch.object(worker, "WORKER_GO", worker_go),
                mock.patch.object(parent, "PARENT_GO", parent_go),
                mock.patch.object(worker, "verify_worker_go", return_value={}),
                mock.patch.object(parent, "verify_parent_go", return_value={}),
                mock.patch.object(parent, "source_binding", return_value=binding),
            ]
            with ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                with mock.patch.object(parent, "MAX_OUTPUT_BYTES",
                                       total + result_file.stat().st_size - 1):
                    with self.assertRaisesRegex(ValueError, "including result"):
                        parent.audit_saved_attempt(attempt)
                self.assertEqual(parent.audit_saved_attempt(attempt)["status"],
                                 "bound_invalid_or_partial")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (attempt, worker_go, parent_go, binding,
             _total, _result_file) = self._minimal_saved_parent(
                root, status="complete", worker_seconds=2.0,
                parent_seconds=1.0)
            with (mock.patch.object(parent, "ATTEMPT", attempt),
                  mock.patch.object(worker, "WORKER_GO", worker_go),
                  mock.patch.object(parent, "PARENT_GO", parent_go),
                  mock.patch.object(worker, "verify_worker_go", return_value={}),
                  mock.patch.object(parent, "verify_parent_go", return_value={}),
                  mock.patch.object(parent, "source_binding", return_value=binding)):
                with self.assertRaisesRegex(ValueError, "worker outcome invalid"):
                    parent.audit_saved_attempt(attempt)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (attempt, worker_go, parent_go, binding,
             _total, _result_file) = self._minimal_saved_parent(
                root, status="complete", worker_seconds=1.0,
                parent_seconds=2.0)
            with (mock.patch.object(parent, "ATTEMPT", attempt),
                  mock.patch.object(worker, "WORKER_GO", worker_go),
                  mock.patch.object(parent, "PARENT_GO", parent_go),
                  mock.patch.object(worker, "verify_worker_go", return_value={}),
                  mock.patch.object(parent, "verify_parent_go", return_value={}),
                  mock.patch.object(parent, "source_binding", return_value=binding)):
                with self.assertRaisesRegex(ValueError, "evidence files missing"):
                    parent.audit_saved_attempt(attempt)

    def test_public_go_absent_and_parent_does_not_create_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root / "attempt"
            missing_worker_go = root / "missing_worker_go.json"
            missing_parent_go = root / "missing_parent_go.json"
            with (mock.patch.object(worker, "WORKER_GO", missing_worker_go),
                  mock.patch.object(parent, "PARENT_GO", missing_parent_go),
                  mock.patch.object(parent, "ATTEMPT", candidate)):
                with self.assertRaisesRegex(ValueError, "worker HOLD"):
                    worker.verify_worker_go()
                with self.assertRaisesRegex(ValueError, "worker HOLD"):
                    parent.execute(candidate)
                with mock.patch.object(worker, "verify_worker_go", return_value={}):
                    with self.assertRaisesRegex(ValueError, "parent HOLD"):
                        parent.verify_parent_go()
            self.assertFalse(candidate.exists())

    def test_fake_matrix_and_saved_audit_with_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "worker_output"
            fake_go = root / "worker_go.json"
            fake_go.write_text("{}\n")
            clean_pcm = np.array([1], dtype=np.int16)
            noise_pcm = np.array([2], dtype=np.int16)
            pcm = {"clean": clean_pcm, "word_noise": noise_pcm}
            clean_state = np.zeros(causal.STATE_SHAPE, dtype=np.float32)
            noise_state = np.ones(causal.STATE_SHAPE, dtype=np.float32)
            states = {"clean": clean_state, "word_noise": noise_state}
            logits = {"clean": logits_for(1), "word_noise": logits_for(2)}
            processor = FakeProcessor()
            selected = {
                arm: {"output": worker._decode(processor, logits[arm]),
                      "metrics": {"logits": {"sha256": causal.array_sha(logits[arm])}}}
                for arm in pcm
            }

            def normalized_input(value, **_kwargs):
                return np.array([value[0]], dtype=np.float32)

            def baseline_forward(_model, values, _frames):
                arm = "clean" if int(values[0, 0]) == 1 else "word_noise"
                return ({"feature_projection": states[arm], "logits": logits[arm]},
                        {stage: 1 for stage in source.STAGES})

            def patched_forward(_model, values, *, recipient_state, patched_state):
                if np.array_equal(patched_state, clean_state):
                    return logits["clean"], 1
                if np.array_equal(patched_state, noise_state):
                    return logits["word_noise"], 1
                recipient = "clean" if int(values[0, 0]) == 1 else "word_noise"
                self.assertTrue(np.array_equal(recipient_state, states[recipient]))
                return logits[recipient], 1

            patches = [
                mock.patch.object(worker, "WORKER_GO", fake_go),
                mock.patch.object(causal, "PCM_HASHES", {a: causal.array_sha(v) for a, v in pcm.items()}),
                mock.patch.object(causal, "STATE_HASHES", {a: causal.array_sha(v) for a, v in states.items()}),
                mock.patch.object(causal, "_selected_rows", return_value=selected),
                mock.patch.object(worker.pilot, "stimuli", return_value={causal.CASE: pcm}),
                mock.patch.object(source, "normalized_input", side_effect=normalized_input),
                mock.patch.object(worker.pilot, "wav2vec2_model_forward_v2", side_effect=baseline_forward),
                mock.patch.object(causal, "patched_forward", side_effect=patched_forward),
                mock.patch("transformers.Wav2Vec2Processor.from_pretrained", return_value=processor),
            ]
            with ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                worker.execute_after_admission(output, model=object(), processor=processor)
                audited = parent.audit_worker_output(output)
                self.assertEqual(audited["status"], "complete_worker_evidence_pass")
                rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
                self.assertEqual([r["type"] for r in rows].count("baseline"), 2)
                self.assertEqual([r["type"] for r in rows].count("patch"), 16)
                self.assertTrue(all(r["edit"]["realized_edit_sha256"] for r in rows if r["type"] == "patch"))
                self.assertTrue(all(r["self_logits_exact"] for r in rows if r.get("arm") == "self_W"))
                self.assertTrue(all(r["full_donor_logits_exact"] for r in rows if r.get("arm") == "full_donor"))
                original_records = (output / "records.jsonl").read_text()
                rows[3]["edit"]["realized_edit_sha256"] = "0" * 64
                (output / "records.jsonl").write_text(
                    "\n".join(worker.pilot.canonical(row) for row in rows) + "\n")
                with self.assertRaisesRegex(ValueError, "edit tensor hash"):
                    parent.audit_worker_output(output)
                (output / "records.jsonl").write_text(original_records)
                logits_file = output / rows[3]["logits"]["path"]
                with logits_file.open("ab") as stream:
                    stream.write(b"mutation")
                with self.assertRaisesRegex(ValueError, "sidecar bytes differ"):
                    parent.audit_worker_output(output)


if __name__ == "__main__":
    unittest.main()
