"""No-checkpoint tests for the held causal worker and saved-output auditor."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack
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
                logits_file = output / rows[3]["logits"]["path"]
                with logits_file.open("ab") as stream:
                    stream.write(b"mutation")
                with self.assertRaisesRegex(ValueError, "sidecar bytes differ"):
                    parent.audit_worker_output(output)


if __name__ == "__main__":
    unittest.main()
