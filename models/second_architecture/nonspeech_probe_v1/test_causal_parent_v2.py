"""No-weight parent and saved-audit helper tests."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import causal_audit_v2 as audit
from . import causal_parent_v2 as parent
from . import causal_plan_v1 as plan
from . import causal_worker_v2 as worker


class ParentTests(unittest.TestCase):
    def test_host_probe_follows_launch_start_and_precedes_worker_exit(self) -> None:
        started = datetime(2026, 11, 19, tzinfo=timezone.utc)
        pre = started + timedelta(seconds=3)
        exited = pre + timedelta(seconds=15)
        post = exited + timedelta(seconds=10)
        self.assertTrue(audit.valid_host_timing(started, pre, exited, post))
        self.assertFalse(audit.valid_host_timing(pre, started, exited, post))
        self.assertFalse(audit.valid_host_timing(started, pre, exited,
                                                exited + timedelta(seconds=121)))

    @staticmethod
    def _save_ref(root: Path, name: str, value: np.ndarray) -> dict:
        path = root / name
        np.save(path, value, allow_pickle=False)
        return {"path": name, "file_sha256": worker.sha_file(path),
                "tensor_sha256": worker.sha_bytes(value.tobytes()),
                "shape": list(value.shape), "dtype": "float32",
                "bytes": path.stat().st_size}

    def test_full_and_partial_progress_census(self) -> None:
        names = parent.expected_names()
        self.assertEqual(names, [worker.call_name(row) for row in plan.schedule()])
        self.assertEqual(len(names), len(plan.schedule()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            self.assertEqual(parent.progress_census(path, names)["censored_call_names"], names)
            rows = []
            for i, name in enumerate(names[:7], 1):
                rows += [{"type": "attempt", "call_index": i, "call_name": name},
                         {"type": "complete", "call_index": i, "call_name": name}]
            rows.append({"type": "attempt", "call_index": 8, "call_name": names[7]})
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            found = parent.progress_census(path, names)
            self.assertEqual(found["completed_call_names"], names[:7])
            self.assertEqual(found["failed_attempted_call_name"], names[7])
            self.assertEqual(found["censored_call_names"], names[8:])
            with path.open("a") as stream:
                stream.write(json.dumps({"type": "complete", "call_index": 8,
                                         "call_name": names[7]}) + "\n")
            self.assertIsNone(parent.progress_census(path, names)["failed_attempted_call_name"])

    def test_live_sampled_rss_kill_on_fake_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = parent.run_bounded(
                [sys.executable, "-c", "import time; x=bytearray(64*1024*1024); time.sleep(3)"],
                Path(tmp), seconds=5, rss_bytes=1*1024**2,
                evidence_bytes=64*1024**2)
            self.assertEqual(result["worker_stop_reason"], "rss_limit")
            self.assertGreater(result["worker_rss_samples"], 0)
            self.assertGreaterEqual(result["worker_peak_tree_rss_bytes"], 1*1024**2)

    def test_sidecar_byte_and_tensor_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "x.npy"
            value = np.arange(12, dtype=np.float32).reshape(1, 4, 3)
            np.save(path, value, allow_pickle=False)
            ref = {"path": "x.npy", "file_sha256": worker.sha_file(path),
                   "tensor_sha256": worker.sha_bytes(value.tobytes()),
                   "shape": [1, 4, 3], "dtype": "float32", "bytes": path.stat().st_size}
            self.assertTrue(np.array_equal(audit.saved_array(root, ref), value))
            with path.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaises(ValueError):
                audit.saved_array(root, ref)

    def test_random_edit_reconstruction_is_deterministic(self) -> None:
        source = np.zeros((1, 4, 3), dtype=np.float32)
        donor = np.ones_like(source)
        row = {"arm": "random_1_both", "active_rows": 2,
               "seeds": {"detect": 17, "transcribe": 19}}
        left, support, seed = audit.expected_edit(source, donor, row, "detect")
        right, _, _ = audit.expected_edit(source, donor, row, "detect")
        different, _, _ = audit.expected_edit(source, donor, row, "transcribe")
        self.assertEqual(support, [0, 1500])
        self.assertEqual(seed, 17)
        self.assertTrue(np.array_equal(left, right))
        self.assertFalse(np.array_equal(left, different))
        self.assertAlmostEqual(float(np.linalg.norm(left.astype(np.float64))),
                               float(np.linalg.norm(donor.astype(np.float64))), places=5)

    def test_saved_output_requires_complete_donor_raw_vectors(self) -> None:
        class Tokenizer:
            eos_token_id = 50257

            @staticmethod
            def decode(_ids, skip_special_tokens=False):
                return "x" if skip_special_tokens else "<tokens>x"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = np.zeros(51865, dtype=np.float32)
            raw[291] = 1.0
            hidden = np.zeros((1, 4, 384), dtype=np.float32)
            first = self._save_ref(root, "first.npy", raw)
            state = self._save_ref(root, "state.npy", hidden)
            second = self._save_ref(root, "second.npy", raw)
            output = {
                "sequence_ids": [50258, 50259, 50359, 50363, 291, 50257],
                "prefix_ids": [50258, 50259, 50359, 50363],
                "generated_ids": [291, 50257], "actual_generation_steps": 2,
                "eos_seen": True, "termination": "eos", "censored": False,
                "transcript": "x", "text_with_special_tokens": "<tokens>x",
                "score_rows": [
                    {"token_id": 291, "top_id": 291,
                     "raw_top_id": 291, "raw_selected_logit": 1.0,
                     "raw_logits_sha256": first["tensor_sha256"]},
                    {"token_id": 50257, "top_id": 50257,
                     "raw_top_id": 291, "raw_selected_logit": 0.0,
                     "raw_logits_sha256": second["tensor_sha256"]}],
                "language_detection": None}
            row = {"arm": "no_hook", "save_full_raw_logits_each_step": True}
            sidecars = {"first_raw": first, "first_decoder_state": state,
                        "full_raw": [first, second]}
            ids, vectors = audit._validate_output(output, sidecars, row, root,
                                                   Tokenizer(), {"<|en|>": 50259}, set())
            self.assertEqual(ids, output["sequence_ids"])
            self.assertEqual(len(vectors), 2)
            sidecars["full_raw"] = [first]
            with self.assertRaisesRegex(ValueError, "full raw logit steps missing"):
                audit._validate_output(output, sidecars, row, root, Tokenizer(),
                                       {"<|en|>": 50259}, set())

    def test_selected_phase_hook_edit_reconstructed_and_tamper_rejected(self) -> None:
        native = np.zeros((1, 1500, 384), dtype=np.float32)
        donor = np.ones_like(native)
        sources = {(case, site, phase): native if case == "d02_beep_train_low" else donor
                   for case in ("d02_beep_train_low", "d02_silence")
                   for site in audit.SITES for phase in audit.PHASES}
        row_plan = {"stage": "A", "case_id": "d02_beep_train_low",
                    "donor_id": "d02_silence", "arm": "donor_both",
                    "active_rows": 100}
        native_hash = worker.sha_bytes(native.tobytes())
        donor_hash = worker.sha_bytes(donor.tobytes())
        donor_norm = float(np.linalg.norm((donor-native).astype(np.float64).ravel()))
        hook = {phase: {
            "pre_block0": {"hook_calls": 1, "native_sha256": native_hash,
                           "edited_sha256": native_hash, "edit_l2": 0.0,
                           "donor_delta_l2": donor_norm, "support_rows": [0, 0],
                           "seed": None},
            "final_encoder": {"hook_calls": 1, "native_sha256": native_hash,
                              "edited_sha256": donor_hash, "edit_l2": donor_norm,
                              "donor_delta_l2": donor_norm,
                              "support_rows": [0, 1500], "seed": None}}
            for phase in audit.PHASES}
        row = {"hook": hook, "native_state_refs": {}}
        audit.audit_hooks(row, row_plan, sources)
        hook["detect"]["final_encoder"]["edited_sha256"] = native_hash
        with self.assertRaisesRegex(ValueError, "selected-site edited tensor"):
            audit.audit_hooks(row, row_plan, sources)


if __name__ == "__main__":
    unittest.main()
