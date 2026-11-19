"""No-weight saved-evidence checks, including suppression-mask tampering."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from models.second_architecture.nonspeech_probe_v1 import padding_channel_audit_v1 as audit
from models.second_architecture.nonspeech_probe_v1 import padding_channel_core_v1 as core
from models.second_architecture.nonspeech_probe_v1 import padding_channel_worker_v1 as worker
from models.second_architecture.nonspeech_probe_v1 import padding_channel_plan_v4 as plan


class AuditFakeTests(unittest.TestCase):
    def test_saved_array_rehashes_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            budget = worker.Budget(root)
            ref = worker.save_array(root, "raw/x.npy", np.zeros(4, np.float32), budget)
            self.assertEqual(audit.saved_array(root, ref).shape, (4,))
            (root / ref["path"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "byte hash"):
                audit.saved_array(root, ref)

    def test_processed_negative_infinity_and_mask_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            budget = worker.Budget(root)
            original = np.array([0.2, -np.inf, 0.0], np.float32)
            ref = worker.save_array(root, "processed/x.npy", original, budget,
                                    processed=True)
            self.assertTrue(np.isneginf(audit.saved_array(root, ref)[1]))
            with self.assertRaisesRegex(ValueError, "suppression mask"):
                core.processed_score_maxabs(original,
                    np.array([0.2, -2.0, 0.0], np.float32))

    def test_first_decoder_evidence_full_prefix_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            budget = worker.Budget(root)
            ref = worker.save_array(root, "decoder/x.npy",
                                    np.zeros((1, 4, 384), np.float32), budget)
            self.assertEqual(audit.saved_array(root, ref).shape, (1, 4, 384))
            with self.assertRaisesRegex(ValueError, "unsafe sidecar path"):
                audit.saved_array(root, {**ref, "path": "../escape.npy"})

    def test_reconstructs_selected_edit_and_rejects_tamper(self):
        row = next(item for item in plan.schedule() if item["case_id"]
                   == "d02_beep_train_low" and item["site"] == "log_mel"
                   and item["arm"] == "RD")
        native = {}
        for case, fill in ((row["case_id"], 0.0), (row["donor_id"], 1.0)):
            for site in plan.SITES:
                for phase in plan.PHASES:
                    native[(case, site, phase)] = np.full(
                        core.expected_shape(site), fill, np.float32)
        hooks = {}
        for phase in plan.PHASES:
            hooks[phase] = {}
            for site in plan.SITES:
                source = native[(row["case_id"], site, phase)]
                if site == row["site"]:
                    edited, detail = core.edit(source, source,
                        native[(row["donor_id"], site, phase)], row, site, phase)
                else:
                    edited = source
                    detail = {"support": [0, 0], "seed": None,
                              "target_l2": 0.0, "realized_l2": 0.0,
                              "cosine": None}
                hooks[phase][site] = {"calls": 1,
                    "native_sha256": core.sha_array(source),
                    "edited_sha256": core.sha_array(edited), **detail}
        item = {"hook": hooks, "hook_counts": {
            phase: {site: 1 for site in plan.SITES} for phase in plan.PHASES}}
        audit._check_hooks(item, row, native)
        item["hook"]["detect"]["log_mel"]["edited_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "reconstruction"):
            audit._check_hooks(item, row, native)


if __name__ == "__main__":
    unittest.main()
