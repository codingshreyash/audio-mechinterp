"""CPU synthetic and tamper tests for independent Stage-A saved-data audit."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

import voxtral_old_fit_handoff_stage_a_audit_v1 as audit
import voxtral_old_fit_handoff_native_stage_a_v1 as native
from test_voxtral_old_fit_handoff_native_stage_a_v1 import FakeModel, ROWS


class AuditTests(unittest.TestCase):
    def test_independent_fixed_probes_and_control_scale(self):
        expected = audit.expected_probes()
        candidate = native.fixed_probes()
        self.assertTrue(all(torch.equal(expected[k], candidate[k]) for k in expected))
        self.assertTrue(torch.equal(audit.expected_unit(), native.fixed_lm0_control_unit()))
        source = torch.zeros(3072, dtype=torch.bfloat16)
        ours = audit.choose_scale(source, audit.expected_unit(), 1, .125)
        theirs = native.choose_lm0_control_scale(source, audit.expected_unit(), 1, .125)
        audit.close(ours, theirs)
        self.assertTrue(ours["valid"])

    def test_trace_handoff_and_tampered_source_or_ulp(self):
        model = FakeModel()
        batch = {"input_ids": torch.cat((torch.full((1, 375), 99),
                                        torch.zeros((1, 21), dtype=torch.long)), dim=1),
                 "input_features": torch.zeros((1, 1500, 1280))}
        clean = native.trace_forward(model, batch, torch.float32, ROWS, device="cpu",
                                     archive_full_source=True)["saved"]
        edited = native.trace_forward(model, batch, torch.float32, ROWS, device="cpu",
                                      patch_site="encoder31", direction=torch.ones(1280),
                                      alpha=.1, archive_full_source=True)["saved"]
        audit.validate_trace(clean, dtype=torch.float32, clean=clean, patch_site=None, cycle=0)
        audit.validate_trace(edited, dtype=torch.float32, clean=clean,
                             patch_site="encoder31", cycle=0)
        original_events = edited["hook_events"]
        edited["hook_events"] = {**original_events, "lm29": {"calls": 2,
                                  "shape": [1, 396, 3072], "dtype": "torch.float32"}}
        with self.assertRaisesRegex(ValueError, "hook events"):
            audit.validate_trace(edited, dtype=torch.float32, clean=clean,
                                 patch_site="encoder31", cycle=0)
        edited["hook_events"] = original_events
        original = edited["native_ulp_spacing"]["lm29"].clone()
        edited["native_ulp_spacing"]["lm29"][0] += 1
        with self.assertRaisesRegex(ValueError, "ULP spacing"):
            audit.validate_trace(edited, dtype=torch.float32, clean=clean,
                                 patch_site="encoder31", cycle=0)
        edited["native_ulp_spacing"]["lm29"] = original
        edited["source_full_pre"]["encoder31"][0, 0, 0] += 1
        with self.assertRaises(ValueError):
            audit.validate_trace(edited, dtype=torch.float32, clean=clean,
                                 patch_site="encoder31", cycle=0)

    def test_full_boundary_descriptors_and_each_signed_cycle(self):
        saved = {"projector_full": torch.zeros(375, 3072),
                 "lm_input_full": torch.zeros(1, 396, 3072),
                 "lm_terminal": {"lm29": torch.zeros(3072)}}
        saved["projector_full"][100, 0] = 2
        saved["lm_input_full"][0, 100, 0] = 3
        self.assertEqual(audit.full_stage(saved, "projector_word").numel(), 375 * 3072)
        self.assertEqual(audit.full_stage(saved, "lm_input_word").numel(), 396 * 3072)
        self.assertEqual(float(audit.full_stage(saved, "projector_word").sum()), 2)
        self.assertEqual(float(audit.full_stage(saved, "lm_input_word").sum()), 3)
        rows = []
        for cycle in range(3):
            for magnitude in (.00625, .025, .1):
                rows.extend(({"cycle": cycle, "alpha": magnitude,
                              "measured": [float(cycle + 1), float(magnitude)]},
                             {"cycle": cycle, "alpha": -magnitude,
                              "measured": [float(-(cycle + 1)), float(magnitude)]}))
        ours = audit.recompute_odd_even(rows)
        theirs = native.cycle_odd_even(rows)
        audit.close(ours, theirs)
        self.assertEqual(ours["0.1"]["cycles"]["2"]["measured_odd"], [3.0, 0.0])
        ours["0.1"]["cycles"]["2"]["measured_odd"][0] += 1
        with self.assertRaisesRegex(ValueError, "number differs"):
            audit.close(ours, theirs)

    def test_content_addressed_blob_rejects_alias_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            blobs = attempt / "blobs"
            blobs.mkdir()
            data = b"synthetic probe bytes"
            digest = hashlib.sha256(data).hexdigest()
            correct = blobs / f"{digest}.pt"
            correct.write_bytes(data)
            result = {"probes_path": f"blobs/{digest}.pt", "probes_sha256": digest}
            self.assertEqual(audit.blob_path(attempt, result, "probes"), correct)
            result["probes_path"] = "blobs/alias.pt"
            with self.assertRaisesRegex(ValueError, "descriptor"):
                audit.blob_path(attempt, result, "probes")
            result["probes_path"] = f"blobs/{digest}.pt"
            correct.write_bytes(data + b"x")
            with self.assertRaisesRegex(ValueError, "bytes"):
                audit.blob_path(attempt, result, "probes")

    def test_layout_rejects_unplanned_fp32_parameter(self):
        rows = [{"name": f"layer.{i}", "dtype": "torch.bfloat16", "device": "mps",
                 "shape": [1], "numel": 1} for i in range(761)]
        rows.append({"name": "model.audio_tower.embed_positions.weight",
                     "dtype": "torch.float32", "device": "mps", "shape": [1500, 1280],
                     "numel": 1500 * 1280})
        layout = {"valid": True, "arm": "bf16", "named_parameter_count": 762,
                  "parameters": rows,
                  "dtype_device_counts": {"torch.bfloat16/mps": 761, "torch.float32/mps": 1}}
        audit.validate_layout(layout, "bf16")
        rows[0]["dtype"] = "torch.float32"
        with self.assertRaises(ValueError):
            audit.validate_layout(layout, "bf16")

    def test_missing_progress_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            path.write_text(json.dumps({"phase": "attempt_started", "arm": "bf16"}) + "\n")
            with self.assertRaisesRegex(ValueError, "start/end"):
                audit.validate_progress(path, "bf16")


if __name__ == "__main__":
    unittest.main()
