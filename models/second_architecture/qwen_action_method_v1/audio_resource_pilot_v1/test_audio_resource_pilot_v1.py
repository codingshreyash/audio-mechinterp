"""No-weight source and adversarial contract tests for the technical pilot."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import gate, plan, saved_audit, worker


class PilotSourceTests(unittest.TestCase):
    def test_rendered_nonstudy_wav_is_exact(self) -> None:
        row = plan.verify_wav()
        self.assertEqual(row["pcm_samples"], 256000)
        self.assertEqual(row["wav_sha256"], plan.sha(plan.WAV))
        self.assertEqual(plan.WAV.read_bytes(), plan.render())
        self.assertFalse(plan.CASE_ID.startswith(("D_", "H_")))

    def test_go_and_attempt_absent(self) -> None:
        self.assertFalse(plan.ADMISSION.exists())
        self.assertFalse(plan.REVIEW.exists())
        self.assertFalse(plan.GO.exists())
        self.assertFalse(plan.ATTEMPT.exists())

    def test_direct_worker_refuses_before_model_import(self) -> None:
        cmd = [sys.executable, "-m",
            "models.second_architecture.qwen_action_method_v1.audio_resource_pilot_v1.worker",
            "--worker", "--attempt", str(plan.ATTEMPT)]
        result = subprocess.run(cmd, cwd=plan.ROOT, capture_output=True,
                                text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(plan.ATTEMPT.exists())
        self.assertNotIn("transformers", result.stderr.lower())

    def test_missing_go_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            gate.validate_go(live=False)

    def test_tampered_wav_refused(self) -> None:
        with patch.object(plan, "WAV_SHA256", "0" * 64):
            with self.assertRaises(ValueError):
                plan.verify_wav()

    def test_stop_reason_and_full_path_parity(self) -> None:
        eos = 151645
        self.assertEqual(worker.stop_reason([9, eos], eos), "eos")
        self.assertEqual(worker.stop_reason([9] * 12, eos), "length_censored")
        self.assertEqual(worker.stop_reason([eos, 9], eos), "early_eos_malformed")
        self.assertEqual(worker.compare_paths([9, eos], [9, eos], eos)["exact_path_parity"], True)
        with self.assertRaises(ValueError):
            worker.compare_paths([9, eos], [9, 10, eos], eos)
        with self.assertRaises(ValueError):
            worker.stop_reason([9] * 13, eos)

    def test_sidecar_hash_and_shape_rejected(self) -> None:
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            meta = worker.save_array(folder, "probe", np.array([1, 2], dtype=np.float32))
            self.assertTrue(np.array_equal(
                saved_audit.load_sidecar(folder, meta, expected="probe.npy"),
                [1, 2]))
            with self.assertRaises(ValueError):
                saved_audit.load_sidecar(folder, {**meta, "sha256": "0" * 64})
            with self.assertRaises(ValueError):
                saved_audit.load_sidecar(folder, {**meta, "shape": [3]})
            with self.assertRaises(ValueError):
                saved_audit.load_sidecar(folder, {**meta, "path": "../probe.npy"})


if __name__ == "__main__":
    unittest.main()
