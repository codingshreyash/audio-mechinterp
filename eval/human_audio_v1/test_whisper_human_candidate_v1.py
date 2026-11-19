"""No-target-inference tests for the candidate development gate and schedule."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import whisper_human_candidate_gate_v1 as gate
from . import whisper_human_candidate_runner_v1 as runner
from . import whisper_human_preflight_v1 as frozen


class CandidateTests(unittest.TestCase):
    def test_real_plan_has_four_development_pairs_and_fixed_rows(self) -> None:
        plan = frozen.frozen_plan()
        pairs = [pair for pair in plan["comparisons"]
                 if pair["split"] == "development"]
        self.assertEqual(len(pairs), 4)
        for pair in pairs:
            zero, noise = runner._pair_rows(plan, pair)
            self.assertEqual(zero["condition"], "z32000")
            self.assertEqual(zero["wav_sha256"], pair["zero_wav_sha256"])
            self.assertEqual(noise["wav_sha256"], pair["noise_wav_sha256"])
            self.assertEqual(len(pair["boundary_rows"]), 14)
            self.assertEqual(len(pair["displaced_source_rows"]), 14)
            self.assertTrue(set(pair["boundary_rows"]).isdisjoint(
                pair["displaced_source_rows"]))

    def test_missing_go_blocks_target_before_import_or_output(self) -> None:
        with mock.patch.object(runner, "_target") as target, \
             mock.patch.object(runner.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "missing or symlink"):
                runner.run()
            target.assert_not_called()
            launch.assert_not_called()
        self.assertFalse(gate.ATTEMPT.exists())

    def test_external_watchdog_terminates_hung_worker(self) -> None:
        from models.second_architecture.nonspeech_probe_v1 import whisper_tail_causal_host_v1
        process = mock.Mock()
        process.poll.return_value = None
        process.communicate.return_value = ("", "")
        with mock.patch.object(gate, "validate_go", return_value={}), \
             mock.patch.object(whisper_tail_causal_host_v1, "quick_probe",
                               return_value={"external_connected": True}), \
             mock.patch.object(runner.subprocess, "Popen", return_value=process), \
             mock.patch.object(runner.time, "monotonic", side_effect=(0.0, 7201.0)):
            with self.assertRaisesRegex(RuntimeError, "wall_limit"):
                runner.run()
        process.terminate.assert_called_once()

    def test_admission_does_not_accept_fake_human_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            third = folder / "third"
            third.mkdir()
            (third / "ADJUDICATION_MAPPING.json").write_text("{}")
            a, b, receipt = (folder / name for name in ("a.json", "b.json", "receipt.json"))
            for path in (a, b, receipt):
                path.write_text("{}")
            with mock.patch.object(gate, "sources", return_value={}):
                with self.assertRaises((KeyError, ValueError)):
                    gate.admission(third=third, a=a, b=b, c=None, receipt=receipt)

    def test_cli_admission_never_overwrites_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "admission.json"
            output.write_text("already exists")
            with mock.patch.object(gate, "ADMISSION", output), \
                 mock.patch.object(gate, "admission") as build:
                with self.assertRaisesRegex(ValueError, "already exists"):
                    runner.admission_cli(output, output, output, None, output, output)
                build.assert_not_called()

    def test_split_and_decode_are_bound_in_candidate_admission(self) -> None:
        fake = {"status": "HOLD_NO_TARGET_INFERENCE",
                "listening": {"receipt_sha256": "a" * 64, "review_a_sha256": "b" * 64,
                              "review_b_sha256": "c" * 64, "review_c_sha256": None,
                              "adjudication_sha256": "d" * 64,
                              "passed_clips": 16, "clips": 16, "disagreements": 0},
                "local_assets": {"snapshot": "/fake"}, "installed": {"versions": {}}}
        with mock.patch.object(gate, "sources", return_value={"source": "f" * 64}), \
             mock.patch.object(frozen, "readiness", return_value=fake):
            value = gate.admission(third=Path("/tmp/third"), a=Path("/tmp/a"),
                                   b=Path("/tmp/b"), c=None, receipt=Path("/tmp/receipt"))
        self.assertFalse(value["sealed_test_inference"])
        self.assertEqual(value["split"], "development")
        self.assertEqual(value["decode"], gate.DECODE)
        self.assertEqual(len(value["wav_sha256"]), 16)
        self.assertEqual(value["limits"]["max_model_calls"], 106)

    def test_candidate_is_development_only(self) -> None:
        self.assertEqual(gate.ATTEMPT.name, "attempt_001")
        self.assertNotIn("sealed_test", str(gate.ATTEMPT))
        self.assertEqual(gate.DECODE["language"], None)
        self.assertFalse(gate.DECODE["do_sample"])

    def test_encoder_patch_edits_only_selected_rows(self) -> None:
        import numpy as np
        import torch
        native = torch.zeros((1, 1500, 384), dtype=torch.float32)
        donor = np.ones((1, 1500, 384), dtype=np.float32)
        boundary = list(range(100, 114))
        speech = list(range(20, 34))
        for arm, expected_rows in (("boundary", boundary),
                                   ("source_donor", speech),
                                   ("displaced_delta", speech),
                                   ("full_donor", list(range(1500))),
                                   ("self", [])):
            hooks = runner.Hooks(object(), "post_block0", arm, donor,
                                 boundary, speech)
            edited = hooks._capture("post_block0", native)
            self.assertEqual(hooks.counts["transcribe"]["post_block0"], 1)
            observed = np.flatnonzero(edited.numpy()[0].any(axis=1)).tolist()
            self.assertEqual(observed, expected_rows)
            if arm == "displaced_delta":
                self.assertAlmostEqual(hooks.edit_l2, hooks.boundary_delta_l2,
                                       places=5)
        hooks = runner.Hooks(object(), "post_block0", "full_donor", donor,
                             boundary, speech)
        hooks.phase = "detect"
        self.assertIsNone(hooks._capture("post_block0", native))
        self.assertTrue(np.array_equal(hooks.native["detect"]["post_block0"],
                                       hooks.edited["detect"]["post_block0"]))

    def test_censored_equal_path_not_qualified_as_complete(self) -> None:
        a = {"output": {"termination": "length_censored", "sequence_ids": [1, 2]}}
        b = {"output": {"termination": "length_censored", "sequence_ids": [1, 2]}}
        self.assertTrue(runner._same_path(a, b))
        self.assertFalse(runner._complete_path(a, b))


if __name__ == "__main__":
    unittest.main()
