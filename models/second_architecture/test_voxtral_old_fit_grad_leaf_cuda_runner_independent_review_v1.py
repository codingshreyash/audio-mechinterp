"""Independent negative CPU checks for the prospective CUDA worker.

These tests intentionally fail against runner commit 5663b94. They preserve
source-contract gaps for correction before an executable freeze or GPU load.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import voxtral_old_fit_grad_leaf_cuda_runner_v1 as runner
from test_voxtral_old_fit_handoff_replay_stage_b_v1 import fake_model_and_batch


class IndependentRunnerNegativeTests(unittest.TestCase):
    def test_claimed_preflight_failure_writes_error_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "claimed_attempt"
            admission = {
                "environment_receipt_sha256": "a" * 64,
                "checkpoint_tree_sha256": "b" * 64,
                "input_bundle_path": str(Path(temporary) / "missing.pt"),
                "input_bundle_sha256": runner.BEAVER_ARCHIVE_SHA,
            }
            with patch.object(runner, "checked_beaver_archive",
                              side_effect=ValueError("archived input invalid")):
                with self.assertRaisesRegex(ValueError, "archived input invalid"):
                    runner.run_case(output, case_index=0,
                                    attempt_id="claimed_attempt",
                                    freeze_sha256="c" * 64,
                                    admission=admission)
            self.assertTrue((output / "error.json").is_file(),
                            "A claimed failed worker needs its error receipt")

    def test_nonfinite_captured_lm_terminal_rejected(self):
        model, batch = fake_model_and_batch()
        model.eval()
        source = runner.passive_forward(model, batch, 0, device="cpu")["sources"]["encoder31"]
        original_hooks = runner.language_hooks

        def poisoned_hooks(current_model):
            hits, handles = original_hooks(current_model)

            def poison_saved_lm0(_module, _args, output):
                hits["lm0"][0] = torch.full_like(output, float("nan"))

            handles.append(current_model.model.language_model.layers[0].register_forward_hook(
                poison_saved_lm0))
            return hits, handles

        with patch.object(runner, "language_hooks", side_effect=poisoned_hooks):
            with self.assertRaisesRegex(ValueError, "LM.*(finite|finiteness)"):
                runner.full_forward(model, batch, site="encoder31",
                                    full_source=source, mode="N0",
                                    ordinal=0, repeat=0, device="cpu")


if __name__ == "__main__":
    unittest.main()
