"""CPU-only negative checks for the CUDA launch gate source review.

These expected failures document admission gaps in the reviewed source.  Remove
expectedFailure only after the gate rejects both fixtures.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "voxtral_old_fit_grad_leaf_cuda_launch_gate_v1_review",
    HERE / "voxtral_old_fit_grad_leaf_cuda_launch_gate_v1.py",
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class IndependentGateNegativeTests(unittest.TestCase):
    @unittest.expectedFailure
    def test_quoted_go_inside_hold_review_must_not_admit_source(self) -> None:
        with tempfile.TemporaryDirectory(dir=gate.REPO) as directory:
            source = Path(directory) / "source.py"
            review = Path(directory) / "review.md"
            source.write_text("source\n")
            review.write_text(
                "Decision: HOLD. A future review might say: " + gate.GO_SOURCE + "\n"
            )

            def fake_git(*args: str) -> bytes:
                if args[0] == "show":
                    return source.read_bytes()
                if args[0] == "log":
                    return b"source_commit\n"
                raise AssertionError(args)

            with patch.object(gate, "committed_same"), \
                    patch.object(gate, "first_add", return_value="review_commit"), \
                    patch.object(gate, "git", side_effect=fake_git), \
                    patch.object(gate, "ancestor"):
                with self.assertRaises(ValueError):
                    gate.reviewed_source((source,), review, gate.GO_SOURCE)

    @unittest.expectedFailure
    def test_final_campaign_audit_must_bind_two_distinct_case_and_cell_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempts = ["20261118T190000000000Z", "20261118T190001000000Z"]
            capture = root / attempts[1]
            capture.mkdir()
            audit = {
                "schema": "voxtral_old_fit_grad_leaf_cuda_audit_v1",
                "planned_case_count": 2,
                "planned_forward_count": 66,
                "planned_cell_count": 4,
                "direct_full_comparison_count": 96,
                "cases": [{"case_id": gate.CASE_IDS[0], "status": "audited_saved_data_only"}] * 2,
                "cells": [{"case_id": gate.CASE_IDS[0], "site": "encoder31",
                           "label": "same_pattern"}] * 4,
            }
            gate.put(root / "campaign_audit.json", audit)
            with patch.object(gate, "RUNS", root), \
                    patch.object(gate, "WORKER_RUNS", root / "worker"), \
                    patch.object(gate, "bounded", return_value=(0, 1.0, None)):
                with self.assertRaises(ValueError):
                    gate._audit_campaign(attempts, capture, root / "worker" / attempts[1])


if __name__ == "__main__":
    unittest.main()
