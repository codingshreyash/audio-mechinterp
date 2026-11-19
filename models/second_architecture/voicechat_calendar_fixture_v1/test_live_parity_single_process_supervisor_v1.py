"""Fake-child tests for a single fresh adapter process; no model inference."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from .live_parity_saved_capture_v1 import save_capture
from .live_parity_single_process_supervisor_v1 import launch_adapter_once
from .live_parity_source_plan_v1 import NO_CALL_BYTES, canonical_bytes


class OneProcessSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.paths = {name: self.root / name for name in
                      ("nemo", "checkpoint", "nano", "qa_manifest",
                       "qa_ledger", "qa_receipt", "wav", "output")}
        self.plan = self.root / "plan.json"
        self.freeze = self.root / "freeze.json"
        self.review = self.root / "review.json"
        self.plan.write_bytes(canonical_bytes({"synthetic_test_only": True}))
        self.events: list[str] = []
        self.hashes = {"tools": "0" * 64}
        self.sources = {"worker": "1" * 64}
        self.admission = {
            "status": "fixture_admitted_for_fresh_process_source_only",
            "source_plan_sha256": hashlib.sha256(self.plan.read_bytes()).hexdigest(),
            "freeze_sha256": "2" * 64, "seed": 17,
            "official_sample_receipt_sha256": "3" * 64,
            "input_hashes": self.hashes, "source_sha256": self.sources,
            "caps": {"wall_seconds": 60, "vram_bytes": 1024 * 1024 * 100,
                     "arm_output_bytes": 50000,
                     "total_output_bytes": 100000},
        }

    def _admit(self, **kwargs):
        self.events.append("admit")
        assert kwargs["paths"] == self.paths
        return self.admission

    def _sample(self, _admission):
        self.events.append("sample")
        return "3" * 64

    def _rehash(self, _plan, _paths):
        self.events.append("rehash")
        return dict(self.hashes)

    def _sources(self):
        return dict(self.sources)

    def _execute(self, command, output, caps, environment):
        self.events.append("execute")
        self.assertEqual(command[1], "-c")
        self.assertIn("live_parity_fixture_worker_v1 import main; main()", command[2])
        self.assertEqual(caps["wall_seconds"], 60)
        self.assertIn("PATH", environment)
        arm = output / "adapter"
        arm.mkdir()
        first = {"tokens_text": torch.tensor([[1]])}
        second = {"audio": torch.tensor([0.125])}
        saved = {"arm": "adapter", "trace": {
            "pass1": {"raw_result": first}, "pass2": {"raw_result": second},
            "replay": None, "external_tool_execution": False},
            "calls": [{"decode_audio": False}, {"decode_audio": True}],
            "passive_frames": {"pass1": {"pass_label": "pass1"},
                               "pass2": {"pass_label": "pass2"}}}
        capture = save_capture(saved, arm / "raw_capture", byte_cap=50000,
                               trusted_root=output)
        result = {"status": "fixture_admitted_for_fresh_process_source_only",
                  "arm": "adapter", "process_id": 4242, "seed": 17,
                  "source_plan_sha256": self.admission["source_plan_sha256"],
                  "freeze_sha256": self.admission["freeze_sha256"],
                  "input_hashes": self.hashes, "capture": capture}
        (arm / "arm_result.json").write_bytes(canonical_bytes(result))
        return {"returncode": 0, "process_id": 4242,
                "peak_vram_mib": 10, "elapsed_monotonic_seconds": 1.0,
                "cap_reason": None, "logs_complete": True}

    def _run(self, **extra):
        kwargs = dict(plan_path=self.plan, freeze_path=self.freeze,
                      review_path=self.review, paths=self.paths,
                      freeze_review_key=b"k" * 32, controller_key=b"c" * 32,
                      admit=self._admit, sample_verify=self._sample,
                      execute=self._execute, rehash=self._rehash,
                      source_digest=self._sources)
        kwargs.update(extra)
        return launch_adapter_once(**kwargs)

    def test_default_sample_gate_blocks_before_output_or_child(self):
        kwargs = dict(plan_path=self.plan, freeze_path=self.freeze,
                      review_path=self.review, paths=self.paths,
                      freeze_review_key=b"k" * 32, controller_key=b"c" * 32,
                      admit=self._admit, execute=self._execute,
                      rehash=self._rehash, source_digest=self._sources)
        with self.assertRaisesRegex(RuntimeError, "sample receipt binding not installed"):
            launch_adapter_once(**kwargs)
        self.assertEqual(self.events, ["admit"])
        self.assertFalse(self.paths["output"].exists())

    def test_one_child_persists_raw_capture_and_derived_response(self):
        result = self._run()
        self.assertEqual(result["status"], "adapter_capture_only_no_parity_claim")
        self.assertEqual(result["response_sha256"],
                         hashlib.sha256(NO_CALL_BYTES).hexdigest())
        self.assertEqual((self.paths["output"] / "api_response.json").read_bytes(),
                         NO_CALL_BYTES)
        self.assertEqual(self.events,
                         ["admit", "sample", "rehash", "execute", "rehash"])
        with self.assertRaises(FileExistsError):
            self._run()

    def test_tampered_input_after_child_keeps_run_consumed(self):
        def changed(_plan, _paths):
            self.events.append("rehash")
            return self.hashes if self.events.count("rehash") == 1 else {"tools": "x" * 64}
        with self.assertRaisesRegex(ValueError, "Fixture changed after child"):
            self._run(rehash=changed)
        self.assertTrue(self.paths["output"].exists())
        self.assertFalse((self.paths["output"] / "adapter_capture_receipt.json").exists())

    def test_missing_saved_capture_rejects_successful_fake_child(self):
        def missing(_command, output, _caps, _environment):
            (output / "adapter").mkdir()
            (output / "adapter" / "arm_result.json").write_bytes(canonical_bytes({
                "status": "fixture_admitted_for_fresh_process_source_only",
                "arm": "adapter", "process_id": 4242, "seed": 17,
                "source_plan_sha256": self.admission["source_plan_sha256"],
                "freeze_sha256": self.admission["freeze_sha256"],
                "input_hashes": self.hashes, "capture": {"manifest_sha256": "0" * 64}}))
            return {"returncode": 0, "process_id": 4242,
                    "peak_vram_mib": 10, "elapsed_monotonic_seconds": 1.0,
                    "cap_reason": None, "logs_complete": True}
        with self.assertRaisesRegex(ValueError, "saved capture directory absent"):
            self._run(execute=missing)


if __name__ == "__main__":
    unittest.main()
