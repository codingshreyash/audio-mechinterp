"""CPU-only admission tests; temporary reviews are synthetic code-path data."""

from __future__ import annotations

import hashlib
import hmac
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import live_parity_production_admission_v1 as gate
from . import live_parity_source_plan_v1 as plan_module
from .live_parity_source_plan_v1 import (
    ARMS, CASE_ID, PHASE_IDENTITY_FIELDS, VOICE_ID, canonical_bytes,
)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name).resolve()
        self.root = root
        self.paths = {name: root / name for name in
                      ("nemo", "checkpoint", "nano", "qa_manifest",
                       "qa_ledger", "qa_receipt", "wav", "output")}
        self.key = b"k" * 32
        self.controller = b"c" * 32
        self.hashes = {name: "0" * 64 for name in
                       ("selected_wav", "qa_manifest", "qa_ledger",
                        "qa_receipt", "checkpoint_config", "checkpoint_weight",
                        "tools")}
        report = {field: "fake-stable" for field in PHASE_IDENTITY_FIELDS}
        report.update({"qa_manifest_sha256": self.hashes["qa_manifest"],
                       "qa_ledger_sha256": self.hashes["qa_ledger"],
                       "config_sha256": self.hashes["checkpoint_config"],
                       "weight_sha256": self.hashes["checkpoint_weight"],
                       "nano_offline_resolver": {},
                       "nano_preload_environment": "offline_pinned_cache_no_bytecode"})
        self.ready = {"phase": "ready", "started_utc": "2026-11-18T00:00:00Z",
                      "finished_utc": "2026-11-18T00:00:01Z",
                      "reported": {**report,
                                   "status": "local_ready_preflight_passed_no_model_or_network"}}
        self.preload = {"phase": "preload", "started_utc": "2026-11-18T00:00:02Z",
                        "finished_utc": "2026-11-18T00:00:03Z",
                        "reported": {**report,
                                     "status": "local_preload_preflight_passed_no_model_or_network"}}
        self.plan = {
            "schema": "voicechat_live_parity_source_plan_v1",
            "status": "source_plan_only_no_model_work",
            "synthetic_test_injection": False,
            "selected": {"case_id": CASE_ID, "voice_id": VOICE_ID,
                         "wav_path": str(self.paths["wav"]),
                         "wav_sha256": self.hashes["selected_wav"]},
            "source": {"nemo_dir": str(self.paths["nemo"]),
                       "checkpoint_dir": str(self.paths["checkpoint"]),
                       "nano_home": str(self.paths["nano"])},
            "controller_id": "synthetic-test-controller",
            "external_anchor_audit_id": "synthetic-code-path-only",
            "input_hashes": self.hashes, "seed": 17,
            "arm_order": ["official_sample", *ARMS],
            "fresh_process_per_arm": True,
            "external_tool_execution": False,
            "ready": self.ready, "preload": self.preload,
        }
        self.plan_path = root / "plan.json"
        self.freeze_path = root / "freeze.json"
        self.source_path = root / "source-review.json"
        self.review_path = root / "freeze-review.json"
        self._write(self.plan_path, self.plan)
        self.events: list[str] = []
        self._write_reviews()

    @staticmethod
    def _write(path, value):
        path.write_bytes(canonical_bytes(value))

    def _write_reviews(self):
        sources = gate.source_hashes()
        source = {"schema": "voicechat_fixture_source_review_v1",
                  "decision": "GO", "source_sha256": sources,
                  "source_commit": "0" * 40, "reviewer_id": "synthetic-source",
                  "reviewed_utc": "2026-11-18T00:00:04Z"}
        self._write(self.source_path, source)
        source_sha = gate.file_hash(self.source_path)
        expected = {"source_plan_sha256": gate.file_hash(self.plan_path),
                    "input_hashes": self.hashes, "source_sha256": sources,
                    "paths": {key: str(value) for key, value in self.paths.items()},
                    "seed": 17, "arm_order": ["official_sample", *ARMS],
                    "external_anchor_audit_id": "synthetic-code-path-only"}
        freeze = {"schema": "voicechat_fixture_executable_freeze_v1",
                  "status": "externally_frozen_fixture", "run_id": "run_test",
                  "created_utc": "2026-11-18T00:00:05Z",
                  "expected": expected,
                  "caps": {"wall_seconds": 60, "vram_bytes": 1000,
                           "arm_output_bytes": 10000,
                           "total_output_bytes": 50000},
                  "official_sample_receipt_sha256": "3" * 64,
                  "source_review": {"path": str(self.source_path),
                                    "sha256": source_sha}}
        self.paths["output"] = self.paths["output"].parent / "run_test"
        expected["paths"]["output"] = str(self.paths["output"])
        self._write(self.freeze_path, freeze)
        review = {"schema": "voicechat_fixture_freeze_review_v1",
                  "status": "independently_reviewed_executable_freeze",
                  "freeze_sha256": gate.file_hash(self.freeze_path),
                  "source_review_sha256": source_sha,
                  "reviewer_id": "synthetic-freeze",
                  "reviewed_utc": "2026-11-18T00:00:06Z"}
        review["hmac_sha256"] = hmac.new(self.key, canonical_bytes(review),
                                        hashlib.sha256).hexdigest()
        self._write(self.review_path, review)

    def _qa(self, _plan, _paths, _key):
        self.events.append("qa")
        return {"human_eligible": 12}

    def _anchor(self, _plan, _cohort):
        self.events.append("anchor")
        return "synthetic-code-path-only"

    def _preflight(self, _plan, _paths):
        self.events.append("preflight")
        return self.ready, self.preload

    def _rehash(self, _plan, _paths):
        self.events.append("rehash")
        return dict(self.hashes)

    def _admit(self, **extra):
        args = dict(plan_path=self.plan_path, freeze_path=self.freeze_path,
                    review_path=self.review_path, paths=self.paths,
                    freeze_review_key=self.key, controller_key=self.controller,
                    qa_verify=self._qa, ready_preload=self._preflight,
                    rehash=self._rehash,
                    commit_verify=lambda _commit, _sources: None)
        args.update(extra)
        return gate.admit_fixture(**args)

    def test_default_external_anchor_is_closed_before_preflight(self):
        with self.assertRaisesRegex(RuntimeError, "attestation not installed"):
            self._admit()
        self.assertEqual(self.events, ["rehash", "qa"])

    def test_exact_signed_review_and_ready_preload_path(self):
        admitted = self._admit(anchor_attest=self._anchor)
        self.assertEqual(admitted["status"],
                         "fixture_admitted_for_fresh_process_source_only")
        self.assertEqual(self.events,
                         ["rehash", "qa", "anchor", "preflight", "rehash"])
        self.assertEqual(admitted["input_hashes"], self.hashes)

    def test_tampered_freeze_review_and_source_are_rejected(self):
        review = self.review_path.read_bytes()
        self.review_path.write_bytes(review.replace(b"synthetic-freeze", b"other-reviewer"))
        with self.assertRaisesRegex(ValueError, "HMAC differs"):
            self._admit(anchor_attest=self._anchor)
        self.review_path.write_bytes(review)
        self.source_path.write_bytes(self.source_path.read_bytes().replace(
            b"synthetic-source", b"other-source"))
        with self.assertRaisesRegex(ValueError, "source review changed"):
            self._admit(anchor_attest=self._anchor)

    def test_missing_or_synthetic_plan_fails_before_qa(self):
        self.plan["synthetic_test_injection"] = True
        self._write(self.plan_path, self.plan)
        with self.assertRaisesRegex(ValueError, "synthetic, incomplete"):
            self._admit(anchor_attest=self._anchor)
        self.assertEqual(self.events, [])

    def test_nonexistent_source_commit_cannot_bind_reviewed_bytes(self):
        with self.assertRaisesRegex(ValueError, "lacks exact admission bytes"):
            gate.verify_source_commit("0" * 40, gate.source_hashes())

    def test_full_map_rehash_rejects_tools_change(self):
        fixture = self.root / "local_fixture"
        fixture.mkdir()
        for name in ("system_message.txt", "tools.json", "protocol", "concept",
                     "preflight", "render.json"):
            (fixture / name).write_bytes(b'{"rows":[]}\n' if name == "render.json"
                                         else b"synthetic source only\n")
        for name in ("qa_manifest", "qa_ledger", "qa_receipt", "wav"):
            self.paths[name].write_bytes(b"synthetic bytes\n")
        for name in ("config.json", "model.safetensors"):
            path = self.paths["checkpoint"] / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"synthetic checkpoint\n")
        ref = self.paths["nano"] / "hub" / "models--nvidia--NVIDIA-Nemotron-Nano-9B-v2" / "refs" / "main"
        ref.parent.mkdir(parents=True)
        ref.write_bytes(b"synthetic ref\n")
        snapshot = self.paths["nano"] / "hub" / "models--nvidia--NVIDIA-Nemotron-Nano-9B-v2" / "snapshots" / "6533e8de2c68e4536bf7c411d7a3ce5734111476"
        snapshot.mkdir(parents=True)
        for name in plan_module.NANO_METADATA_FILES:
            (snapshot / name).write_bytes(b"synthetic nano\n")
        for relative in (plan_module.SAMPLE_SCRIPT, plan_module.SAMPLE_WAV,
                         plan_module.SAMPLE_RESPONSE, plan_module.TEMPLATE,
                         plan_module.HELPER, *plan_module.OTHER_PINNED_SOURCE):
            path = self.paths["nemo"] / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic nemo\n")
        with patch.multiple(plan_module, HERE=fixture,
                            RENDER=fixture / "render.json",
                            PROTOCOL=fixture / "protocol",
                            CONCEPT_REVIEW=fixture / "concept",
                            PREFLIGHT=fixture / "preflight"):
            hash_kwargs = {"nemo_dir": self.paths["nemo"],
                           "checkpoint_dir": self.paths["checkpoint"],
                           "nano_home": self.paths["nano"],
                           "qa_manifest": self.paths["qa_manifest"],
                           "qa_ledger": self.paths["qa_ledger"],
                           "qa_receipt": self.paths["qa_receipt"],
                           "wav_path": self.paths["wav"]}
            expected = plan_module._input_hashes(**hash_kwargs)
            self.assertEqual(gate.rehash_bound_inputs(
                {"input_hashes": expected}, self.paths), expected)
            (fixture / "tools.json").write_bytes(b"changed synthetic tools\n")
            with self.assertRaisesRegex(ValueError, "Complete fixture source-plan input changed"):
                gate.rehash_bound_inputs({"input_hashes": expected}, self.paths)


if __name__ == "__main__":
    unittest.main()
