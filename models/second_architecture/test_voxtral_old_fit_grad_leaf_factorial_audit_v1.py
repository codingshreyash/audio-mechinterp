"""Small CPU and tamper tests for the independent 2x2 saved-data audit."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

import voxtral_old_fit_grad_leaf_factorial_audit_v1 as audit
from test_voxtral_old_fit_grad_mode_parity_audit_v1 import compact_fixture


def factorial_fixture():
    old, spec = compact_fixture()
    sources, batch = old["sources_bf16"], old["batch"]
    positions = [1, 2, 3]
    calls = []
    for site, mode, repeat in audit.full_schedule():
        source = sources[site].float().clone()
        projector = (source.clone() if site == "projector" else
                     torch.full(spec.projector, 4, dtype=torch.float32))
        lm = torch.zeros(spec.lm_input, dtype=torch.float32)
        lm[0, positions] = projector
        value = {"N0": 0.0, "N1": 0.05, "G0": 0.1, "G1": 0.3}[mode]
        terminal = {name: torch.full((spec.lm_width,), value)
                    for name in audit.base.STAGES}
        logits = torch.full((spec.logits,), value)
        calls.append({"kind": "full", "ordinal": len(calls), "site": site,
                      "mode": mode, "repeat": repeat,
                      "factor_observed": {"grad_enabled": mode.startswith("G"),
                                          "leaf_requires_grad": mode.endswith("1")},
                      "downstream_grad_flags": {
                          "lm_input": {"grad_enabled": mode.startswith("G"),
                                       "requires_grad": mode == "G1"},
                          "lm0_pre": {"grad_enabled": mode.startswith("G"),
                                      "requires_grad": mode == "G1"}},
                      "source_sha256": spec.source_sha256[site],
                      "source_post_full": source, "projector_full": projector,
                      "lm_input_full": lm, "lm0_input_full": lm.clone(),
                      "lm_terminal": terminal, "first_answer_logits": logits,
                      "audio_placeholder_positions": positions.copy()})
    origins = {site: next(row for row in calls
                          if row["site"] == site and row["mode"] == "N0")
               for site in sources}
    for site, origin_mode, mode, repeat in audit.direct_schedule():
        origin = origins[site]
        fixed = origin["lm_input_full"].clone()
        value = {"N0": 0.0, "N1": 0.05, "G0": 0.1, "G1": 0.3}[mode]
        terminal = {name: torch.full((spec.lm_width,), value)
                    for name in audit.base.STAGES}
        logits = torch.full((spec.logits,), value)
        calls.append({"kind": "direct", "ordinal": len(calls), "site": site,
                      "origin_mode": origin_mode, "origin_full_ordinal": origin["ordinal"],
                      "mode": mode, "repeat": repeat,
                      "factor_observed": {"grad_enabled": mode.startswith("G"),
                                          "leaf_requires_grad": mode.endswith("1")},
                      "downstream_grad_flags": {
                          "lm_input": {"grad_enabled": mode.startswith("G"),
                                       "requires_grad": mode.endswith("1")},
                          "lm0_pre": {"grad_enabled": mode.startswith("G"),
                                      "requires_grad": mode.endswith("1")}},
                      "input_sha256": audit.base.tensor_sha(fixed),
                      "lm_input_full": fixed, "lm0_input_full": fixed.clone(),
                      "lm_terminal": terminal, "first_answer_logits": logits})
    return {"schema": audit.SCHEMA, "sources_bf16": sources, "batch": batch,
            "audio_token_id": 24, "calls": calls}, spec


class FactorialAuditTests(unittest.TestCase):
    def setUp(self):
        self.payload, self.spec = factorial_fixture()

    def report(self, payload=None):
        return audit.audit_payload(self.payload if payload is None else payload,
                                   spec=self.spec)

    def test_fixed_factorial_schedule_and_contrasts(self):
        self.assertEqual(len(audit.full_schedule()), 16)
        self.assertEqual(len(audit.direct_schedule()), 16)
        result = self.report()
        for site in ("encoder31", "projector"):
            full = result["full_model"][site]
            self.assertEqual(full["contrasts"]["N0_vs_G1"]["first_resolved_difference"],
                             "lm0")
            self.assertTrue(full["contrasts"]["N0_vs_G1"]
                            ["deterministic_factor_contrast_supported"])
            self.assertEqual(len(full["contrasts"]["N0_vs_G1"]
                                 ["boundaries"]["lm0"]["cross_pairs"]), 4)
            self.assertAlmostEqual(full["interaction"]["lm0"]["matched_repeat_l2"][0],
                                   0.3, places=6)
            self.assertTrue(result["direct_lm"][site]["fixed_direct_input_bitwise_equal"])
            for partition in ("audio_lm_input", "non_audio_lm_input"):
                self.assertIsNone(result["modality_partition"][site][partition]
                                  ["contrasts"]["N0_vs_G1"]["first_bitwise_difference"])

    def test_repeat_spread_prevents_deterministic_claim(self):
        payload = copy.deepcopy(self.payload)
        # The second encoder31 G1 call is ordinal 6.
        payload["calls"][6]["lm_terminal"]["lm0"][0] = 5.0
        report = self.report(payload)
        row = report["full_model"]["encoder31"]["contrasts"]["N0_vs_G1"]
        self.assertFalse(row["boundaries"]["lm0"]["all_within_condition_bitwise_stable"])
        self.assertFalse(row["deterministic_factor_contrast_supported"])
        self.assertEqual(row["first_resolved_difference"], "lm7")

    def test_rejects_changed_batch_source_and_schedule(self):
        for mutation, message in (
            (lambda p: p["batch"]["input_ids"][0, 0].fill_(999),
             "Frozen batch tensor hash differs"),
            (lambda p: p["sources_bf16"]["encoder31"][0, 0, 0].add_(1),
             "source hash differs"),
            (lambda p: p["calls"].pop(), "32 saved calls"),
            (lambda p: p["calls"].__setitem__(0, p["calls"][1]), "schedule/schema"),
        ):
            payload = copy.deepcopy(self.payload)
            mutation(payload)
            with self.assertRaisesRegex(ValueError, message):
                self.report(payload)

    def test_rejects_projector_handoff_and_direct_origin_drift(self):
        payload = copy.deepcopy(self.payload)
        payload["calls"][0]["lm_input_full"][0, 1, 0] += 1
        payload["calls"][0]["lm0_input_full"][0, 1, 0] += 1
        with self.assertRaisesRegex(ValueError, "projector/LM handoff"):
            self.report(payload)
        payload = copy.deepcopy(self.payload)
        payload["calls"][16]["lm_input_full"][0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "fixed direct input"):
            self.report(payload)

    def test_rejects_mislabeled_factor_flags(self):
        payload = copy.deepcopy(self.payload)
        payload["calls"][0]["factor_observed"]["grad_enabled"] = True
        with self.assertRaisesRegex(ValueError, "recorded factor flags"):
            self.report(payload)
        payload = copy.deepcopy(self.payload)
        payload["calls"][16]["factor_observed"]["leaf_requires_grad"] = True
        with self.assertRaisesRegex(ValueError, "recorded factor flags"):
            self.report(payload)
        payload = copy.deepcopy(self.payload)
        payload["calls"][0]["factor_observed"]["grad_enabled"] = 0
        with self.assertRaisesRegex(ValueError, "recorded factor flags"):
            self.report(payload)

    def test_content_addressed_attempt_and_pin(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            blobdir = path / "blobs"
            blobdir.mkdir()
            temp = blobdir / "raw.pt"
            torch.save(self.payload, temp)
            digest = audit.base.sha(temp)
            sidecar = blobdir / f"{digest}.pt"
            temp.rename(sidecar)
            (path / "model_layout.json").write_text('{"valid":true}\n')
            (path / "progress.jsonl").write_text("{}\n")
            meta = {"status": audit.META_STATUS, "scope": audit.SCOPE,
                    "planned_counts": audit.COUNTS,
                    "source_sha256": self.spec.source_sha256,
                    "full_schedule": [list(x) for x in audit.full_schedule()],
                    "direct_schedule": [list(x) for x in audit.direct_schedule()],
                    "protocol_sha256": "a" * 64, "runner_sha256": "b" * 64,
                    "failed_stage_b_fp32_manifest_sha256": "c" * 64,
                    "failed_v1_diagnostic_manifest_sha256": audit.V1_FAILED_MANIFEST_SHA}
            (path / "meta.json").write_text(json.dumps(meta))
            result = {"status": audit.STATUS, "scope": audit.SCOPE,
                      "planned_counts": audit.COUNTS, "actual_counts": audit.COUNTS,
                      "source_sha256": self.spec.source_sha256,
                      "audio_token_id": self.spec.audio_token_id,
                      **{key: meta[key] for key in (
                          "protocol_sha256", "runner_sha256",
                          "failed_stage_b_fp32_manifest_sha256",
                          "failed_v1_diagnostic_manifest_sha256")},
                      "meta_sha256": audit.base.sha(path / "meta.json"),
                      "model_layout_sha256": audit.base.sha(path / "model_layout.json"),
                      "progress_sha256": audit.base.sha(path / "progress.jsonl"),
                      "details_sha256": digest, "details_path": f"blobs/{digest}.pt",
                      "details_bytes": sidecar.stat().st_size,
                      "elapsed_seconds": 1.0, "peak_mps_driver_bytes": 0}
            (path / "result.json").write_text(json.dumps(result))
            out = path / "audit.json"
            self.assertEqual(audit.audit_run(path, out, spec=self.spec)["status"],
                             audit.AUDIT_STATUS)
            self.assertTrue(out.is_file())
            with self.assertRaisesRegex(ValueError, "new file"):
                audit.audit_run(path, out, spec=self.spec)
            out.unlink()
            sidecar.write_bytes(sidecar.read_bytes() + b"x")
            with self.assertRaisesRegex(ValueError, "sidecar differs"):
                audit.audit_run(path, out, spec=self.spec)


if __name__ == "__main__":
    unittest.main()
