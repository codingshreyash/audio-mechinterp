"""Small synthetic/tamper tests for the independent CPU saved-data auditor."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

import voxtral_old_fit_grad_mode_parity_audit_v1 as audit


def compact_fixture():
    sources = {
        "encoder31": torch.arange(8, dtype=torch.float32).reshape(1, 4, 2).to(torch.bfloat16),
        "projector": torch.arange(12, dtype=torch.float32).reshape(3, 4).to(torch.bfloat16),
    }
    spec = audit.ShapeSpec(
        source={"encoder31": (1, 4, 2), "projector": (3, 4)},
        projector=(3, 4), lm_input=(1, 5, 4), lm_width=4, logits=7,
        input_features=(1, 2, 3), audio_count=3, audio_token_id=24,
        source_sha256={site: audit.tensor_sha(tensor) for site, tensor in sources.items()},
    )
    ids = torch.tensor([[1, 24, 24, 24, 2]], dtype=torch.int64)
    positions = [1, 2, 3]
    base_lm = torch.zeros(spec.lm_input, dtype=torch.float32)
    calls = []
    for site, mode, repeat in audit._full_schedule():
        source_post = sources[site].float().clone()
        projector = (source_post.clone() if site == "projector" else
                     torch.full(spec.projector, 4.0, dtype=torch.float32))
        lm_input = base_lm.clone()
        lm_input[0, positions] = projector
        # A mode-linked change can first arise after the LM handoff.
        terminal = {name: torch.zeros(spec.lm_width, dtype=torch.float32)
                    for name in audit.STAGES}
        if mode == "G":
            for name in audit.STAGES:
                terminal[name][0] = 0.2
        logits = torch.zeros(spec.logits, dtype=torch.float32)
        if mode == "G":
            logits[0] = 0.3
        calls.append({
            "kind": "full", "ordinal": len(calls), "site": site,
            "mode": mode, "repeat": repeat,
            "source_sha256": spec.source_sha256[site],
            "source_post_full": source_post, "projector_full": projector,
            "lm_input_full": lm_input, "lm0_input_full": lm_input.clone(),
            "lm_terminal": terminal, "first_answer_logits": logits,
            "audio_placeholder_positions": positions.copy(),
        })
    origins = {(site, mode): next(row for row in calls if row["site"] == site
                                    and row["mode"] == mode)
               for site in sources for mode in ("N", "G")}
    for site, origin_mode, mode, repeat in audit._direct_schedule():
        origin = origins[(site, origin_mode)]
        lm_input = origin["lm_input_full"].clone()
        terminal = {name: torch.zeros(spec.lm_width, dtype=torch.float32)
                    for name in audit.STAGES}
        if mode == "G":
            for name in audit.STAGES:
                terminal[name][0] = 0.1
        logits = torch.zeros(spec.logits, dtype=torch.float32)
        if mode == "G":
            logits[0] = 0.4
        calls.append({
            "kind": "direct", "ordinal": len(calls), "site": site,
            "origin_mode": origin_mode, "origin_full_ordinal": origin["ordinal"],
            "mode": mode, "repeat": repeat,
            "input_sha256": audit.tensor_sha(origin["lm_input_full"]),
            "lm_input_full": lm_input, "lm0_input_full": lm_input.clone(),
            "lm_terminal": terminal, "first_answer_logits": logits,
        })
    payload = {"schema": audit.SCHEMA, "sources_bf16": sources,
               "batch": {"input_ids": ids,
                         "attention_mask": torch.ones_like(ids),
                         "input_features": torch.zeros(spec.input_features)},
               "audio_token_id": 24, "calls": calls}
    return payload, spec


class SavedTensorAuditTests(unittest.TestCase):
    def setUp(self):
        self.payload, self.spec = compact_fixture()

    def audit(self, payload=None):
        return audit.audit_payload(self.payload if payload is None else payload,
                                   spec=self.spec)

    def test_valid_trace_reports_first_resolved_at_lm0(self):
        report = self.audit()
        self.assertEqual(report["status"], audit.AUDIT_STATUS)
        for site in ("encoder31", "projector"):
            self.assertEqual(report["full_model"][site]["first_bitwise_difference"], "lm0")
            self.assertEqual(report["full_model"][site]["first_resolved_difference"], "lm0")
            self.assertTrue(report["full_model"][site]["deterministic_mode_contrast_supported"])
            self.assertEqual(len(report["full_model"][site]["boundaries"]["lm0"]
                                 ["cross_NG"]), 4)
            self.assertEqual(report["full_model"][site]["boundaries"]["lm0"]
                             ["cross_NG"][0]["relative_l2"], 1.0)
            self.assertEqual(report["modality_partition"][site]
                             ["audio_lm_input_N_vs_G"]["changed_coordinates"], 0)
        for group in report["direct_lm"].values():
            self.assertEqual(group["first_resolved_difference"], "lm0")
            self.assertEqual(group["boundaries"]["lm0"]["within_mode_l2_floor"], 0)
            self.assertTrue(group["fixed_direct_input_bitwise_equal"])

    def test_bitwise_signed_zero_is_visible_but_not_resolved(self):
        payload = copy.deepcopy(self.payload)
        payload["calls"][1]["lm_terminal"]["lm0"][0] = 0.0
        payload["calls"][2]["lm_terminal"]["lm0"][0] = 0.0
        payload["calls"][1]["lm_terminal"]["lm0"][1] = -0.0
        report = self.audit(payload)
        row = report["full_model"]["encoder31"]
        self.assertEqual(row["first_bitwise_difference"], "lm0")
        self.assertEqual(row["first_resolved_difference"], "lm7")
        self.assertFalse(row["boundaries"]["lm0"]["cross_NG_all_bitwise_equal"])

    def test_repeat_floor_blocks_a_spurious_cross_mode_signal(self):
        payload = copy.deepcopy(self.payload)
        # Within-G repeat spread exceeds the N/G first-pair effect at LM0.
        payload["calls"][2]["lm_terminal"]["lm0"][0] = 0.8
        report = self.audit(payload)
        row = report["full_model"]["encoder31"]
        self.assertFalse(row["boundaries"]["lm0"]["resolved_beyond_observed_repeat_floor"])
        self.assertEqual(row["first_resolved_difference"], "lm7")
        self.assertFalse(row["all_within_mode_bitwise_stable"])
        self.assertFalse(row["deterministic_mode_contrast_supported"])

    def test_relative_l2_undefined_for_two_zero_tensors(self):
        zeros = torch.zeros(3, dtype=torch.float32)
        row = audit.difference(zeros, zeros.clone())
        self.assertIsNone(row["relative_l2"])
        self.assertEqual(row["l2"], 0.0)

    def test_rejects_tampered_source_tensor_even_if_record_hash_unchanged(self):
        payload = copy.deepcopy(self.payload)
        payload["sources_bf16"]["encoder31"][0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "source hash differs"):
            self.audit(payload)

    def test_rejects_projector_to_audio_placeholder_disconnect(self):
        payload = copy.deepcopy(self.payload)
        payload["calls"][0]["lm_input_full"][0, 1, 0] += 1
        payload["calls"][0]["lm0_input_full"][0, 1, 0] += 1
        with self.assertRaisesRegex(ValueError, "projector differs"):
            self.audit(payload)

    def test_rejects_stale_direct_origin_hash(self):
        payload = copy.deepcopy(self.payload)
        payload["calls"][8]["input_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "origin descriptor/hash"):
            self.audit(payload)

    def test_rejects_direct_input_drift_with_valid_origin_hash(self):
        payload = copy.deepcopy(self.payload)
        payload["calls"][8]["lm_input_full"][0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "direct input differs"):
            self.audit(payload)

    def test_rejects_call_order_and_missing_repeats(self):
        for alteration in ("swap", "drop"):
            payload = copy.deepcopy(self.payload)
            if alteration == "swap":
                payload["calls"][1], payload["calls"][2] = (payload["calls"][2],
                                                            payload["calls"][1])
            else:
                payload["calls"].pop()
            with self.assertRaisesRegex(ValueError, "schedule/schema|24 saved calls"):
                self.audit(payload)

    def test_rejects_nonfinite_stage_and_wrong_audio_ids(self):
        payload = copy.deepcopy(self.payload)
        payload["calls"][0]["lm_terminal"]["lm0"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finiteness"):
            self.audit(payload)
        payload = copy.deepcopy(self.payload)
        payload["batch"]["input_ids"][0, 1] = 1
        with self.assertRaisesRegex(ValueError, "Audio placeholders"):
            self.audit(payload)

    def test_content_addressed_bundle_and_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary)
            blobs = attempt / "blobs"
            blobs.mkdir()
            raw = blobs / "details.tmp"
            torch.save(self.payload, raw)
            digest = audit.sha(raw)
            sidecar = blobs / f"{digest}.pt"
            raw.rename(sidecar)
            (attempt / "model_layout.json").write_text('{"valid":true}\n')
            (attempt / "progress.jsonl").write_text("{}\n")
            meta = {
                "status": "started_old_fit_fp32_graph_inference_parity_diagnostic_only",
                "scope": audit.SCOPE, "planned_counts": audit.COUNTS,
                "source_sha256": self.spec.source_sha256,
                "full_schedule": [list(row) for row in audit._full_schedule()],
                "direct_schedule": [list(row) for row in audit._direct_schedule()],
                "protocol_sha256": "a" * 64, "runner_sha256": "b" * 64,
                "failed_stage_b_fp32_manifest_sha256": "c" * 64,
            }
            (attempt / "meta.json").write_text(json.dumps(meta))
            result = {
                "status": audit.STATUS, "scope": audit.SCOPE,
                "planned_counts": audit.COUNTS, "actual_counts": audit.COUNTS,
                "source_sha256": self.spec.source_sha256,
                "audio_token_id": self.spec.audio_token_id,
                "protocol_sha256": meta["protocol_sha256"],
                "runner_sha256": meta["runner_sha256"],
                "failed_stage_b_fp32_manifest_sha256":
                    meta["failed_stage_b_fp32_manifest_sha256"],
                "meta_sha256": audit.sha(attempt / "meta.json"),
                "model_layout_sha256": audit.sha(attempt / "model_layout.json"),
                "progress_sha256": audit.sha(attempt / "progress.jsonl"),
                "details_sha256": digest, "details_path": f"blobs/{digest}.pt",
                "details_bytes": sidecar.stat().st_size,
                "elapsed_seconds": 1.0, "peak_mps_driver_bytes": 0,
            }
            (attempt / "result.json").write_text(json.dumps(result))
            out = attempt / "audit.json"
            self.assertEqual(audit.audit_run(attempt, out, spec=self.spec)["status"],
                             audit.AUDIT_STATUS)
            self.assertTrue(out.exists())
            with self.assertRaisesRegex(ValueError, "new file"):
                audit.audit_run(attempt, out, spec=self.spec)
            out.unlink()
            sidecar.write_bytes(sidecar.read_bytes() + b"x")
            with self.assertRaisesRegex(ValueError, "sidecar bytes differ"):
                audit.audit_run(attempt, out, spec=self.spec)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
