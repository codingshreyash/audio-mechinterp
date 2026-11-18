"""CPU-only tests for independently recomputed non-beaver pilot arithmetic."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import voxtral_fp32_nonbeaver_linearity_audit_v1 as audit


def small_payload(engine: str = "piper", recipient: str = "silver") -> tuple[dict, audit.AuditSpec]:
    """Complete 96-forward synthetic capture with small tensor dimensions."""
    donor_word = "teal" if recipient == "silver" else "silver"
    index = audit.CONTEXTS.index((engine, recipient))
    shapes = {"encoder31": (1, 3, 2), "projector": (2, 4)}
    rows = {"encoder31": [0], "projector": [0]}
    basis = torch.eye(4, dtype=torch.float64).contiguous()
    ids = torch.tensor([[24, 24, 1]], dtype=torch.int64)
    mask = torch.ones_like(ids)
    features = torch.zeros((1, 2, 3), dtype=torch.float32)
    batch = {"input_ids": ids, "attention_mask": mask,
             "input_features": features}
    hashes = {"input_ids": audit.tensor_sha(ids),
              "attention_mask": audit.tensor_sha(mask),
              (engine, recipient): audit.tensor_sha(features),
              (engine, donor_word): audit.tensor_sha(features)}
    spec = audit.AuditSpec(shapes, (1, 3, 4), (4,), (1, 2, 3), rows,
                           basis, hashes)
    encoder_donor = torch.zeros(shapes["encoder31"], dtype=torch.float32)
    encoder_donor[0, 0, 0] = 1
    projector_donor = torch.zeros(shapes["projector"], dtype=torch.float32)
    projector_donor[0, 0] = 1
    donor_lm = torch.zeros(spec.lm_input_shape, dtype=torch.float32)
    donor_lm[0, :2] = projector_donor
    donor = {"kind": "donor", "mode": "N0", "encoder31_full": encoder_donor,
             "projector_full": projector_donor, "lm_input_full": donor_lm,
             "lm_terminal": {stage: torch.zeros(4, dtype=torch.float32)
                             for stage in ("lm0", "lm14", "lm29")},
             "first_answer_logits": torch.zeros(4, dtype=torch.float32),
             "audio_positions": [0, 1],
             "runtime_flags": {"grad_enabled": False,
                               "lm0_grad_enabled": False,
                               "lm_input_requires_grad": False}}
    recipient_passive = copy.deepcopy(donor)
    recipient_passive["kind"] = "recipient"
    recipient_passive["encoder31_full"] = torch.zeros_like(encoder_donor)
    recipient_passive["projector_full"] = torch.zeros_like(projector_donor)
    recipient_passive["lm_input_full"] = torch.zeros_like(donor_lm)
    recipient_passive["encoder31_full"][0, 0, 0] = -0.0
    recipient_passive["projector_full"][0, 0] = -0.0
    recipient_passive["lm_input_full"][0, 0, 0] = -0.0
    deltas = {"encoder31": torch.tensor([1., 0.], dtype=torch.float32),
              "projector": torch.tensor([1., 0., 0., 0.], dtype=torch.float32)}

    def make_trace(site: str, mode: str, alpha: float,
                   full: bool) -> dict:
        source = torch.zeros(shapes[site], dtype=torch.float32)
        if site == "encoder31":
            source[0, 0, 0] = -0.0
        else:
            source[0, 0] = -0.0
        post = source.clone()
        if site == "encoder31":
            if alpha:
                post[0, 0] += alpha * deltas[site]
            pre_rows, post_rows = source[0, rows[site]].clone(), post[0, rows[site]].clone()
        else:
            if alpha:
                post[0] += alpha * deltas[site]
            pre_rows, post_rows = source[rows[site]].clone(), post[rows[site]].clone()
        effective = (post_rows - pre_rows).contiguous()
        projector = post.clone() if site == "projector" else torch.zeros(shapes["projector"])
        lm_input = torch.zeros(spec.lm_input_shape)
        lm_input[0, :2] = projector
        terminal = torch.tensor([alpha, 0., 0., 0.], dtype=torch.float32)
        if alpha:
            fidelity = audit.edit_fidelity(effective, deltas[site], alpha)
            metrics = {"effective_norm": fidelity["effective_norm"],
                       "mean_fidelity_ratio": fidelity["mean_fidelity_ratio"],
                       "row_nonuniformity_ratio": fidelity["row_nonuniformity_ratio"],
                       "unchanged_fraction": float((effective == 0).float().mean())}
        else:
            metrics = {"effective_norm": 0.0, "mean_fidelity_ratio": 0.0,
                       "row_nonuniformity_ratio": 0.0, "unchanged_fraction": 1.0}
        flag = mode == "G1"
        record = {"site": site, "mode": mode, "alpha": float(alpha),
                  "source_pre_rows": pre_rows, "source_post_rows": post_rows,
                  "effective_rows": effective,
                  "source_pre_sha256": audit.tensor_sha(source),
                  "source_post_sha256": audit.tensor_sha(post),
                  "projector_sha256": audit.tensor_sha(projector.contiguous()),
                  "lm_input_sha256": audit.tensor_sha(lm_input.contiguous()),
                  "lm0_pre_sha256": audit.tensor_sha(lm_input.contiguous()),
                  "lm_terminal": {stage: terminal.clone()
                                  for stage in ("lm0", "lm14", "lm29")},
                  "first_answer_logits": torch.zeros(4),
                  "audio_positions": [0, 1],
                  "runtime_flags": {key: flag for key in audit.TRACE_FLAGS},
                  "runtime_checks": {"selected_edit_bitwise": True,
                                     "zero_source_bitwise": True,
                                     "projector_lm_input_bitwise": True,
                                     "lm_input_lm0_pre_bitwise": True},
                  "edit_metrics": metrics}
        if full:
            record.update({"source_pre_full": source,
                           "source_post_full": post,
                           "projector_full": projector,
                           "lm_input_full": lm_input})
        return record

    site_bundle = {}
    for site in audit.SITES:
        nrows, width = len(rows[site]), shapes[site][-1]
        gradients = torch.zeros((4, nrows, width), dtype=torch.float32)
        gradients[0, 0, 0] = 1
        site_bundle[site] = {"g1_clean": make_trace(site, "G1", 0, True),
                             "n0_clean": make_trace(site, "N0", 0, True),
                             "gradients": gradients}
    forwards = []
    for ordinal, (site, mode, cycle, alpha, label) in enumerate(audit.patched_schedule(engine)):
        forwards.append({"ordinal": ordinal, "site": site, "mode": mode,
                         "cycle": cycle, "alpha": alpha, "label": label,
                         "trace": make_trace(site, mode, 0 if label == "zero" else alpha,
                                             label == "zero")})
    return ({"schema": audit.SCHEMA,
             "context": {"index": index, "label": audit.CONTEXT_LABELS[index], "engine": engine,
                         "recipient": recipient, "donor": donor_word},
             "basis": basis.clone(), "batches": {"recipient": batch,
                                          "donor": copy.deepcopy(batch)},
             "rows": rows, "delta": deltas, "donor": donor,
             "recipient_passive": recipient_passive,
             "sites": site_bundle, "forwards": forwards,
             "generations": {"recipient": {"suffix_ids": [2]},
                             "donor": {"suffix_ids": [2]}}}, spec)


class ArithmeticTests(unittest.TestCase):
    def test_saved_data_audit_rehashes_sidecar_and_result(self) -> None:
        payload, spec = small_payload()
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            (run_dir / "blobs").mkdir()
            temporary = run_dir / "blobs" / "pending.pt"
            torch.save(payload, temporary)
            digest = audit.file_sha(temporary)
            sidecar = run_dir / "blobs" / f"{digest}.pt"
            temporary.rename(sidecar)
            preflight = run_dir / "preflight.json"
            preflight.write_text("{}\n")
            context = payload["context"]
            meta = {"schema": "voxtral_fp32_nonbeaver_linearity_meta_v1",
                    "status": "started_source_only_reviewed_context",
                    "scope": audit.SCOPE, "context": context,
                    "planned_counts": audit.COUNTS,
                    "patched_schedule": audit.patched_schedule("piper"),
                    "protocol_v2_sha256": audit.PROTOCOL_V2_SHA256,
                    "protocol_v3_sha256": audit.PROTOCOL_V3_SHA256,
                    "protocol_v4_sha256": audit.PROTOCOL_V4_SHA256,
                    "protocol_v5_sha256": audit.PROTOCOL_V5_SHA256,
                    "preflight_receipt_sha256": audit.file_sha(preflight),
                    "selection_sha256": audit.SELECTION_SHA256,
                    "masks_sha256": audit.MASKS_SHA256,
                    "attempt_id": "test-attempt", "freeze_sha256": "a" * 64,
                    "runner_sha256": audit.file_sha(audit.RUNNER_PATH),
                    "source_dependencies":
                        {key: audit.file_sha(path) for key, path in
                         audit.SOURCE_DEPENDENCY_FILES.items()},
                    "basis_sha256": audit.BASIS_SHA256,
                    "wav_sha256": {word: audit.WAV_SHA256[("piper", word)]
                                    for word in ("silver", "teal")},
                    "rows": spec.rows,
                    "batch_sha256": {role: {key: audit.tensor_sha(value)
                                              for key, value in batch.items()}
                                     for role, batch in payload["batches"].items()},
                    "device": "mps", "fresh_v2_corpus_opened": False,
                    "model_seconds_cap": audit.MAX_SECONDS,
                    "max_mps_driver_bytes": audit.MAX_MPS_BYTES,
                    "max_worker_bytes": audit.MAX_WORKER_BYTES,
                    "max_output_bytes": 2 * 1024**3}
            (run_dir / "meta.json").write_text(json.dumps(meta))
            (run_dir / "progress.jsonl").write_text("{}\n")
            (run_dir / "model_layout.json").write_text('{"valid":true,"arm":"fp32"}\n')
            result = {"status": audit.RESULT_STATUS, "scope": audit.SCOPE,
                      "context": context, "planned_counts": audit.COUNTS,
                      "actual_counts": audit.COUNTS,
                      "protocol_v2_sha256": audit.PROTOCOL_V2_SHA256,
                      "protocol_v3_sha256": audit.PROTOCOL_V3_SHA256,
                      "protocol_v4_sha256": audit.PROTOCOL_V4_SHA256,
                      "protocol_v5_sha256": audit.PROTOCOL_V5_SHA256,
                      "preflight_receipt_sha256": audit.file_sha(preflight),
                      "meta_sha256": audit.file_sha(run_dir / "meta.json"),
                      "progress_sha256": audit.file_sha(run_dir / "progress.jsonl"),
                      "model_layout_sha256": audit.file_sha(run_dir / "model_layout.json"),
                      "details_sha256": digest, "details_path": f"blobs/{digest}.pt",
                      "details_bytes": sidecar.stat().st_size,
                      "elapsed_seconds": 1.0, "peak_mps_driver_bytes": 1,
                      "invalidities": [], "attempt_id": "test-attempt",
                      "freeze_sha256": "a" * 64}
            result["output_bytes_before_result"] = sum(
                path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
            (run_dir / "result.json").write_text(json.dumps(result))
            with (mock.patch.object(audit, "PREFLIGHT_PATH", preflight),
                  mock.patch.object(audit, "PREFLIGHT_SHA256", audit.file_sha(preflight)),
                  mock.patch.object(audit, "frozen_rows", return_value=spec.rows),
                  mock.patch.object(audit, "production_spec", return_value=spec)):
                report = audit.audit_run(run_dir, run_dir / "audit.json")
            self.assertEqual(report["status"], audit.AUDIT_STATUS)
            self.assertEqual(report["details_sha256"], digest)
            self.assertTrue((run_dir / "audit.json").is_file())
            with self.assertRaisesRegex(ValueError, "new"):
                audit.audit_run(run_dir, run_dir / "audit.json")
            (run_dir / "audit.json").unlink()
            original_meta = (run_dir / "meta.json").read_bytes()
            (run_dir / "meta.json").write_text('{"status":"tampered"}\n')
            with (mock.patch.object(audit, "PREFLIGHT_PATH", preflight),
                  mock.patch.object(audit, "PREFLIGHT_SHA256", audit.file_sha(preflight)),
                  mock.patch.object(audit, "frozen_rows", return_value=spec.rows),
                  mock.patch.object(audit, "production_spec", return_value=spec)):
                with self.assertRaises(ValueError):
                    audit.audit_run(run_dir, run_dir / "audit2.json")
            (run_dir / "meta.json").write_bytes(original_meta)
            with sidecar.open("ab") as stream:
                stream.write(b"tampered")
            with (mock.patch.object(audit, "PREFLIGHT_PATH", preflight),
                  mock.patch.object(audit, "PREFLIGHT_SHA256", audit.file_sha(preflight)),
                  mock.patch.object(audit, "frozen_rows", return_value=spec.rows),
                  mock.patch.object(audit, "production_spec", return_value=spec)):
                with self.assertRaises(ValueError):
                    audit.audit_run(run_dir, run_dir / "audit3.json")
    def test_complete_synthetic_payload_and_campaign_gate(self) -> None:
        payload, spec = small_payload()
        report = audit.audit_payload(payload, spec)
        self.assertEqual(report["status"], audit.AUDIT_STATUS)
        self.assertEqual(report["computed_invalidities"], [])
        self.assertEqual(len(report["sites"]["encoder31"]["cells"]), 6)
        self.assertEqual(report["sites"]["encoder31"]["zero_drift"]["G1"]["pair_count"], 66)
        self.assertTrue(all(cell["valid"] and cell["resolved"] and
                            cell["primary_max_cycle_error"] == 0
                            for site in audit.SITES for cell in report["sites"][site]["cells"]))
        reports = [audit.audit_payload(*small_payload(engine, recipient))
                   for engine, recipient in audit.CONTEXTS]
        campaign = audit.aggregate_campaign(reports)
        self.assertTrue(campaign["advance_to_fresh_corpus_feasibility_gate"])
        self.assertEqual(campaign["site_gate"]["projector"]["planned"], 24)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit.aggregate_campaign([reports[0]] * 4)

    def test_missing_zero_and_wrong_basis_rejected(self) -> None:
        payload, spec = small_payload()
        payload["forwards"].pop(0)
        with self.assertRaisesRegex(ValueError, "96"):
            audit.audit_payload(payload, spec)
        payload, spec = small_payload()
        payload["basis"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "basis differs"):
            audit.audit_payload(payload, spec)

    def test_tampered_gradient_and_nonfinite_edit_rejected(self) -> None:
        payload, spec = small_payload()
        payload["sites"]["projector"]["gradients"] = torch.zeros((4, 2, 4))
        with self.assertRaisesRegex(ValueError, "VJPs"):
            audit.audit_payload(payload, spec)
        payload, spec = small_payload()
        payload["forwards"][1]["trace"]["effective_rows"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finiteness"):
            audit.audit_payload(payload, spec)

    def test_false_handoff_flags_rejected_on_full_and_edited_traces(self) -> None:
        for ordinal, key in (
            (0, "lm_input_lm0_pre_bitwise"),
            (1, "projector_lm_input_bitwise"),
            (1, "lm_input_lm0_pre_bitwise"),
        ):
            with self.subTest(ordinal=ordinal, key=key):
                payload, spec = small_payload()
                payload["forwards"][ordinal]["trace"]["runtime_checks"][key] = False
                with self.assertRaisesRegex(ValueError, "handoff failure"):
                    audit.audit_payload(payload, spec)

    def test_edited_lm_input_lm0_digest_claims_must_agree(self) -> None:
        payload, spec = small_payload()
        payload["forwards"][1]["trace"]["lm0_pre_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "digest claims differ"):
            audit.audit_payload(payload, spec)

    def test_full_projector_handoff_recomputed_from_retained_tensors(self) -> None:
        payload, spec = small_payload()
        zero = payload["forwards"][0]["trace"]
        zero["projector_full"][1, 1] = 1.0
        zero["projector_sha256"] = audit.tensor_sha(zero["projector_full"])
        with self.assertRaisesRegex(ValueError, "Retained projector/LM-input handoff"):
            audit.audit_payload(payload, spec)

    def test_clean_zero_parity_failure_keeps_planned_cells(self) -> None:
        payload, spec = small_payload()
        trace = payload["forwards"][0]["trace"]
        trace["first_answer_logits"][0] = 1.0
        report = audit.audit_payload(payload, spec)
        self.assertEqual(len(report["sites"]["encoder31"]["cells"]), 6)
        self.assertTrue(all(not cell["valid"] for cell in report["sites"]["encoder31"]["cells"]))
        self.assertTrue(report["computed_invalidities"])

    def test_bad_zero_source_is_recorded_and_kept_in_denominator(self) -> None:
        payload, spec = small_payload()
        trace = payload["forwards"][0]["trace"]
        trace["source_post_full"][0, 0, 0] = 1.0
        trace["source_post_rows"][0, 0] = 1.0
        trace["effective_rows"][0, 0] = 1.0
        trace["source_post_sha256"] = audit.tensor_sha(trace["source_post_full"])
        trace["runtime_checks"]["zero_source_bitwise"] = False
        report = audit.audit_payload(payload, spec)
        self.assertEqual(len(report["sites"]["encoder31"]["cells"]), 6)
        self.assertTrue(all(not row["valid"] for row in report["sites"]["encoder31"]["cells"]))
        self.assertIn("zero_source_bitwise", report["computed_invalidities"][0]["failed_checks"])

    def test_signed_zero_in_zero_source_fails_exact_identity(self) -> None:
        payload, spec = small_payload()
        trace = payload["forwards"][0]["trace"]
        trace["source_post_full"][0, 0, 0] = +0.0
        trace["source_post_rows"][0, 0] = +0.0
        trace["effective_rows"] = (trace["source_post_rows"] -
                                   trace["source_pre_rows"]).contiguous()
        trace["source_post_sha256"] = audit.tensor_sha(trace["source_post_full"])
        trace["runtime_checks"]["zero_source_bitwise"] = False
        report = audit.audit_payload(payload, spec)
        self.assertFalse(report["sites"]["encoder31"]["cells"][0]["valid"])
        self.assertIn("zero_source_bitwise", report["computed_invalidities"][0]["failed_checks"])

    def test_signed_zero_clean_identity_failure_invalidates_whole_site(self) -> None:
        payload, spec = small_payload()
        clean = payload["sites"]["encoder31"]["g1_clean"]
        clean["source_post_full"][0, 0, 0] = +0.0
        clean["source_post_rows"][0, 0] = +0.0
        clean["effective_rows"] = (clean["source_post_rows"] -
                                   clean["source_pre_rows"]).contiguous()
        clean["source_post_sha256"] = audit.tensor_sha(clean["source_post_full"])
        clean["runtime_checks"]["zero_source_bitwise"] = False
        report = audit.audit_payload(payload, spec)
        self.assertFalse(report["sites"]["encoder31"][
            "site_clean_zero_identity_valid"]["G1"])
        self.assertEqual(report["computed_invalidities"][0]["stage"],
                         "site_clean_zero_identity")
        self.assertTrue(all(not cell["valid"] for cell in
                            report["sites"]["encoder31"]["cells"]))

    def test_passive_recipient_parity_invalidates_site_without_dropping_cells(self) -> None:
        payload, spec = small_payload()
        payload["recipient_passive"]["encoder31_full"][0, 2, 0] = 1.0
        report = audit.audit_payload(payload, spec)
        self.assertFalse(report["sites"]["encoder31"][
            "passive_recipient_to_n0_clean_full_prepatch_bitwise"])
        self.assertTrue(all(not cell["valid"] for cell in
                            report["sites"]["encoder31"]["cells"]))
        self.assertEqual(len(report["sites"]["encoder31"]["cells"]), 6)
        self.assertEqual(report["computed_invalidities"][0]["stage"],
                         "passive_recipient_to_n0_clean")
        self.assertTrue(all(cell["valid"] for cell in
                            report["sites"]["projector"]["cells"]))

    def test_signed_zero_in_full_zero_logit_breaks_clean_parity(self) -> None:
        payload, spec = small_payload()
        payload["forwards"][0]["trace"]["first_answer_logits"][0] = -0.0
        report = audit.audit_payload(payload, spec)
        self.assertEqual(report["sites"]["encoder31"][
            "clean_zero_parity_failures"][0]["boundary"],
                         "first_answer_logits")
        self.assertTrue(all(not cell["valid"] for cell in
                            report["sites"]["encoder31"]["cells"]))

    def test_primary_error_uses_worst_of_both_cycles(self) -> None:
        payload, spec = small_payload()
        # Encoder G1 cycle 1 uses the reversed alpha order: +.00625 is last.
        edited = payload["forwards"][23]["trace"]
        edited["lm_terminal"]["lm29"][0] = 0.0125
        report = audit.audit_payload(payload, spec)
        cell = report["sites"]["encoder31"]["cells"][0]
        self.assertTrue(cell["valid"] and cell["resolved"])
        self.assertAlmostEqual(cell["cycles"][0]["numeric"]["uniform_relative_error"], 0)
        self.assertAlmostEqual(cell["cycles"][1]["numeric"]["uniform_relative_error"], 0.5)
        self.assertAlmostEqual(cell["primary_max_cycle_error"], 0.5)

    def test_donor_direction_must_use_passive_recipient_source(self) -> None:
        payload, spec = small_payload()
        payload["recipient_passive"]["encoder31_full"][0, 0, 0] = 0.25
        with self.assertRaisesRegex(ValueError, "donor-minus-passive-recipient"):
            audit.audit_payload(payload, spec)

    def test_frozen_forward_schedule(self) -> None:
        piper = audit.patched_schedule("piper")
        kokoro = audit.patched_schedule("kokoro")
        self.assertEqual(len(piper), 96)
        self.assertEqual(piper[0], ("encoder31", "G1", 0, 0.00625, "zero"))
        self.assertEqual(piper[1], ("encoder31", "G1", 0, 0.00625, "edit"))
        self.assertEqual(piper[12], ("encoder31", "G1", 1, -0.1, "zero"))
        self.assertEqual(kokoro[0][0], "projector")
        with self.assertRaisesRegex(ValueError, "voice"):
            audit.patched_schedule("unlisted")

    def test_frozen_basis_digest(self) -> None:
        basis = audit.fixed_basis()
        self.assertEqual(tuple(basis.shape), (3072, 4))
        self.assertEqual(basis.dtype, torch.float64)

    def test_zero_drift_uses_all_66_pairs(self) -> None:
        basis = torch.eye(4, dtype=torch.float64)
        zeros = [torch.tensor([float(i), 0, 0, 0], dtype=torch.float32)
                 for i in range(12)]
        outcome = audit.zero_drift(basis, zeros)
        self.assertEqual(outcome["pair_count"], 66)
        self.assertEqual(outcome["observed_max_l2"], 11.0)
        self.assertEqual(outcome["resolution_floor"], 33.0)
        with self.assertRaisesRegex(ValueError, "twelve"):
            audit.zero_drift(basis, zeros[:-1])

    def test_realized_row_and_uniform_estimands(self) -> None:
        gradients = torch.tensor([[[1., 0.], [0., 1.]],
                                  [[2., 0.], [0., 2.]],
                                  [[1., 1.], [1., 1.]],
                                  [[-1., 0.], [0., -1.]]], dtype=torch.float32)
        effective = torch.tensor([[0.1, 0.0], [0.0, 0.2]], dtype=torch.float32)
        delta = torch.tensor([0.1, 0.2], dtype=torch.float32)
        result = audit.derivative_predictions(gradients, effective, delta, 0.025)
        self.assertAlmostEqual(result["rowwise"][0].item(), 0.3, places=6)
        self.assertAlmostEqual(result["uniform"][0].item(), 0.15, places=6)
        self.assertEqual(tuple(result["ideal"].shape), (4,))

    def test_edit_fidelity_and_zero_rejection(self) -> None:
        delta = torch.tensor([1., 2.], dtype=torch.float32)
        effective = torch.tensor([[0.025, 0.05], [0.025, 0.05]], dtype=torch.float32)
        good = audit.edit_fidelity(effective, delta, 0.025)
        self.assertTrue(good["valid"])
        self.assertAlmostEqual(good["row_nonuniformity_ratio"], 0)
        self.assertFalse(audit.edit_fidelity(torch.zeros_like(effective), delta,
                                             0.025)["valid"])
        with self.assertRaisesRegex(ValueError, "finiteness"):
            audit.edit_fidelity(torch.full_like(effective, float("nan")), delta, 0.025)

    def test_site_gate_keeps_all_planned_cells(self) -> None:
        cells = [{"valid": True, "resolved": True,
                  "primary_max_cycle_error": 0.2} for _ in range(24)]
        self.assertTrue(audit.site_gate(cells)["pass"])
        cells[0] = {"valid": False, "resolved": False,
                    "primary_max_cycle_error": None}
        failed = audit.site_gate(cells)
        self.assertEqual(failed["planned"], 24)
        self.assertEqual(failed["valid"], 23)
        self.assertFalse(failed["pass"])
        with self.assertRaisesRegex(ValueError, "24"):
            audit.site_gate(cells[:-1])
        cells[0] = {"valid": True, "resolved": False,
                    "primary_max_cycle_error": float("nan")}
        with self.assertRaisesRegex(ValueError, "error"):
            audit.site_gate(cells)

    def test_bitwise_signed_zero(self) -> None:
        self.assertFalse(audit.exact_equal(torch.tensor([0.0]),
                                           torch.tensor([-0.0])))

    def test_duplicate_json_receipt_keys_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "meta.json"
            path.write_text('{"status":"valid","status":"tampered"}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                audit.json_object(path)


if __name__ == "__main__":
    unittest.main()
