"""Synthetic CPU receipts only; no model, MPS, ASR, or active run output."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

import voxtral_v1_fit_backward_result_audit as audit


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def phases() -> list[str]:
    sequence = ["attempt_started", "preflight_validated", "mps_ready", "before_model_load",
                "model_loaded", "natural_generation", "passive_generation", "no_grad_prompt",
                "donor_no_grad_prompt", "recipient_shared_graph"]
    for coord in audit.COORDS:
        sequence += [f"before_vjp_{coord}", f"after_vjp_{coord}"]
    for site in audit.SITES:
        sequence += [f"zero_patch_{site}", f"alpha_patch_{site}"]
    return sequence + ["details_saved"]


def hook_records(prompt_len: int) -> dict:
    return {"encoder31": {"shape": [1, 1500, 1280], "dtype": "torch.bfloat16"},
            "projector": {"shape": [375, 3072], "dtype": "torch.bfloat16"},
            "lm29": {"shape": [1, prompt_len, 3072], "dtype": "torch.bfloat16"}}


def synthetic_completed(root: Path) -> tuple[Path, dict]:
    context = audit.source_context()
    tree = context["snapshot_tree"]["files"]
    snapshot = {"revision": "3060fe34b35ba5d44202ce9ff3c097642914f8f3",
                "model_id": "mistralai/Voxtral-Mini-3B-2507",
                "config": {"sha256": "368124c9a9171b3a6c0cdb35cd7fedeff465eefecf315e2a621684bbed7dcd7e", "size_bytes": tree["config.json"]["size"]},
                "snapshot_tree": {"sha256": audit.PIN["snapshot_tree"], "size_bytes": audit.SNAPSHOT_TREE.stat().st_size},
                "weight_index": {"sha256": "7506794dd65e4324685f9634b3d83f20d36a723543650c1e6b72d81965289930", "size_bytes": tree["model.safetensors.index.json"]["size"]},
                "processor_config": {"sha256": context["index"]["processor_config_sha256"], "size_bytes": tree["preprocessor_config.json"]["size"]},
                "tokenizer_asset": {"sha256": context["index"]["tokenizer_asset_sha256"], "size_bytes": tree["tekken.json"]["size"]},
                "generation_config": {"sha256": "cebeed3d4a1680c9b311863385d32b91008480dfbf8cf7abc2142447b8f73b76", "size_bytes": tree["generation_config.json"]["size"]},
                "weight_shards": {name: {"sha256": tree[name]["lfs_sha256"],
                                         "size_bytes": tree[name]["lfs_size"]}
                                  for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")}}
    run = root / "synthetic_old_fit_attempt"
    run.mkdir()
    log = []
    for i, phase in enumerate(phases()):
        entry = {"phase": phase, "elapsed_seconds": i / 10}
        if i == 0:
            entry.update({"wall_utc": "2026-11-18T00:00:00+00:00",
                          "script_sha256": audit.PIN["pilot_source"],
                          "protocol_sha256": audit.PIN["protocol"]})
        if phase == "preflight_validated":
            entry["installed_sources"] = audit.SOURCE_HASHES
        if i >= 3:
            entry.update({"driver_bytes": 100, "current_bytes": 50,
                          "peak_driver_bytes": 100})
        log.append(entry)
    (run / "progress.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
    meta = {"purpose": "old_v1_fit_backward_feasibility_only",
            "script_sha256": audit.PIN["pilot_source"],
            "protocol_sha256": audit.PIN["protocol"],
            "first_runner_sha256": audit.PIN["first_runner"],
            "frontend_helper_sha256": audit.PIN["frontend"],
            "first_v2_fit_pilot_sha256": audit.PIN["saved_pilot"],
            "masks_sha256": audit.PIN["masks"],
            "llama_modeling_sha256": audit.PIN["llama"],
            "installed_sources": audit.SOURCE_HASHES,
            "model_snapshot": snapshot,
            "versions": {"torch": "2.15.0", "transformers": "5.20.2"},
            "record_key": "animal_08|piper|fit_chosen", "recipient": "beaver",
            "donor": "monkey", "bin": "word_0", "alpha": audit.ALPHA,
            "target_coordinates": list(audit.COORDS), "wav_hashes": context["wav_hashes"],
            "source_rows": context["rows"], "saved_natural_suffix_ids": context["saved_suffix"],
            "fresh_v2_corpus_opened": False, "cap_seconds": audit.TIME_CAP,
            "cap_mps_driver_bytes": audit.MEMORY_CAP,
            "preflight_elapsed_seconds": log[1]["elapsed_seconds"],
            "mps_recommended_max_bytes": 40 * 1024**3,
            "mps_fraction_limit": 0.75}
    write_json(run / "meta.json", meta)
    clean = torch.zeros(3072, dtype=torch.float32)
    no_grad = torch.zeros_like(clean)
    details = {"source_states": {}, "donor_states": {}, "selected_gradients": {},
               "effective_edits": {}, "uniform_means": {}, "intended_edits": {},
               "clean_terminal": clean, "no_grad_terminal": no_grad,
               "measured_terminals": {}}
    result = {"status": "completed_numerical_feasibility_only",
              "scope": "one old FIT pair/direction, one bin, four raw coordinates; no v2 or efficacy",
              "meta_sha256": audit.sha(run / "meta.json"),
              "progress_sha256": audit.sha(run / "progress.jsonl"),
              "model_load_seconds": 0.1, "elapsed_seconds": 3.0,
              "peak_mps_driver_bytes": 1000, "peak_mps_current_bytes": 500,
              "vjp_seconds": [0.1] * 4,
              "natural_suffix_ids": context["saved_suffix"],
              "passive_suffix_ids": context["saved_suffix"],
              "passive_hook_shapes": {"encoder31": [[1, 1500, 1280]],
                                      "projector": [[375, 3072]],
                                      "lm29": [[1, context["prompt_len"], 3072]]},
              "no_grad_hook_events": hook_records(context["prompt_len"]),
              "shared_graph_hook_events": hook_records(context["prompt_len"]),
              "shared_graph_vjp_connected_both_sources": True,
              "raw_source_vjp_dtypes": {s: ["torch.bfloat16"] for s in audit.SITES},
              "selected_gradient_storage_dtype": "torch.float32",
              "contraction_dtype": "torch.float32",
              "source_shapes": {s: list(audit.SHAPES[s]) for s in audit.SITES},
              "terminal_shape": [3072], "no_grad_minus_grad_four": [0.0] * 4,
              "source_hashes": {}, "donor_hashes": {}, "measured_terminal_hashes": {},
              "edits": {}, "predictions": {}, "measured": {}}
    for site in audit.SITES:
        shape = audit.SHAPES[site]
        rows = context["rows"][site]
        rec = torch.zeros(shape, dtype=torch.bfloat16)
        donor = torch.zeros(shape, dtype=torch.bfloat16)
        rec[rows] = 1.0
        donor[rows] = 2.0
        grad = torch.full((4, len(rows), shape[1]), 0.125, dtype=torch.float32)
        rec_rows = rec[rows].float()
        intended = audit.ALPHA * (donor[rows].float().mean(0)-rec_rows.mean(0))
        effective = (rec_rows+intended).to(torch.bfloat16).float()-rec_rows
        mean = effective.mean(0)
        details["source_states"][site] = rec
        details["donor_states"][site] = donor
        details["selected_gradients"][site] = grad
        details["effective_edits"][site] = effective
        details["uniform_means"][site] = mean
        details["intended_edits"][site] = intended
        zero = torch.zeros_like(clean)
        alpha = torch.zeros_like(clean)
        alpha[list(audit.COORDS)] = torch.tensor([0.1, 0.2, 0.3, 0.4])
        details["measured_terminals"][site] = {"zero": zero, "alpha": alpha}
        result["source_hashes"][site] = audit.tensor_sha(rec)
        result["donor_hashes"][site] = audit.tensor_sha(donor)
        result["measured_terminal_hashes"][site] = {"zero": audit.tensor_sha(zero),
                                                      "alpha": audit.tensor_sha(alpha)}
        intended_norm = float(torch.linalg.vector_norm(intended))
        realized_norm = float(torch.linalg.vector_norm(effective))
        fidelity = float(torch.linalg.vector_norm(mean-intended)) / max(intended_norm, 1e-6)
        nonuniformity = (float(torch.linalg.vector_norm(effective-mean)) / len(rows)**0.5 /
                         max(float(torch.linalg.vector_norm(mean)), 1e-6))
        result["edits"][site] = {"intended_norm": intended_norm,
                                  "realized_norm": realized_norm,
                                  "mean_fidelity_ratio": fidelity,
                                  "row_nonuniformity_ratio": nonuniformity,
                                  "mean_fidelity_over_10pct": fidelity > 0.1,
                                  "row_nonuniformity_over_10pct": nonuniformity > 0.1,
                                  "effective_sha256": audit.tensor_sha(effective)}
        predicted = {"uniform_mean": [], "rowwise_oracle": [], "ideal_uniform": []}
        for k in audit.COORDS:
            g = grad[k]
            predicted["uniform_mean"].append(float(torch.sum(g.sum(0)*mean)))
            predicted["rowwise_oracle"].append(float(torch.sum(g*effective)))
            predicted["ideal_uniform"].append(float(torch.sum(g.sum(0)*intended)))
        result["predictions"][site] = predicted
        delta = (alpha-zero)[list(audit.COORDS)].tolist()
        result["measured"][site] = {"zero_hook_events": hook_records(context["prompt_len"]),
                                    "alpha_hook_events": hook_records(context["prompt_len"]),
                                    "delta_four": delta, "zero_minus_clean_four": [0.0]*4,
                                    "uniform_error": audit.vector_error(predicted["uniform_mean"],delta),
                                    "rowwise_error": audit.vector_error(predicted["rowwise_oracle"],delta),
                                    "ideal_error": audit.vector_error(predicted["ideal_uniform"],delta)}
    torch.save(details, run / "details.pt")
    result["details_sha256"] = audit.sha(run / "details.pt")
    write_json(run / "result.json", result)
    return run, result


class OldFitBackwardResultAuditTests(unittest.TestCase):
    def test_source_validate_only(self):
        self.assertEqual(audit.source_context()["rows"]["projector"], [24, 25, 26])

    def test_failed_attempt_receipt_is_not_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            start = {"phase": "attempt_started", "elapsed_seconds": 0.0,
                     "wall_utc": "2026-11-18T00:00:00+00:00",
                     "script_sha256": audit.PIN["pilot_source"],
                     "protocol_sha256": audit.PIN["protocol"]}
            (run / "progress.jsonl").write_text(json.dumps(start) + "\n")
            write_json(run / "error.json", {"status": "failed", "error_type": "RuntimeError",
                                           "error": "synthetic failure", "elapsed_seconds": 0.1,
                                           "progress_sha256": audit.sha(run / "progress.jsonl")})
            self.assertEqual(audit.audit(run, allow_external_test_dir=True)["status"],
                             "audited_failed_attempt_no_numerical_result")

    def test_completed_recomputation_and_tamper_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, result = synthetic_completed(Path(temporary))
            self.assertEqual(audit.audit(run, allow_external_test_dir=True)["status"],
                             "audited_completed_numerical_feasibility_only")
            result["predictions"]["projector"]["rowwise_oracle"][2] += 1.0
            write_json(run / "result.json", result)
            with self.assertRaisesRegex(ValueError, "numerical mismatch"):
                audit.audit(run, allow_external_test_dir=True)

    def test_measured_terminal_hash_tamper_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, result = synthetic_completed(Path(temporary))
            result["measured_terminal_hashes"]["encoder31"]["alpha"] = "0"*64
            write_json(run / "result.json", result)
            with self.assertRaisesRegex(ValueError, "terminal hash mismatch"):
                audit.audit(run, allow_external_test_dir=True)

    def test_nonfinite_and_resource_tamper_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, result = synthetic_completed(Path(temporary))
            result["vjp_seconds"][0] = float("nan")
            write_json(run / "result.json", result)
            with self.assertRaisesRegex(ValueError, "nonfinite JSON"):
                audit.audit(run, allow_external_test_dir=True)
            result["vjp_seconds"][0] = 0.1
            result["peak_mps_driver_bytes"] = audit.MEMORY_CAP
            write_json(run / "result.json", result)
            with self.assertRaisesRegex(ValueError, "MPS driver cap"):
                audit.audit(run, allow_external_test_dir=True)

    def test_external_path_rejected_by_public_auditor(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, _ = synthetic_completed(Path(temporary))
            with self.assertRaisesRegex(ValueError, "outside old-FIT backward run root"):
                audit.audit(run)


if __name__ == "__main__":
    unittest.main()
