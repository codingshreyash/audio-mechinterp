"""CPU synthetic fixtures and tamper tests for the CUDA saved-result auditor."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

import voxtral_old_fit_grad_leaf_cuda_audit_v1 as audit


SPEC = audit.ShapeSpec({"encoder31": (1, 4, 2), "projector": (3, 4)},
                       (3, 4), (1, 5, 4), (1, 2, 3), (6,), 3)


def fixture(case_index: int):
    ids = torch.tensor([[1, 24, 24, 24, 2]], dtype=torch.int64)
    batch = {"input_ids": ids, "attention_mask": torch.ones((1, 5), dtype=torch.int64),
             "input_features": torch.full(SPEC.features, 10 + case_index, dtype=torch.float32)}
    source_bf16 = {"encoder31": torch.ones(SPEC.source["encoder31"], dtype=torch.bfloat16),
                   "projector": torch.full(SPEC.source["projector"], 2, dtype=torch.bfloat16)}
    sources = {k: v.float() for k, v in source_bf16.items()}
    hashes = {k: audit.tensor_sha(v) for k, v in sources.items()}
    passive = []
    if case_index == 1:
        for ordinal in (0, 1):
            passive.append({"ordinal": ordinal, "sources": {k: v.clone() for k, v in sources.items()},
                            "source_sha256": hashes.copy(), "hook_counts": dict.fromkeys(audit.SITES, 1)})
        source_bf16 = {}
    full = []
    for ordinal, (site, mode, repeat) in enumerate(audit.schedule()):
        projector = sources["projector"].clone()
        lm = torch.zeros(SPEC.lm)
        lm[0, [1, 2, 3]] = projector
        value = 1.0 if mode == "G1" else 0.0
        terminal = {stage: torch.full((4,), value) for stage in audit.STAGES}
        flags = {"grad_enabled": mode.startswith("G"), "leaf_requires_grad": mode.endswith("1")}
        downstream = {stage: {"grad_enabled": mode.startswith("G"), "requires_grad": mode == "G1"}
                      for stage in ("lm_input", "lm0_pre")}
        full.append({"kind": "full", "ordinal": ordinal, "site": site, "mode": mode, "repeat": repeat,
                     "factor_observed": flags, "downstream_grad_flags": downstream,
                     "source_sha256": hashes[site], "source_post_full": sources[site].clone(),
                     "projector_full": projector, "lm_input_full": lm, "lm0_input_full": lm.clone(),
                     "lm_terminal": terminal, "first_answer_logits": torch.full((6,), value),
                     "audio_placeholder_positions": [1, 2, 3],
                     "hook_counts": dict.fromkeys(("source_override", "source_observer", "projector", "lm_input",
                                                    "lm0_pre", *audit.STAGES), 1)})
    origins = {site: next(r for r in full if r["site"] == site and r["mode"] == "N0")
               for site in audit.SITES}
    direct = []
    for index, (site, mode, repeat) in enumerate(audit.schedule()):
        origin = origins[site]
        full_row = next(r for r in full if (r["site"], r["mode"], r["repeat"]) == (site, mode, repeat))
        fixed = origin["lm_input_full"].clone()
        direct.append({"kind": "direct", "ordinal": 16 + index, "site": site, "mode": mode,
                       "repeat": repeat, "origin_mode": "N0", "origin_full_ordinal": origin["ordinal"],
                       "factor_observed": copy.deepcopy(full_row["factor_observed"]),
                       "downstream_grad_flags": {stage: {"grad_enabled": mode.startswith("G"),
                                                          "requires_grad": mode.endswith("1")}
                                                 for stage in ("lm_input", "lm0_pre")},
                       "input_sha256": audit.tensor_sha(fixed), "lm_input_full": fixed,
                       "lm0_input_full": fixed.clone(),
                       "lm_terminal": {k: v.clone() for k, v in full_row["lm_terminal"].items()},
                       "first_answer_logits": full_row["first_answer_logits"].clone(),
                       "hook_counts": dict.fromkeys(("lm0_pre", *audit.STAGES), 1)})
    payload = {"schema": "voxtral_old_fit_grad_leaf_cuda_tensors_v1",
               "case_id": audit.CASE_IDS[case_index], "case_index": case_index,
               "batch": batch, "sources_bf16": source_bf16, "sources_fp32": sources,
               "passive": passive, "calls": full + direct}
    batch_hashes = {k: audit.tensor_sha(v) for k, v in batch.items()}
    bf16_hashes = {k: audit.tensor_sha(v) for k, v in source_bf16.items()}
    return payload, batch_hashes, bf16_hashes, hashes


def write_case(root: Path, index: int):
    payload, bh, bf16, fp32 = fixture(index)
    path = root / audit.CASE_IDS[index]
    path.mkdir()
    (path / "blobs").mkdir()
    raw = path / "blobs" / "temporary.pt"
    torch.save(payload, raw)
    digest = audit.sha(raw)
    raw.rename(path / "blobs" / f"{digest}.pt")
    count = {"passive_forwards": 2 if index else 0, "full_forwards": 16,
             "direct_forwards": 16, "vjps": 0, "generations": 0}
    meta = {"schema": "voxtral_old_fit_grad_leaf_cuda_meta_v1", "status": "started",
            "case_id": audit.CASE_IDS[index], "case_index": index,
            "attempt_id": f"test-{index}", "freeze_sha256": "f" * 64,
            "protocol_v1_sha256": audit.V1_SHA256, "protocol_v2_sha256": audit.V2_SHA256,
            "schema_sha256": audit.SCHEMA_SHA256, "runner_sha256": "a" * 64,
            "environment_receipt_sha256": "b" * 64, "checkpoint_tree_sha256": "c" * 64,
            "source_rule": audit.SOURCE_RULES[index], "source_sha256": fp32 if index == 0 else {},
            "batch_sha256": bh,
            "full_schedule": audit.schedule(),
            "direct_schedule": [(site, "N0", mode, repeat) for site, mode, repeat in audit.schedule()],
            "planned_counts": count,
            "caps": {"cuda_bytes": audit.MAX_CUDA_BYTES, "rss_bytes": audit.MAX_RAM_BYTES,
                     "worker_seconds": audit.MAX_WORKER_SECONDS,
                     "worker_output_bytes": audit.MAX_CASE_BYTES - 24 * 1024**2,
                     "total_output_bytes": audit.MAX_CASE_BYTES}}
    receipt = {"schema": "voxtral_old_fit_grad_leaf_cuda_source_receipt_v1",
               "case_id": audit.CASE_IDS[index], "source_rule": audit.SOURCE_RULES[index],
               "source_fp32_sha256": fp32, "source_bf16_sha256": bf16,
               "archived_sidecar_sha256": audit.BEAVER_ARCHIVE_SHA if index == 0 else None,
               "passive_source_sha256": [fp32, fp32] if index else [],
               "passive_equal": dict.fromkeys(audit.SITES, True) if index else {}}
    layout = {"valid": True, "errors": []}
    for name, value in (("meta.json", meta), ("source_receipt.json", receipt),
                        ("model_layout.json", layout)):
        (path / name).write_text(json.dumps(value))
    phases = (["preflight_complete_before_model_load", "cuda_configured_before_model_load",
               "model_loaded"] + [f"after_passive_{i}" for i in range(2 if index else 0)] +
              ["source_frozen"] +
              [f"after_full_{site}_{mode}_{repeat}" for site, mode, repeat in audit.schedule()] +
              [f"after_direct_{site}_{mode}_{repeat}" for site, mode, repeat in audit.schedule()] +
              ["details_saved"])
    with (path / "progress.jsonl").open("w") as stream:
        for ordinal, phase in enumerate(phases):
            stream.write(json.dumps({"phase": phase, "elapsed_seconds": ordinal * .1,
                                     "cuda_allocated_bytes": 0, "cuda_reserved_bytes": 0,
                                     "process_rss_bytes": 0, "utc": "2026-11-18T00:00:00Z"}) + "\n")
    result = {"schema": "voxtral_old_fit_grad_leaf_cuda_result_v1", "status": "completed",
              "case_id": audit.CASE_IDS[index], "case_index": index,
              "attempt_id": meta["attempt_id"], "freeze_sha256": meta["freeze_sha256"],
              "protocol_v1_sha256": audit.V1_SHA256, "protocol_v2_sha256": audit.V2_SHA256,
              "schema_sha256": audit.SCHEMA_SHA256, "runner_sha256": "a" * 64,
              "meta_sha256": audit.sha(path / "meta.json"),
              "source_receipt_sha256": audit.sha(path / "source_receipt.json"),
              "model_layout_sha256": audit.sha(path / "model_layout.json"),
              "progress_sha256": audit.sha(path / "progress.jsonl"),
              "details_sha256": digest, "details_path": f"blobs/{digest}.pt",
              "details_bytes": (path / "blobs" / f"{digest}.pt").stat().st_size,
              "actual_counts": count, "elapsed_seconds": 4.0,
              "peak_cuda_allocated_bytes": 0, "peak_cuda_reserved_bytes": 0,
              "peak_process_rss_bytes": 0, "audio_token_id": 24,
              "execution_metadata": {"deterministic_algorithms": True,
                                     "deterministic_warn_only": False,
                                     "matmul_allow_tf32": False, "cudnn_allow_tf32": False,
                                     "matmul_precision": "highest",
                                     "requested_attention_implementation": "sdpa",
                                     "model_eval": True, "language_model_eval": True,
                                     "torch_version": "synthetic", "cuda_runtime_version": "synthetic",
                                     "cudnn_version": 1, "gpu_name": "synthetic", "gpu_capability": [8, 9],
                                     "platform": "synthetic", "transformers_version": "synthetic",
                                     "accelerate_version": "synthetic",
                                     "cublas_workspace_config": ":4096:8", "flash_sdp_enabled": True,
                                     "mem_efficient_sdp_enabled": True, "math_sdp_enabled": True,
                                     "config_attention_implementation": "sdpa"}}
    (path / "result.json").write_text(json.dumps(result))
    return path, bh, bf16, fp32


class SavedAuditTests(unittest.TestCase):
    def check(self, payload, case_index, bh, bf16, fp32):
        return audit.audit_payload(payload, case_index, spec=SPEC,
                                   expected_batch_hashes=(bh, bh),
                                   expected_bf16=bf16, expected_fp32=fp32)

    def test_two_cases_and_96_required_direct_full_comparisons(self):
        outputs = []
        for index in (0, 1):
            args = fixture(index)
            report = self.check(args[0], index, *args[1:])
            outputs.append(report)
            self.assertEqual(report["direct_full_comparison_count"], 48)
            self.assertEqual(report["saved_forwards"], 32 + (2 if index else 0))
            for site in audit.SITES:
                self.assertEqual(report["cells"][site]["label"], "same_pattern")
                self.assertEqual(len(report["cells"][site]["direct_full_comparisons"]), 24)
                self.assertEqual(report["cells"][site]["full"]["contrasts"]["N0_vs_G1"]
                                 ["first_above_floor_difference"], "lm0")
        self.assertEqual(sum(r["direct_full_comparison_count"] for r in outputs), 96)
        self.assertEqual(sum(r["saved_forwards"] for r in outputs), 66)

    def test_single_direct_full_signed_zero_mismatch_changes_label(self):
        payload, bh, bf16, fp32 = fixture(0)
        payload["calls"][16]["first_answer_logits"][0] = -0.0
        report = self.check(payload, 0, bh, bf16, fp32)
        self.assertEqual(report["cells"]["encoder31"]["label"], "different_pattern")
        self.assertEqual(report["cells"]["projector"]["label"], "same_pattern")
        self.assertEqual(report["cells"]["encoder31"]["direct_full_equal_count"], 23)
        mismatch = next(x for x in report["cells"]["encoder31"]["direct_full_comparisons"]
                        if not x["bitwise_equal"])
        self.assertEqual(mismatch["boundary"], "logits")
        self.assertEqual(mismatch["changed_coordinates"], 0)  # signed zero changes bytes

    def test_repeats_above_floor_and_upstream_drift(self):
        payload, bh, bf16, fp32 = fixture(0)
        payload["calls"][6]["lm_terminal"]["lm0"][0] = 8.0
        report = self.check(payload, 0, bh, bf16, fp32)
        self.assertEqual(report["cells"]["encoder31"]["label"], "different_pattern")
        self.assertFalse(report["cells"]["encoder31"]["repeats_stable"])
        payload, bh, bf16, fp32 = fixture(0)
        # Preserve handoff within this call; the cross-mode upstream state differs.
        payload["calls"][1]["lm_input_full"][0, 0, 0] = 0.5
        payload["calls"][1]["lm0_input_full"][0, 0, 0] = 0.5
        report = self.check(payload, 0, bh, bf16, fp32)
        self.assertEqual(report["cells"]["encoder31"]["label"], "different_pattern")
        self.assertEqual(report["cells"]["encoder31"]["full"]["contrasts"]["N0_vs_G1"]
                         ["first_saved_bitwise_difference"], "lm_input")
        payload, bh, bf16, fp32 = fixture(0)
        for row in payload["calls"]:
            if row["site"] == "encoder31":
                row["lm0_input_full"][0, 0, 0] = 0.25
        report = self.check(payload, 0, bh, bf16, fp32)
        self.assertEqual(report["cells"]["encoder31"]["label"], "different_pattern")
        self.assertFalse(report["cells"]["encoder31"]["lm_input_to_lm0_pre_equal"])

    def test_invalid_source_schedule_hooks_and_passive(self):
        for index in (0, 1):
            original, bh, bf16, fp32 = fixture(index)
            for mutate in (
                lambda p: p["calls"].pop(),
                lambda p: p["calls"][0]["hook_counts"].__setitem__("lm0", 0),
                lambda p: p["calls"][0].__setitem__("mode", "G1"),
                lambda p: p["calls"][0]["source_post_full"].add_(1),
                lambda p: p["calls"][16].__setitem__("origin_full_ordinal", 999),
            ):
                payload = copy.deepcopy(original)
                mutate(payload)
                with self.assertRaises((ValueError, KeyError)):
                    self.check(payload, index, bh, bf16, fp32)
            if index == 1:
                payload = copy.deepcopy(original)
                payload["passive"][1]["sources"]["encoder31"][0, 0, 0] += 1
                with self.assertRaisesRegex(ValueError, "passive source differs"):
                    self.check(payload, index, bh, bf16, fp32)

    def test_campaign_retains_four_invalid_cells_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "audit.json"
            report = audit.audit_campaign([root / "missing_beaver", root / "missing_waterfall"], output)
            self.assertEqual(len(report["cells"]), 4)
            self.assertEqual({x["label"] for x in report["cells"]}, {"invalid"})
            self.assertEqual(report["planned_forward_count"], 66)
            with self.assertRaisesRegex(ValueError, "already exists"):
                audit.audit_campaign([root / "missing_beaver", root / "missing_waterfall"], output)

    def test_saved_receipts_content_addressing_and_single_case_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases = [write_case(root, index) for index in (0, 1)]
            bh = tuple(x[1] for x in cases)
            case0 = audit.audit_one_case(cases[0][0], 0, root / "case0_audit.json", spec=SPEC,
                                         expected_batch_hashes=bh, expected_bf16=cases[0][2],
                                         expected_fp32=cases[0][3])
            self.assertEqual(case0["status"], "audited_saved_data_only")
            self.assertEqual(case0["direct_full_comparison_count"], 48)
            report = audit.audit_campaign([x[0] for x in cases], root / "campaign.json", spec=SPEC,
                                          expected_batch_hashes=bh, expected_bf16=cases[0][2],
                                          expected_fp32=cases[0][3])
            self.assertEqual(report["direct_full_comparison_count"], 96)
            self.assertEqual(report["saved_forward_count"], 66)
            self.assertEqual({x["label"] for x in report["cells"]}, {"same_pattern"})
            # A changed receipt invalidates the case even if the tensor file is unchanged.
            receipt = cases[1][0] / "source_receipt.json"
            receipt.write_bytes(receipt.read_bytes() + b" ")
            bad = audit.audit_one_case(cases[1][0], 1, root / "bad.json", spec=SPEC,
                                       expected_batch_hashes=bh, expected_bf16=cases[0][2],
                                       expected_fp32=cases[0][3])
            self.assertEqual(bad["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
