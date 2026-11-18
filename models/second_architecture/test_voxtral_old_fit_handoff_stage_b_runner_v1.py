"""CPU-only admission and fixed-count tests for the prospective Stage-B driver."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import voxtral_old_fit_handoff_stage_b_runner_v1 as driver


def fake_pairs():
    sources = {"encoder31": torch.zeros((1, 1500, 1280), dtype=torch.bfloat16),
               "projector": torch.zeros((375, 3072), dtype=torch.bfloat16)}
    lm = torch.zeros((1, 396, 3072), dtype=torch.bfloat16)
    pairs = []
    for site in ("encoder31", "projector"):
        for alpha in driver.replay.ALPHAS:
            edited = sources[site].clone()
            if site == "encoder31":
                edited[0, 96, 0] = alpha
            else:
                edited[24, 0] = alpha
            pairs.append({"site": site, "cycle": 0, "alpha": alpha,
                          "zero": {"source_full_post": {site: sources[site]},
                                   "lm_input_full": lm,
                                   "lm_terminal": {"lm29": torch.zeros(3072)},
                                   "first_answer_logits": torch.zeros(1)},
                          "edited": {"source_full_post": {site: edited},
                                     "lm_input_full": lm,
                                     "lm_terminal": {"lm29": torch.zeros(3072)},
                                     "first_answer_logits": torch.zeros(1)}})
    return pairs


def fake_replay_forward(_model, _batch, *, site, full_source, probes, grad=False,
                        self_clone=False, device="cpu"):
    del _model, _batch, probes, grad, self_clone, device
    scalar = float(full_source[0, 96, 0] if site == "encoder31" else full_source[24, 0])
    values = torch.zeros(3072)
    values[0] = scalar
    saved = {"source_sha256": driver.native.first.tensor_sha(full_source),
             "projector_word": torch.zeros(9216), "lm_input_word": torch.zeros(9216),
             "lm_terminal": {f"lm{i}": values.clone() for i in driver.native.LM_LAYERS},
             "first_answer_logits": torch.tensor([scalar]),
             "audio_placeholder_positions": list(range(375))}
    return {"saved": saved, "graph_source": object(), "graph_targets": object()}


def fake_vjps(_trace, _probes, site):
    return {stage: torch.zeros((count, len(driver.replay.ROWS[site]),
                                1280 if site == "encoder31" else 3072))
            for stage, count in driver.replay.SOURCE_REPLAY_STAGES[site]}


class StageBDriverTests(unittest.TestCase):
    def test_real_failed_bf16_posthoc_and_fp32_continuation_are_bound(self):
        bound = driver.verify_stage_a_manifests()
        self.assertEqual(bound["original_bf16_launch_status"], "launch_failed")
        self.assertEqual(bound["stage_a_model_work_seconds"], 218.5633733337745)
        self.assertEqual(bound["arms"]["bf16"]["audit_path"], driver.continuation.POSTHOC_AUDIT)
        self.assertEqual(bound["arms"]["fp32"]["manifest_sha256"],
                         "3605289cdefe106db9075033dedd9b20dff9cb1f106fd5e9771a243a40bdaa3f")
        self.assertEqual(bound["arms"]["fp32"]["result"]["actual_counts"],
                         driver.stage_a_runner.planned_counts())
        self.assertEqual(driver.phase_preflight("bf16_direct_lm", stage_a=bound)[2]
                         ["estimated_tensor_payload_bytes"], 140550144)
        with self.assertRaisesRegex(ValueError, "Bound file absent"):
            driver.verify_bf16_stage_b_manifest(bound)

    def test_fp32_manifest_in_memory_tamper_and_missing_review_fail_closed(self):
        original = driver.read_json
        def changed_manifest(path):
            value = original(path)
            if path == driver.STAGE_A_MANIFESTS["fp32"]:
                value["files"][0]["sha256"] = "0" * 64
            return value
        with mock.patch.object(driver, "read_json", side_effect=changed_manifest):
            with self.assertRaisesRegex(ValueError, "raw file changed"):
                driver.verify_stage_a_manifests()
        with mock.patch.object(driver, "STAGE_A_FP32_RESULT_REVIEW_SHA", "0" * 64):
            with self.assertRaisesRegex(ValueError, "independent result review lacks GO"):
                driver.verify_stage_a_manifests()

    def test_one_shot_attempt_and_symlinked_bound_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fp32/once").mkdir(parents=True)
            self.assertEqual(driver._one_attempt(root, "fp32", "once"), root / "fp32/once")
            (root / "fp32/twice").mkdir()
            with self.assertRaisesRegex(ValueError, "retried"):
                driver._one_attempt(root, "fp32", "once")
            file = root / "source.json"
            copy_path = root / "copy.json"
            copy_path.write_text("same bytes\n")
            file.symlink_to(copy_path.name)
            with self.assertRaisesRegex(ValueError, "symlink"):
                driver.committed_same(root, file)

    def test_parent_elapsed_must_cover_worker_and_respect_phase_cap(self):
        self.assertEqual(driver.checked_parent_elapsed(12.5, 12.0, 600), 12.5)
        for parent, worker in ((11.0, 12.0), (600.0, 12.0),
                               (float("nan"), 12.0), (12.0, float("inf")),
                               (True, 1.0), (12.0, 0.0)):
            with self.subTest(parent=parent, worker=worker):
                with self.assertRaisesRegex(ValueError, "parent/worker elapsed"):
                    driver.checked_parent_elapsed(parent, worker, 600)

    def test_source_is_closed_without_external_review_or_stage_a(self):
        with self.assertRaisesRegex(ValueError, "external freeze absent"):
            driver.required_live_freeze()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "attempt"
            with mock.patch.object(driver, "verify_stage_a_manifests",
                                   side_effect=FileNotFoundError("Stage A absent")):
                with self.assertRaisesRegex(FileNotFoundError, "Stage A absent"):
                    driver.run_phase("bf16_direct_lm", output,
                                     model_factory=lambda _: self.fail("model constructed"),
                                     device="cpu")
            self.assertFalse(output.exists())
        with mock.patch.object(sys, "argv", ["runner", "--run", "--phase", "bf16_direct_lm"]):
            with self.assertRaisesRegex(ValueError, "external freeze absent"):
                driver.main()

    def test_cpu_fake_phase_writes_exclusive_content_addressed_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "runs/bf16_direct_lm/once"
            stage_a = {"stage_a_model_work_seconds": 100.0,
                       "arms": {arm: {"manifest_sha256": arm * 8,
                                      "manifest": {"result_sha256": arm * 8,
                                                   "audit_sha256": arm * 8}}
                                for arm in ("bf16", "fp32")}}
            fake_layout = {"valid": True, "named_parameter_count": 1}
            with mock.patch.object(driver, "RUNS", root / "runs"), \
                 mock.patch.object(driver, "STAGE_B_BF16_MANIFEST", root / "absent.json"), \
                 mock.patch.object(driver, "verify_stage_a_manifests", return_value=stage_a), \
                 mock.patch.object(driver, "phase_preflight", return_value=({}, (), {
                     "estimated_tensor_payload_bytes": 100, "per_process_output_cap_bytes": driver.MAX_B_BYTES})), \
                 mock.patch.object(driver.native, "preflight_cpu", return_value={"batches": {"beaver": {}}}), \
                 mock.patch.object(driver.prior, "parameter_layout", return_value=fake_layout), \
                 mock.patch.object(driver, "direct_replay", return_value=([{"fake": True}], 24)):
                result = driver.run_phase("bf16_direct_lm", output,
                                          model_factory=lambda _dtype: torch.nn.Linear(1, 1),
                                          device="cpu")
            self.assertEqual(result["actual_counts"], driver.COUNTS["bf16_direct_lm"])
            self.assertEqual(result["details_path"], f"blobs/{result['details_sha256']}.pt")
            self.assertEqual(driver.sha(output / result["details_path"]), result["details_sha256"])
            self.assertEqual(driver.sha(output / "meta.json"), result["meta_sha256"])
            self.assertEqual(driver.sha(output / "progress.jsonl"), result["progress_sha256"])
            self.assertFalse((output / "details.pt").exists())
            with self.assertRaises(FileExistsError):
                driver.run_phase("bf16_direct_lm", output,
                                 model_factory=lambda _dtype: self.fail("model constructed"),
                                 device="cpu")

    def test_fixed_phase_counts_and_order(self):
        self.assertEqual(driver.planned_counts("bf16_direct_lm"),
                         {"direct_lm_forwards": 24, "source_pair_forwards": 0,
                          "source_parity_forwards": 0, "source_vjps": 0,
                          "full_model_override_forwards": 0})
        self.assertEqual(driver.planned_counts("fp32_replay"),
                         {"direct_lm_forwards": 24, "source_pair_forwards": 24,
                          "source_parity_forwards": 6, "source_vjps": 68,
                          "full_model_override_forwards": 4})
        with self.assertRaisesRegex(ValueError, "Unknown"):
            driver.planned_counts("other")

    def test_direct_replay_exact_24_and_native_parity(self):
        pairs = fake_pairs()
        calls = []
        def direct(_model, _batch, lm_input, dtype, *, device="cpu"):
            calls.append((dtype, device, driver.native.first.tensor_sha(lm_input)))
            return {"lm_terminal": {"lm29": torch.zeros(3072)},
                    "first_answer_logits": torch.zeros(1),
                    "input_sha256": driver.native.first.tensor_sha(lm_input)}
        with mock.patch.object(driver.replay, "direct_lm_forward", side_effect=direct):
            rows, count = driver.direct_replay(None, {}, pairs, torch.bfloat16,
                                               device="cpu", parity_native=True)
        self.assertEqual(count, 24)
        self.assertEqual(len(calls), 24)
        self.assertEqual([(r["site"], r["alpha"]) for r in rows],
                         [(site, alpha) for site in ("encoder31", "projector")
                          for alpha in driver.replay.ALPHAS])
        self.assertEqual(len(rows[0]["zero"]["input"].shape), 3)

    def test_fp32_source_parity_pairs_and_68_vjps(self):
        pairs = fake_pairs()
        calls = []
        def forward(*args, **kwargs):
            calls.append((kwargs["site"], kwargs.get("grad", False),
                          kwargs.get("self_clone", False)))
            return fake_replay_forward(*args, **kwargs)
        with mock.patch.object(driver.replay, "source_replay_forward", side_effect=forward), \
             mock.patch.object(driver.replay, "replay_source_vjps", side_effect=fake_vjps):
            output, counts = driver.fp32_source_replays(None, {}, pairs, {}, device="cpu")
        self.assertEqual(counts, {"source_pair_forwards": 24,
                                  "source_parity_forwards": 6, "source_vjps": 68})
        self.assertEqual(len(output["pairs"]), 12)
        self.assertEqual(len(calls), 30)
        self.assertEqual(calls[:3], [("encoder31", True, False),
                                     ("encoder31", False, False),
                                     ("encoder31", False, True)])
        self.assertEqual(calls[15:18], [("projector", True, False),
                                        ("projector", False, False),
                                        ("projector", False, True)])
        bad = copy.deepcopy(pairs)
        changed = bad[1]["zero"]["source_full_post"]["encoder31"].clone()
        changed[0, 96, 0] = 2
        bad[1]["zero"]["source_full_post"]["encoder31"] = changed
        with mock.patch.object(driver.replay, "source_replay_forward", side_effect=forward), \
             mock.patch.object(driver.replay, "replay_source_vjps", side_effect=fake_vjps):
            with self.assertRaisesRegex(ValueError, "clean replay source varies"):
                driver.fp32_source_replays(None, {}, bad, {}, device="cpu")

    def test_fp32_override_checks_four_and_rejects_mismatch(self):
        pairs = fake_pairs()
        direct = []
        for site in ("encoder31", "projector"):
            direct.append({"site": site,
                           "zero": {"input_sha256": "a", "replayed": {
                               "lm_terminal": {"lm29": torch.zeros(3072)},
                               "first_answer_logits": torch.zeros(1)}},
                           "edited": {"input_sha256": "a", "replayed": {
                               "lm_terminal": {"lm29": torch.zeros(3072)},
                               "first_answer_logits": torch.zeros(1)}}})
        with mock.patch.object(driver.replay, "full_model_lm_override",
                               return_value={"lm_terminal": {"lm29": torch.zeros(3072)},
                                             "first_answer_logits": torch.zeros(1)}) as full:
            records, count = driver.fp32_lm_validation(None, {}, pairs, direct, device="cpu")
        self.assertEqual(count, 4)
        self.assertEqual(len(records), 4)
        self.assertEqual(full.call_count, 4)
        with mock.patch.object(driver.replay, "full_model_lm_override",
                               return_value={"lm_terminal": {"lm29": torch.ones(3072)},
                                             "first_answer_logits": torch.zeros(1)}):
            with self.assertRaisesRegex(ValueError, "lm29"):
                driver.fp32_lm_validation(None, {}, pairs, direct, device="cpu")


if __name__ == "__main__":
    unittest.main()
