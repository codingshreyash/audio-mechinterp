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
    def test_two_audited_stage_a_manifests_bind_exact_files_and_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            freeze = root / "freeze.json"
            freeze.write_text(json.dumps({
                "status": "frozen_old_fit_handoff_stage_a_v1",
                "protocol_sha256": driver.replay.PROTOCOL_SHA256,
                "runner_sha256": driver.STAGE_A_RUNNER_SHA,
                "native_source_sha256": driver.replay.NATIVE_SOURCE_SHA256,
                "auditor_sha256": driver.STAGE_A_AUDITOR_SHA}))
            frozen_sha = driver.sha(freeze)
            manifests = {}
            for arm, elapsed in (("bf16", 20.0), ("fp32", 30.0)):
                attempt = runs / arm / "once"
                runner = attempt / "runner"
                blobs = runner / "blobs"
                blobs.mkdir(parents=True)
                for name in ("meta.json", "model_layout.json", "progress.jsonl"):
                    (runner / name).write_text("{}\n")
                result = {"status": driver.stage_a_runner.RESULT_STATUS, "arm": arm,
                          "script_sha256": driver.STAGE_A_RUNNER_SHA,
                          "native_source_sha256": driver.replay.NATIVE_SOURCE_SHA256,
                          "protocol_sha256": driver.replay.PROTOCOL_SHA256,
                          "parity_passed": True, "elapsed_seconds": elapsed - 1,
                          "planned_counts": driver.stage_a_runner.planned_counts(),
                          "actual_counts": driver.stage_a_runner.planned_counts()}
                for name in ("meta", "model_layout", "progress"):
                    path = runner / ("progress.jsonl" if name == "progress" else f"{name}.json")
                    result[f"{name}_sha256"] = driver.sha(path)
                for name in ("probes", "details"):
                    payload = f"{arm}:{name}".encode()
                    digest = driver.hashlib.sha256(payload).hexdigest()
                    (blobs / f"{digest}.pt").write_bytes(payload)
                    result[f"{name}_sha256"] = digest
                    result[f"{name}_path"] = f"blobs/{digest}.pt"
                    result[f"{name}_bytes"] = len(payload)
                (runner / "result.json").write_text(json.dumps(result))
                audit = {"status": driver.STAGE_A_AUDIT_STATUS, "arm": arm,
                         "result_sha256": driver.sha(runner / "result.json"),
                         "auditor_sha256": driver.STAGE_A_AUDITOR_SHA,
                         "runner_sha256": driver.STAGE_A_RUNNER_SHA,
                         "native_source_sha256": driver.replay.NATIVE_SOURCE_SHA256,
                         "protocol_sha256": driver.replay.PROTOCOL_SHA256,
                         "probes_sha256": result["probes_sha256"],
                         "details_sha256": result["details_sha256"]}
                (attempt / "audit.json").write_text(json.dumps(audit))
                start = {"arm": arm, "freeze_receipt_sha256": frozen_sha,
                         "runner_sha256": driver.STAGE_A_RUNNER_SHA,
                         "auditor_sha256": driver.STAGE_A_AUDITOR_SHA,
                         "wall_utc": "2026-11-18T01:00:00+00:00" if arm == "bf16"
                         else "2026-11-18T03:00:00+00:00"}
                finish = {"status": driver.STAGE_A_LAUNCH_STATUS, "arm": arm,
                          "freeze_receipt_sha256": frozen_sha,
                          "post_run_source_verified": True, "runner_returncode": 0,
                          "auditor_returncode": 0, "runner_result_sha256": driver.sha(runner / "result.json"),
                          "audit_sha256": driver.sha(attempt / "audit.json"),
                          "model_work_elapsed_seconds": elapsed,
                          "wall_utc": "2026-11-18T02:00:00+00:00" if arm == "bf16"
                          else "2026-11-18T04:00:00+00:00"}
                (attempt / "launch_start.json").write_text(json.dumps(start))
                (attempt / "launch_finish.json").write_text(json.dumps(finish))
                manifest = root / f"{arm}.json"
                manifest.write_text(json.dumps({
                    "status": driver.STAGE_A_LAUNCH_STATUS, "arm": arm,
                    "attempt_relative_path": "once", "freeze_receipt_sha256": frozen_sha,
                    "launch_start_sha256": driver.sha(attempt / "launch_start.json"),
                    "launch_finish_sha256": driver.sha(attempt / "launch_finish.json"),
                    "result_sha256": driver.sha(runner / "result.json"),
                    "audit_sha256": driver.sha(attempt / "audit.json"),
                    "model_work_elapsed_seconds": elapsed}))
                manifests[arm] = manifest
            with mock.patch.object(driver, "committed_same"), \
                 mock.patch.object(driver, "strictly_later"):
                bound = driver.verify_stage_a_manifests(repo=root, runs=runs,
                                                         freeze=freeze, manifests=manifests)
                self.assertEqual(bound["stage_a_model_work_seconds"], 50)
                blob = next((runs / "bf16/once/runner/blobs").iterdir())
                original = blob.read_bytes()
                blob.write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "blob differs"):
                    driver.verify_stage_a_manifests(repo=root, runs=runs,
                                                     freeze=freeze, manifests=manifests)
                blob.write_bytes(original)
                # A second attempt is a permanent no-retry failure.
                (runs / "fp32/extra").mkdir()
                with self.assertRaisesRegex(ValueError, "retried"):
                    driver.verify_stage_a_manifests(repo=root, runs=runs,
                                                     freeze=freeze, manifests=manifests)

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
