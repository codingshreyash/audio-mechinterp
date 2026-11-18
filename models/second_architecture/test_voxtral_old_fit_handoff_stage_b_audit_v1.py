"""Synthetic saved-data and tamper tests; no model load or inference."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import voxtral_old_fit_handoff_stage_b_audit_v1 as audit


def fixture():
    source = {site: torch.zeros(shape, dtype=torch.bfloat16)
              for site, shape in audit.SHAPES.items()}
    lm = torch.zeros((1, 396, 3072), dtype=torch.bfloat16)
    logits_bf16 = torch.zeros(131072, dtype=torch.bfloat16)
    logits_fp32 = torch.zeros(131072, dtype=torch.float32)
    positions = list(range(375))
    probes = {"lm": torch.zeros((3072, 16), dtype=torch.float64),
              "projector_word": torch.zeros((9216, 4), dtype=torch.float64),
              "lm_input_word": torch.zeros((9216, 4), dtype=torch.float64)}
    native = []
    direct = []
    source_records = []
    for site in audit.SITES:
        for alpha in audit.ALPHAS:
            edited_source = source[site].clone()
            if site == "encoder31":
                edited_source[0, 96, 0] = alpha
                delta = edited_source.float()[0, audit.ROWS[site]]
            else:
                edited_source[24, 0] = alpha
                delta = edited_source.float()[audit.ROWS[site]]
            zero_source = source[site]
            native_trace = {
                "lm_input_full": lm,
                "lm_terminal": {stage: torch.zeros(3072, dtype=torch.bfloat16)
                                for stage in audit.LMS},
                "first_answer_logits": logits_bf16,
            }
            native.append({"site": site, "alpha": alpha,
                           "zero": {**native_trace, "source_full_post": {site: zero_source}},
                           "edited": {**native_trace, "source_full_post": {site: edited_source}}})
            def direct_item():
                observed = {"mode": "direct_lm", "input_sha256": audit.tensor_sha(lm.float()),
                            "lm_terminal": {stage: torch.zeros(3072, dtype=torch.float32)
                                            for stage in audit.LMS},
                            "first_answer_logits": logits_fp32}
                return {"input": lm, "input_sha256": audit.tensor_sha(lm),
                        "replayed": observed}
            direct.append({"site": site, "cycle": 0, "alpha": alpha,
                           "zero": direct_item(), "edited": direct_item()})
            def source_item(value, *, edited=False):
                word = torch.zeros(9216, dtype=torch.float32)
                if site == "projector" and edited:
                    word = delta.reshape(-1).clone()
                trace = {"source_sha256": audit.tensor_sha(value),
                         "projector_word": word, "lm_input_word": word.clone(),
                         "lm_terminal": {stage: torch.zeros(3072, dtype=torch.float32)
                                         for stage in audit.LMS},
                         "first_answer_logits": logits_fp32,
                         "audio_placeholder_positions": positions}
                return {"input": value, "input_sha256": audit.tensor_sha(value),
                        "trace": trace}
            source_records.append({"site": site, "cycle": 0, "alpha": alpha,
                                   "zero": source_item(zero_source),
                                   "edited": source_item(edited_source, edited=True),
                                   "effective_edit": delta,
                                   "terminal_delta_l2": 0.0})
    parities = {}
    vjps = {}
    for site in audit.SITES:
        clean = next(row for row in source_records if row["site"] == site)["zero"]["trace"]
        parities[site] = {"clean": clean, "repeat": clean, "self": clean}
        vjps[site] = {stage: torch.zeros((n, len(audit.ROWS[site]),
                                         audit.SHAPES[site][-1]), dtype=torch.float32)
                      for stage, n in audit.TARGETS[site].items()}
    return native, direct, {"parity": parities, "vjps": vjps, "pairs": source_records}, probes


class AuditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native, cls.direct, cls.source, cls.probes = fixture()

    def test_fp32_source_audit_and_projector_identity(self):
        report = audit.check_source(self.source, self.native, self.probes)
        self.assertEqual(report["vjps"], 68)
        self.assertEqual(report["projector_self_identity_pairs"], 6)
        self.assertEqual(len(report["cells"]), 12)
        self.assertEqual(len(report["stage_summary"]["projector"]), 5)
        self.assertFalse(report["cells"][0]["stages"]["lm29"]["resolved"])

    def test_projector_self_identity_tamper(self):
        bad = copy.deepcopy(self.source)
        target = bad["pairs"][6]["edited"]["trace"]
        target["projector_word"][0] += 1
        target["lm_input_word"][0] += 1
        with self.assertRaisesRegex(ValueError, "Projector self derivative identity"):
            audit.check_source(bad, self.native, self.probes)

    def test_source_input_hash_and_exact_pair_tamper(self):
        bad = copy.deepcopy(self.source)
        bad["pairs"][0]["edited"]["input_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Source input hash"):
            audit.check_source(bad, self.native, self.probes)
        bad = copy.deepcopy(self.source)
        bad["pairs"][0]["effective_edit"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "effective edit"):
            audit.check_source(bad, self.native, self.probes)

    def test_source_vjp_count_parity_and_metric_tamper(self):
        bad = copy.deepcopy(self.source)
        bad["vjps"]["projector"]["lm29"] = bad["vjps"]["projector"]["lm29"][:15]
        with self.assertRaisesRegex(ValueError, "VJP"):
            audit.check_source(bad, self.native, self.probes)
        bad = copy.deepcopy(self.source)
        bad["parity"]["encoder31"]["repeat"] = copy.deepcopy(
            bad["parity"]["encoder31"]["repeat"])
        bad["parity"]["encoder31"]["repeat"]["lm_terminal"]["lm0"][0] = 1
        with self.assertRaisesRegex(ValueError, "parity"):
            audit.check_source(bad, self.native, self.probes)
        bad = copy.deepcopy(self.source)
        bad["pairs"][0]["terminal_delta_l2"] = 12
        with self.assertRaisesRegex(ValueError, "terminal delta"):
            audit.check_source(bad, self.native, self.probes)

    def test_source_vjp_contraction_and_signed_odd_even(self):
        source = copy.deepcopy(self.source)
        probes = copy.deepcopy(self.probes)
        probes["lm_input_word"][0, 0] = 1
        source["vjps"]["projector"]["lm_input_word"][0, 0, 0] = 1
        report = audit.check_source(source, self.native, probes)
        plus = report["cells"][6]["stages"]["lm_input_word"]
        minus = report["cells"][7]["stages"]["lm_input_word"]
        self.assertAlmostEqual(plus["measured"][0], plus["predicted"][0])
        self.assertAlmostEqual(minus["measured"][0], minus["predicted"][0])
        self.assertAlmostEqual(plus["residual_l2"], 0)
        self.assertTrue(plus["resolved"])
        odd = report["stage_summary"]["projector"]["lm_input_word"][
            "measured_odd_even"]["0.00625"]
        self.assertAlmostEqual(odd["even"][0], 0)
        self.assertAlmostEqual(odd["odd"][0], plus["measured"][0])

    def test_discordant_paired_zero_sets_null_resolution(self):
        source = copy.deepcopy(self.source)
        probes = copy.deepcopy(self.probes)
        probes["lm"][0, 0] = 1
        discordant = source["pairs"][1]
        discordant["zero"]["trace"]["lm_terminal"]["lm29"][0] = 1
        discordant["terminal_delta_l2"] = 1.0
        report = audit.check_source(source, self.native, probes)
        summary = report["stage_summary"]["encoder31"]["lm29"]
        cell = report["cells"][1]["stages"]["lm29"]
        self.assertEqual(summary["zero_repeat_spread_l2"], 1.0)
        self.assertEqual(summary["resolution_l2"], 3.0)
        self.assertEqual(cell["measured_l2"], 1.0)
        self.assertFalse(cell["resolved"])
        self.assertIsNone(cell["relative_l2_error_if_resolved"])

    def test_direct_lm_hash_and_order(self):
        report = audit.check_direct(self.direct, self.native, "fp32_replay")
        self.assertEqual(report["forwards"], 24)
        bad = copy.deepcopy(self.direct)
        bad[0]["zero"]["replayed"]["input_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "input hash"):
            audit.check_direct(bad, self.native, "fp32_replay")
        bad = copy.deepcopy(self.direct)
        bad[0]["alpha"] *= -1
        with self.assertRaisesRegex(ValueError, "signed pair order"):
            audit.check_direct(bad, self.native, "fp32_replay")

    def test_bf16_direct_native_parity(self):
        records = copy.deepcopy(self.direct)
        for record in records:
            for label in ("zero", "edited"):
                trace = record[label]["replayed"]
                trace["lm_terminal"] = {stage: tensor.bfloat16()
                                        for stage, tensor in trace["lm_terminal"].items()}
                trace["first_answer_logits"] = trace["first_answer_logits"].bfloat16()
                trace["input_sha256"] = audit.tensor_sha(record[label]["input"])
        audit.check_direct(records, self.native, "bf16_direct_lm")
        records[0]["zero"]["replayed"]["lm_terminal"]["lm0"][0] = 1
        with self.assertRaisesRegex(ValueError, "bitwise parity"):
            audit.check_direct(records, self.native, "bf16_direct_lm")

    def test_fp32_four_override_validation(self):
        records = []
        for site in audit.SITES:
            direct = next(r for r in self.direct if r["site"] == site)
            for label in ("zero", "edited"):
                full = copy.deepcopy(direct[label]["replayed"])
                full["mode"] = "full_model_lm_override"
                records.append({"site": site, "label": label, "alpha": direct["alpha"],
                                "input_sha256": direct[label]["input_sha256"],
                                "full": full})
        self.assertEqual(audit.check_overrides(records, self.direct)[
            "full_model_override_forwards"], 4)
        records[0]["full"]["first_answer_logits"][0] = 1
        with self.assertRaisesRegex(ValueError, "bitwise parity"):
            audit.check_overrides(records, self.direct)

    def test_progress_forward_order_and_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            rows = [{"phase": phase, "elapsed_seconds": i + .1, "mps_driver_bytes": 1}
                    for i, phase in enumerate(audit.expected_progress("fp32_replay"))]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            report = audit.check_progress(path, "fp32_replay", len(rows) + 1, 1)
            self.assertEqual(report["progress_lines"], len(rows))
            rows[3], rows[4] = rows[4], rows[3]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaisesRegex(ValueError, "chronology"):
                audit.check_progress(path, "fp32_replay", len(rows) + 1, 1)

    def test_fixed_probe_and_archive_estimate_tamper(self):
        probes = {"lm": audit._rademacher("lm_residual_16", 3072, 16),
                  "projector_word": audit._rademacher("projector_word_4", 9216, 4),
                  "lm_input_word": audit._rademacher("lm_input_word_4", 9216, 4)}
        audit.check_probes(probes)
        probes["lm"][0, 0] *= -1
        with self.assertRaisesRegex(ValueError, "Fixed probe bank"):
            audit.check_probes(probes)
        reported = {}
        with self.assertRaisesRegex(ValueError, "archive estimate"):
            audit.check_archive_estimate(self.native, reported)

    def test_failed_manifest_raw_sidecar_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            here = Path(tmp)
            attempt_id = "20261118T131755298331Z"
            prefix = f"runs/voxtral_old_fit_handoff_stage_a_launch_v1/bf16/{attempt_id}/"
            files = []
            for i in range(12):
                relative = prefix + (f"runner/blobs/{i:064x}.pt" if i < 2 else f"receipt_{i}.txt")
                path = here / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"saved {i}".encode())
                files.append({"path": relative, "bytes": path.stat().st_size,
                              "sha256": audit.sha(path), "git_committed": i >= 2})
            manifest = {"schema": "voxtral_stage_a_bf16_failed_attempt_manifest_v1",
                        "status": "frozen_auditor_rejected_no_scientific_interpretation",
                        "arm": "bf16", "attempt_id": attempt_id,
                        "local_large_sidecars_required": True, "files": files}
            with mock.patch.object(audit, "committed_same"):
                self.assertEqual(len(audit.verify_manifest_files(manifest, here=here)), 12)
                (here / files[0]["path"]).write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "raw file differs"):
                    audit.verify_manifest_files(manifest, here=here)

    def test_attempt_sidecar_and_resource_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp) / "attempt"
            (attempt / "blobs").mkdir(parents=True)
            phase = "bf16_direct_lm"
            binding = {key: {"bf16": "a" * 64, "fp32": "b" * 64}
                       for key in ("manifest", "result", "audit")}
            meta = {"status": "started_old_fit_handoff_stage_b_replay_only",
                    "phase": phase, "protocol_sha256": audit.PROTOCOL_SHA,
                    "script_sha256": audit.DRIVER_SHA,
                    "replay_primitives_sha256": audit.PRIMITIVES_SHA,
                    "stage_a_manifest_sha256": binding["manifest"],
                    "stage_a_result_sha256": binding["result"],
                    "stage_a_audit_sha256": binding["audit"],
                    "planned_counts": audit.COUNTS[phase], "archive_estimate": {},
                    "max_output_bytes": audit.BYTE_CAP,
                    "max_mps_driver_bytes": audit.DRIVER_CAP,
                    "process_cap_seconds": audit.CAP_SECONDS[phase],
                    "cumulative_cap_seconds": audit.TOTAL_SECONDS,
                    "fresh_v2_corpus_opened": False,
                    "source_rows": audit.ROWS, "cycle": 0,
                    "alphas": list(audit.ALPHAS),
                    "prior_model_work_seconds": 63.245136791840196}
            (attempt / "meta.json").write_text(json.dumps(meta))
            (attempt / "model_layout.json").write_text("{}")
            phases = audit.expected_progress(phase)
            (attempt / "progress.jsonl").write_text("\n".join(json.dumps({
                "phase": name, "elapsed_seconds": i / 10,
                "mps_driver_bytes": 1}) for i, name in enumerate(phases)) + "\n")
            records = {"source_replay": None, "direct_lm": None,
                       "lm_override_validation": None}
            raw = attempt / "details.pt"
            torch.save(records, raw)
            digest = audit.sha(raw)
            detail_path = attempt / "blobs" / f"{digest}.pt"
            raw.rename(detail_path)
            result = {"status": "completed_old_fit_handoff_stage_b_replay_only",
                      "phase": phase, "protocol_sha256": audit.PROTOCOL_SHA,
                      "script_sha256": audit.DRIVER_SHA,
                      "stage_a_manifest_sha256": binding["manifest"],
                      "stage_a_result_sha256": binding["result"],
                      "stage_a_audit_sha256": binding["audit"],
                      "meta_sha256": audit.sha(attempt / "meta.json"),
                      "model_layout_sha256": audit.sha(attempt / "model_layout.json"),
                      "progress_sha256": audit.sha(attempt / "progress.jsonl"),
                      "details_sha256": digest, "details_path": f"blobs/{digest}.pt",
                      "details_bytes": detail_path.stat().st_size,
                      "planned_counts": audit.COUNTS[phase],
                      "actual_counts": audit.COUNTS[phase],
                      "prior_model_work_seconds": 63.245136791840196,
                      "elapsed_seconds": 5.0, "peak_mps_driver_bytes": 1}
            (attempt / "result.json").write_text(json.dumps(result))
            with mock.patch.object(audit, "check_layout"), \
                 mock.patch.object(audit, "check_archive_estimate"), \
                 mock.patch.object(audit, "check_probes"), \
                 mock.patch.object(audit, "check_direct", return_value={"forwards": 24}):
                checked = audit.check_attempt(attempt, self.native, self.probes,
                    stage_a_binding=binding, prior_model_work_seconds=63.245136791840196)
                self.assertEqual(checked["actual_counts"], audit.COUNTS[phase])
                detail_path.write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "sidecar hash/size"):
                    audit.check_attempt(attempt, self.native, self.probes,
                        stage_a_binding=binding, prior_model_work_seconds=63.245136791840196)


if __name__ == "__main__":
    unittest.main()
