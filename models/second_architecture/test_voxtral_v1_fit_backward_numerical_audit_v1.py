"""Synthetic CPU artifacts only; no model, MPS, or fresh v2 corpus."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

import voxtral_v1_fit_backward_numerical_audit_v1 as audit
import voxtral_v1_fit_backward_numerical_followup_v1 as producer


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def make_synthetic(root: Path, arm: str) -> Path:
    context = audit.source_context()
    path = root / arm
    path.mkdir()
    dtype = torch.bfloat16 if arm == "bf16" else torch.float32
    old_meta = context["meta"]
    events = {"encoder31": {"calls": 1, "shape": [1, 1500, 1280], "dtype": str(dtype)},
              "projector": {"calls": 1, "shape": [375, 3072], "dtype": str(dtype)},
              "lm29": {"calls": 1, "shape": [1, 396, 3072], "dtype": str(dtype)}}
    log = []
    patch_count = 0
    for index, phase in enumerate(audit.progress_expected()):
        item = {"phase": phase, "elapsed_seconds": index / 10}
        if index == 0:
            item.update({"arm": arm, "wall_utc": "2026-11-18T00:00:00+00:00",
                         "script_sha256": audit.PINS["runner"],
                         "protocol_sha256": audit.PINS["protocol"]})
        if index == 2:
            item.update({"recommended_bytes": 40 * 1024**3, "fraction_limit": 0.75})
        if index >= 3:
            item.update({"driver_bytes": 1000, "current_bytes": 500, "peak_driver_bytes": 1000})
        if phase.startswith("after_") and ("_zero_" in phase or "_alpha_" in phase):
            patch_count += 1
            item["patch_count"] = patch_count
        log.append(item)
    (path / "progress.jsonl").write_text("".join(json.dumps(e) + "\n" for e in log))
    meta = {"arm": arm, "purpose": "old_fit_numerical_diagnostic_only",
            "script_sha256": audit.PINS["runner"], "protocol_sha256": audit.PINS["protocol"],
            "old_result_sha256": audit.PINS["old_result"],
            "old_details_sha256": audit.PINS["old_details"],
            "source_hashes": audit.SOURCE_HASHES, "rows": audit.ROWS,
            "wav_hashes": old_meta["wav_hashes"],
            "model_snapshot": old_meta["model_snapshot"],
            "fresh_v2_corpus_opened": False,
            "direction_hashes": {s: audit.tensor_sha(context["directions"][s]) for s in audit.SITES},
            "max_seconds": audit.CAP[arm], "max_mps_driver_bytes": audit.MEMORY_CAP,
            "mps_fraction_limit": 0.75,
            "planned_patch_forwards": 72, "planned_pairs": 36,
            "planned_other_prompt_forwards": 5, "planned_vjps": 4,
            "planned_generations": 2}
    write_json(path / "meta.json", meta)
    clean = torch.ones(3072)
    details = {"directions": context["directions"], "selected_gradients_fp32": {},
               "clean_terminal_fp32": clean, "donor_terminal_fp32": clean.clone(),
               "no_grad_terminals_fp32": clean.repeat(3, 1), "observations": {}}
    analysis = {}
    for site in audit.SITES:
        rows = audit.ROWS[site]
        direction = context["directions"][site]
        gradient = torch.full((4, len(rows), audit.WIDTH[site]), 0.00001, dtype=torch.float32)
        details["selected_gradients_fp32"][site] = gradient
        observations = []
        for cycle, order in enumerate(audit.CYCLES):
            for alpha in order:
                source = torch.ones((len(rows), audit.WIDTH[site]), dtype=dtype)
                changed = producer.realized_edit(source, direction, alpha, dtype)
                prediction = producer.contract(gradient, changed)
                zero = clean.to(dtype).float().clone()
                altered = (zero + torch.tensor([alpha / 10, 0, 0, 0] + [0] * 3068)).to(dtype).float()
                observations.append({"site": site, "cycle": cycle, "alpha": alpha,
                    "zero_terminal": zero, "alpha_terminal": altered,
                    "zero_source_rows": source.clone(), "alpha_source_rows": source.clone(),
                    "effective_edit": changed["effective"],
                    "zero_sha": audit.tensor_sha(zero.to(dtype)),
                    "alpha_sha": audit.tensor_sha(altered.to(dtype)),
                    "zero_source_sha": audit.tensor_sha(source),
                    "alpha_source_sha": audit.tensor_sha(source),
                    "zero_ulp_four": producer.ulp_spacing(zero[:4].to(dtype)),
                    "zero_hook_events": copy.deepcopy(events),
                    "alpha_hook_events": copy.deepcopy(events),
                    "prediction": prediction,
                    "edit_metrics": {"effective_norm": changed["effective_norm"],
                       "mean_fidelity_ratio": changed["mean_fidelity_ratio"],
                       "row_nonuniformity_ratio": changed["row_nonuniformity_ratio"],
                       "unchanged_fraction": changed["unchanged_fraction"]}})
        details["observations"][site] = observations
        analysis[site] = producer.analyze_site(observations, gradient)
    torch.save(details, path / "details.pt")
    result = {"status": "completed_old_fit_numerical_diagnostic_only", "arm": arm,
              "scope": "one old v1 FIT case; no v2 transfer or efficacy",
              "meta_sha256": audit.sha(path / "meta.json"),
              "progress_sha256": audit.sha(path / "progress.jsonl"),
              "details_sha256": audit.sha(path / "details.pt"),
              "patch_forward_count": 72, "zero_nonzero_pair_count": 36,
              "paired_observations_per_site": 18,
              "selected_gradient_storage_dtype": "torch.float32",
              "source_element_unchanged_label":
                  "unchanged_by_bf16_cast" if arm == "bf16" else "unchanged_by_fp32_addition_rounding",
              "natural_suffix_ids": old_meta["saved_natural_suffix_ids"] if arm == "bf16" else [7, 2],
              "passive_suffix_ids": old_meta["saved_natural_suffix_ids"] if arm == "bf16" else [7, 2],
              "passive_hook_shapes": {"encoder31": [[1, 1500, 1280]],
                                      "projector": [[375, 3072]], "lm29": [[1, 396, 3072]]},
              "clean_hook_events": events,
              "raw_vjp_dtypes": {s: [str(dtype)] * 4 for s in audit.SITES},
              "elapsed_seconds": log[-1]["elapsed_seconds"] + 0.1,
              "peak_mps_driver_bytes": 1000, "peak_mps_current_bytes": 500,
              "clean_terminal_sha256": audit.tensor_sha(clean),
              "gradient_hashes": {s: audit.tensor_sha(details["selected_gradients_fp32"][s])
                                  for s in audit.SITES},
              "clean_source_hashes": {s: "a" * 64 for s in audit.SITES},
              "donor_source_hashes": {s: "b" * 64 for s in audit.SITES},
              "analysis": analysis}
    write_json(path / "result.json", result)
    return path


class NumericalAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.templates = {arm: make_synthetic(root, arm) for arm in audit.CAP}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def clone(self, arm: str) -> tuple[tempfile.TemporaryDirectory, Path]:
        temp = tempfile.TemporaryDirectory()
        destination = Path(temp.name) / arm
        shutil.copytree(self.templates[arm], destination)
        return temp, destination

    def test_both_arms_audit(self):
        for arm in audit.CAP:
            with self.subTest(arm=arm):
                self.assertEqual(audit.audit(self.templates[arm], arm)["pairs_verified"], 36)

    def test_resigned_prediction_tamper_fails(self):
        temp, path = self.clone("bf16")
        with temp:
            details = torch.load(path / "details.pt", weights_only=True)
            details["observations"]["encoder31"][0]["prediction"]["rowwise"][0] += 0.2
            torch.save(details, path / "details.pt")
            result = audit.json_read(path / "result.json")
            result["details_sha256"] = audit.sha(path / "details.pt")
            write_json(path / "result.json", result)
            with self.assertRaisesRegex(ValueError, "predictions"):
                audit.audit(path, "bf16")

    def test_resigned_edit_and_terminal_tampering_fail(self):
        for field, expected in (("effective_edit", "realized edit"),
                                ("zero_terminal", "terminal hash")):
            with self.subTest(field=field):
                temp, path = self.clone("fp32")
                with temp:
                    details = torch.load(path / "details.pt", weights_only=True)
                    details["observations"]["projector"][0][field].view(-1)[0] += 0.1
                    torch.save(details, path / "details.pt")
                    result = audit.json_read(path / "result.json")
                    result["details_sha256"] = audit.sha(path / "details.pt")
                    write_json(path / "result.json", result)
                    with self.assertRaisesRegex(ValueError, expected):
                        audit.audit(path, "fp32")

    def test_median_and_resource_tampering_fail(self):
        temp, path = self.clone("bf16")
        with temp:
            result = audit.json_read(path / "result.json")
            result["analysis"]["encoder31"]["alpha_groups"]["0.025"]["median_relative_l2_error"] += 0.1
            write_json(path / "result.json", result)
            with self.assertRaisesRegex(ValueError, "median_relative_l2_error"):
                audit.audit(path, "bf16")
            result["analysis"]["encoder31"]["alpha_groups"]["0.025"]["median_relative_l2_error"] -= 0.1
            result["peak_mps_driver_bytes"] = audit.MEMORY_CAP
            write_json(path / "result.json", result)
            with self.assertRaisesRegex(ValueError, "resource bounds"):
                audit.audit(path, "bf16")

    def test_missing_pair_and_progress_fails(self):
        temp, path = self.clone("fp32")
        with temp:
            details = torch.load(path / "details.pt", weights_only=True)
            details["observations"]["projector"].pop()
            torch.save(details, path / "details.pt")
            result = audit.json_read(path / "result.json")
            result["details_sha256"] = audit.sha(path / "details.pt")
            write_json(path / "result.json", result)
            with self.assertRaisesRegex(ValueError, "observation count"):
                audit.audit(path, "fp32")
            (path / "progress.jsonl").write_text("{}\n")
            with self.assertRaises(ValueError):
                audit.audit(path, "fp32")

    def test_nonfinite_and_wrong_dtype_fail(self):
        temp, path = self.clone("bf16")
        with temp:
            details = torch.load(path / "details.pt", weights_only=True)
            details["observations"]["encoder31"][0]["zero_terminal"][0] = float("nan")
            torch.save(details, path / "details.pt")
            result = audit.json_read(path / "result.json")
            result["details_sha256"] = audit.sha(path / "details.pt")
            write_json(path / "result.json", result)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                audit.audit(path, "bf16")
        with self.assertRaises(ValueError):
            audit.audit(self.templates["bf16"], "fp32")

    def test_error_receipt_and_missing_receipt(self):
        temp, path = self.clone("bf16")
        with temp:
            (path / "result.json").unlink()
            write_json(path / "error.json", {"status": "resource_limited_arm", "arm": "bf16",
                "progress_sha256": audit.sha(path / "progress.jsonl"),
                "error_type": "MemoryError", "error": "synthetic",
                "elapsed_seconds": 30.0})
            self.assertEqual(audit.audit(path, "bf16")["status"],
                             "audited_failed_attempt_receipt_only")
            (path / "error.json").unlink()
            with self.assertRaisesRegex(ValueError, "exactly one"):
                audit.audit(path, "bf16")


if __name__ == "__main__":
    unittest.main()
