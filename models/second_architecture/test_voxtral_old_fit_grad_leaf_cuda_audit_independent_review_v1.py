"""Independent negative reproductions for the saved-data auditor source review.

These tests characterize what the current auditor accepts. They are not CUDA
evidence and do not assert that any model forward took place.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import voxtral_old_fit_grad_leaf_cuda_audit_v1 as audit
from test_voxtral_old_fit_grad_leaf_cuda_audit_v1 import SPEC, write_case


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class IdentityNegativeReproductions(unittest.TestCase):
    def _saved_case(self, root: Path) -> tuple[Path, dict]:
        path, batch, bf16, fp32 = write_case(root, 0)
        kwargs = {"spec": SPEC, "expected_batch_hashes": (batch, batch),
                  "expected_bf16": bf16, "expected_fp32": fp32}
        return path, kwargs

    def _audit(self, path: Path, kwargs: dict) -> dict:
        return audit.audit_one_case(path, 0, path.parent / "review_audit.json", **kwargs)

    def test_forged_runner_checkpoint_environment_hashes_still_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            path, kwargs = self._saved_case(Path(temp))
            meta_path, result_path = path / "meta.json", path / "result.json"
            meta, result = json.loads(meta_path.read_text()), json.loads(result_path.read_text())
            meta["runner_sha256"] = result["runner_sha256"] = "e" * 64
            meta["checkpoint_tree_sha256"] = "d" * 64
            meta["environment_receipt_sha256"] = "f" * 64
            _write_json(meta_path, meta)
            result["meta_sha256"] = audit.sha(meta_path)
            _write_json(result_path, result)
            report = self._audit(path, kwargs)
            self.assertEqual(report["status"], "audited_saved_data_only")
            self.assertEqual({x["label"] for x in report["cells"].values()}, {"same_pattern"})

    def test_unsupported_model_layout_still_audits(self):
        with tempfile.TemporaryDirectory() as temp:
            path, kwargs = self._saved_case(Path(temp))
            layout_path, result_path = path / "model_layout.json", path / "result.json"
            _write_json(layout_path, {"valid": True, "errors": [], "named_parameter_count": 0,
                                      "parameter_count": 0, "dtype_device_counts": {},
                                      "parameters": []})
            result = json.loads(result_path.read_text())
            result["model_layout_sha256"] = audit.sha(layout_path)
            _write_json(result_path, result)
            report = self._audit(path, kwargs)
            self.assertEqual(report["status"], "audited_saved_data_only")

    def test_campaign_keeps_invalid_case_in_four_cell_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            beaver, b_hash, bf16, fp32 = write_case(root, 0)
            waterfall, w_hash, _, _ = write_case(root, 1)
            (waterfall / "source_receipt.json").write_text("{}", encoding="utf-8")
            report = audit.audit_campaign([beaver, waterfall], root / "review_campaign.json",
                                          spec=SPEC, expected_batch_hashes=(b_hash, w_hash),
                                          expected_bf16=bf16, expected_fp32=fp32)
            self.assertEqual(report["planned_cell_count"], 4)
            self.assertEqual(report["planned_forward_count"], 66)
            self.assertEqual(len(report["cells"]), 4)
            self.assertEqual([c["label"] for c in report["cells"]],
                             ["same_pattern", "same_pattern", "invalid", "invalid"])


if __name__ == "__main__":
    unittest.main()
