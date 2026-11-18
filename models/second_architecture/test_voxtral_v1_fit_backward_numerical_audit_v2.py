"""CPU synthetic checks for amended parameter-layout receipts."""

import json
import tempfile
import unittest
from pathlib import Path

import voxtral_v1_fit_backward_numerical_audit_v2 as audit


def synthetic_layout(arm: str):
    ordinary = "torch.bfloat16" if arm == "bf16" else "torch.float32"
    params = [{"name": f"model.fake.{i}", "dtype": ordinary, "device": "mps:0",
               "shape": [1], "numel": 1} for i in range(761)]
    params.append({"name": audit.POSITION_NAME, "dtype": "torch.float32", "device": "mps:0",
                   "shape": audit.POSITION_SHAPE, "numel": 1500 * 1280})
    counts = ({"torch.float32/mps": 762} if arm == "fp32" else
              {"torch.bfloat16/mps": 761, "torch.float32/mps": 1})
    return {"arm": arm, "valid": True, "errors": [],
            "named_parameter_count": 762, "parameter_count": 761 + 1500 * 1280,
            "dtype_device_counts": counts,
            "expected_fp32_exception": audit.POSITION_NAME if arm == "bf16" else None,
            "parameters": params}


class ModelLayoutAuditTests(unittest.TestCase):
    def evaluate(self, arm: str, change=None):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            layout = synthetic_layout(arm)
            if change:
                change(layout)
            (path / "model_layout.json").write_text(json.dumps(layout, allow_nan=False) + "\n")
            digest = audit.sha(path / "model_layout.json")
            meta = {"expected_parameter_layout": {
                "named_parameter_count": 762,
                "fp32_exception": audit.POSITION_NAME if arm == "bf16" else None,
                "fp32_exception_shape": audit.POSITION_SHAPE if arm == "bf16" else None,
                "all_parameters_on": "mps"},
                "modeling_utils_sha256": audit.MODELING_UTILS_SHA}
            result = {"model_layout_sha256": digest,
                      "model_layout_counts": layout["dtype_device_counts"]}
            entries = [{"phase": "model_loaded", "model_layout_sha256": digest,
                        "parameter_count": layout["parameter_count"],
                        "named_parameter_count": 762}]
            return audit.audit_model_layout(path, arm, meta, result, entries)

    def test_clean_bf16_and_fp32_layouts(self):
        for arm in ("bf16", "fp32"):
            with self.subTest(arm=arm):
                self.assertEqual(self.evaluate(arm)["named_parameter_count"], 762)

    def test_extra_fp32_and_missing_position_are_rejected(self):
        def extra(layout):
            layout["parameters"][0]["dtype"] = "torch.float32"
            layout["dtype_device_counts"] = {"torch.bfloat16/mps": 760,
                                              "torch.float32/mps": 2}
        with self.assertRaisesRegex(ValueError, "model layout dtype"):
            self.evaluate("bf16", extra)

        def missing(layout):
            layout["parameters"][-1]["name"] = "model.fake.pos"
        with self.assertRaisesRegex(ValueError, "model layout dtype|sole exception"):
            self.evaluate("bf16", missing)

    def test_bad_parameter_shape_and_source_pins_fail(self):
        def bad_shape(layout):
            layout["parameters"][0]["shape"] = [2]
        with self.assertRaisesRegex(ValueError, "device/shape/numel"):
            self.evaluate("fp32", bad_shape)
        self.assertEqual(audit.source_context()["meta"]["record_key"],
                         "animal_08|piper|fit_chosen")


if __name__ == "__main__":
    unittest.main()
