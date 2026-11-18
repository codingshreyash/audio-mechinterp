"""CPU checks for the narrow Voxtral BF16 positional-table exception."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import voxtral_v1_fit_backward_numerical_followup_v2 as followup


def fake_model(arm, *, exception_dtype=None, extra_fp32=False, wrong_device=False,
               missing_position=False):
    ordinary = torch.bfloat16 if arm == "bf16" else torch.float32
    params = []
    for i in range(761 if not missing_position else 762):
        dtype = torch.float32 if extra_fp32 and i == 0 else ordinary
        device = "cpu" if wrong_device and i == 0 else "mps"
        params.append((f"model.fake.{i}", SimpleNamespace(
            dtype=dtype, device=SimpleNamespace(type=device, __str__=lambda self: device),
            shape=(1,), numel=lambda: 1)))
    if not missing_position:
        dtype = exception_dtype or torch.float32
        params.append((followup.POSITION_NAME, SimpleNamespace(
            dtype=dtype, device=SimpleNamespace(type="mps", __str__=lambda self: "mps:0"),
            shape=tuple(followup.POSITION_SHAPE),
            numel=lambda: followup.POSITION_SHAPE[0] * followup.POSITION_SHAPE[1])))
    return SimpleNamespace(named_parameters=lambda: params)


class BF16AmendmentTests(unittest.TestCase):
    def test_exact_bf16_layout_is_accepted_and_receipted(self):
        layout = followup.parameter_layout(fake_model("bf16"), "bf16")
        self.assertTrue(layout["valid"], layout["errors"])
        self.assertEqual(layout["named_parameter_count"], 762)
        self.assertEqual(layout["dtype_device_counts"],
                         {"torch.bfloat16/mps": 761, "torch.float32/mps": 1})
        self.assertEqual(layout["expected_fp32_exception"], followup.POSITION_NAME)
        self.assertEqual(layout["parameters"][-1]["shape"], followup.POSITION_SHAPE)

    def test_additional_fp32_or_wrong_device_is_rejected(self):
        for changes in ({"extra_fp32": True}, {"wrong_device": True},
                        {"exception_dtype": torch.bfloat16}, {"missing_position": True}):
            with self.subTest(changes=changes):
                self.assertFalse(followup.parameter_layout(fake_model("bf16", **changes),
                                                            "bf16")["valid"])

    def test_full_fp32_layout_and_bf16_contamination(self):
        self.assertTrue(followup.parameter_layout(fake_model("fp32"), "fp32")["valid"])
        bad = fake_model("fp32")
        bad.named_parameters()[0][1].dtype = torch.bfloat16
        self.assertFalse(followup.parameter_layout(bad, "fp32")["valid"])

    def test_grid_and_caps_remain_unchanged(self):
        self.assertEqual(followup.CYCLES[0],
                         (0.00625, -0.00625, 0.025, -0.025, 0.1, -0.1))
        self.assertEqual(followup.CYCLES[1], tuple(reversed(followup.CYCLES[0])))
        self.assertEqual(followup.ARMS, {"bf16": ("bfloat16", 900),
                                         "fp32": ("float32", 1200)})
        self.assertEqual(followup.MAX_DRIVER, 30 * 1024**3)

    def test_preflight_failure_has_exclusive_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "attempt"
            with patch.object(followup, "preflight", side_effect=ValueError("fixture")):
                with self.assertRaisesRegex(ValueError, "fixture"):
                    followup.run("bf16", output)
            self.assertTrue((output / "error.json").is_file())
            with self.assertRaises(FileExistsError):
                followup.run("bf16", output)


if __name__ == "__main__":
    unittest.main()
