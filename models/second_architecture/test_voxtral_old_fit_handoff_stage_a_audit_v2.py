"""CPU-only regression for the Stage-A audit-only MPS device spelling amendment."""

from __future__ import annotations

import unittest

import voxtral_old_fit_handoff_stage_a_audit_v1 as v1
import voxtral_old_fit_handoff_stage_a_audit_v2 as v2


def layout(arm: str, device: str) -> dict:
    dtype = "torch.bfloat16" if arm == "bf16" else "torch.float32"
    rows = [{"name": f"layer.{i}", "dtype": dtype, "device": device,
             "shape": [1], "numel": 1} for i in range(761)]
    rows.append({"name": "model.audio_tower.embed_positions.weight",
                 "dtype": "torch.float32", "device": device,
                 "shape": [1500, 1280], "numel": 1500 * 1280})
    counts = {f"{dtype}/mps": 761, "torch.float32/mps": 1} if arm == "bf16" else {"torch.float32/mps": 762}
    return {"valid": True, "arm": arm, "named_parameter_count": 762,
            "parameters": rows, "dtype_device_counts": counts}


class DeviceSpellingAmendmentTests(unittest.TestCase):
    def test_saved_mps_zero_spelling_is_newly_accepted(self):
        for arm in ("bf16", "fp32"):
            with self.subTest(arm=arm):
                candidate = layout(arm, "mps:0")
                with self.assertRaisesRegex(ValueError, "model parameter dtype/device/shape differs"):
                    v1.validate_layout(candidate, arm)
                v2.validate_layout(candidate, arm)
                v2.validate_layout(layout(arm, "mps"), arm)

    def test_other_devices_and_existing_layout_checks_still_reject(self):
        for device in ("cpu", "mps:1", "cuda:0", "MPS:0", "mps:00"):
            with self.subTest(device=device):
                with self.assertRaisesRegex(ValueError, "model parameter dtype/device/shape differs"):
                    v2.validate_layout(layout("bf16", device), "bf16")
        candidate = layout("bf16", "mps:0")
        candidate["parameters"][0]["dtype"] = "torch.float32"
        with self.assertRaisesRegex(ValueError, "model parameter dtype/device/shape differs"):
            v2.validate_layout(candidate, "bf16")
        candidate = layout("bf16", "mps:0")
        candidate["parameters"][-1]["shape"] = [1280, 1500]
        with self.assertRaisesRegex(ValueError, "audio position table shape changed"):
            v2.validate_layout(candidate, "bf16")
        candidate = layout("bf16", "mps:0")
        candidate["dtype_device_counts"]["torch.bfloat16/mps"] = 760
        with self.assertRaisesRegex(ValueError, "model dtype/device counts differ"):
            v2.validate_layout(candidate, "bf16")


if __name__ == "__main__":
    unittest.main()
