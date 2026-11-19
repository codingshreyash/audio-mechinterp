"""No-weight array tests for the exact two-site intervention algebra."""
import unittest

import numpy as np

from models.second_architecture.nonspeech_probe_v1 import padding_channel_core_v1 as core
from models.second_architecture.nonspeech_probe_v1 import padding_channel_plan_v1 as plan


class PaddingCoreTests(unittest.TestCase):
    def test_factorial_all_sites_and_lengths(self) -> None:
        for case_id in ("d02_beep_train_low", "d08_beep_train_high"):
            for site in plan.SITES:
                shape = core.expected_shape(site)
                active = plan.geometry(case_id, site)["active"]
                native = np.zeros(shape, np.float32)
                donor = np.full(shape, 2.0, np.float32)
                donor[core.time_slice(site, 0, active)] = 1.0
                before = donor.copy()
                for arm in ("RD", "DR", "DD", "self"):
                    row = {"case_id": case_id, "site": site,
                           "arm": arm, "hook_policy": "edit"}
                    result, meta = core.edit(native, native, donor, row, site, "detect")
                    expected = native.copy()
                    if arm in ("DR", "DD"):
                        expected[core.time_slice(site, 0, active)] = 1.0
                    if arm in ("RD", "DD"):
                        expected[core.time_slice(site, active, shape[-1 if site == "log_mel" else 1])] = 2.0
                    self.assertTrue(np.array_equal(result, expected), (case_id, site, arm))
                    self.assertEqual(meta["support"], {
                        "RD": [active, shape[-1 if site == "log_mel" else 1]],
                        "DR": [0, active],
                        "DD": [0, shape[-1 if site == "log_mel" else 1]],
                        "self": [0, shape[-1 if site == "log_mel" else 1]],
                    }[arm])
                self.assertTrue(np.array_equal(donor, before))
                self.assertTrue(np.array_equal(native, np.zeros(shape, np.float32)))

    def test_phase_and_random_support(self) -> None:
        case_id, site = "d02_beep_train_low", "log_mel"
        native = np.zeros(core.expected_shape(site), np.float32)
        donor = np.ones_like(native)
        split = {"case_id": case_id, "site": site,
                 "arm": "padding_detect_only", "hook_policy": "edit"}
        unchanged, meta = core.edit(native, native, donor, split, site, "transcribe")
        self.assertTrue(np.array_equal(unchanged, native))
        self.assertEqual(meta["support"], [0, 0])
        changed, _ = core.edit(native, native, donor, split, site, "detect")
        self.assertTrue(np.array_equal(changed[:, :, :200], native[:, :, :200]))
        self.assertTrue(np.array_equal(changed[:, :, 200:], donor[:, :, 200:]))
        random = {"case_id": case_id, "site": site, "arm": "random_padding_1",
                  "hook_policy": "edit", "seeds": {"detect": 123, "transcribe": 456}}
        edited, detail = core.edit(native, native, donor, random, site, "detect")
        self.assertTrue(np.array_equal(edited[:, :, :200], native[:, :, :200]))
        self.assertEqual(detail["seed"], 123)
        self.assertAlmostEqual(detail["realized_l2"], detail["target_l2"], delta=1e-4)
        again, _ = core.edit(native, native, donor, random, site, "detect")
        self.assertTrue(np.array_equal(edited, again))
        zero, detail = core.edit(native, native, native, random, site, "detect")
        self.assertTrue(np.array_equal(zero, native))
        self.assertEqual(detail["target_l2"], 0)

    def test_processed_suppression_mask_and_shape(self) -> None:
        a = np.array([0.5, -np.inf, -1.0], np.float32)
        b = np.array([0.500001, -np.inf, -1.0], np.float32)
        self.assertLess(core.processed_score_maxabs(a, b), 1e-5)
        with self.assertRaisesRegex(ValueError, "suppression mask"):
            core.processed_score_maxabs(a, np.array([0.5, -2.0, -1.0], np.float32))
        for bad in (np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "NaN/\\+inf"):
                core.processed_score_maxabs(a, np.array([0.5, bad, -1.0], np.float32))
        with self.assertRaisesRegex(ValueError, "finite parity"):
            core.maxabs_finite(a, b)
        core.assert_array(np.zeros((1, 1500, 384), np.float32), "pre_block0")
        with self.assertRaisesRegex(ValueError, "tensor invalid"):
            core.assert_array(np.zeros((1, 1500, 383), np.float32), "pre_block0")


if __name__ == "__main__":
    unittest.main()
