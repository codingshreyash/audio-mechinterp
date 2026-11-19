"""No-weight frozen-count regression for the prospective FIT amendment."""
import unittest

from models.second_architecture import wav2vec2_fit_noise_norm_causal_88_plan_v1 as plan_source


class ExecutionPlanTests(unittest.TestCase):
    def test_all_unique_stock_waveforms_and_cells_are_frozen(self):
        plan = plan_source.execution_plan()
        self.assertEqual(len(plan["pairs"]), 8)
        self.assertEqual(len(plan["stock_waveforms"]), 12)
        self.assertEqual(len(plan["stage_ii_cases"]), 4)
        self.assertEqual(plan["scored_forward_count"], 48)
        self.assertEqual(plan["stage_i_stock_no_hook_count"], 12)
        self.assertEqual(plan["stage_i_stock_read_only_count"], 12)
        self.assertEqual(plan["stage_ii_shared_no_hook_count"], 8)
        self.assertEqual(plan["stage_ii_shared_read_only_capture_count"], 8)
        self.assertEqual(plan["total_forward_ceiling"], 88)
        for pair in plan["pairs"]:
            self.assertIn([pair["case"], "clean"], plan["stock_waveforms"])
            self.assertIn([pair["case"], pair["noise_arm"]], plan["stock_waveforms"])


if __name__ == "__main__":
    unittest.main()
