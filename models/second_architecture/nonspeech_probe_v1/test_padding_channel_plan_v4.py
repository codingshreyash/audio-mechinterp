"""No-weight check that V4 preserves V3 schedule and updates only evidence size."""
import unittest

from models.second_architecture.nonspeech_probe_v1 import padding_channel_plan_v1 as v3
from models.second_architecture.nonspeech_probe_v1 import padding_channel_plan_v4 as v4


class V4BudgetTests(unittest.TestCase):
    def test_schedule_identity_and_full_decoder_shape(self) -> None:
        self.assertEqual(v4.schedule(), v3.schedule())
        self.assertEqual(v4.schedule_sha256(),
                         "d939618689e989a2dfe249cc6d4f0d3b32a532c184991680b77f11b4af6a0ace")
        self.assertEqual(v4.FIRST_DECODER_POSITIONS, 4)
        self.assertEqual(v4.DECODER_WIDTH, 384)
        self.assertEqual(v4.worst_case_core_evidence_bytes(), 1_377_212_576)
        self.assertTrue(v4.resource_budget_fits())


if __name__ == "__main__":
    unittest.main()
