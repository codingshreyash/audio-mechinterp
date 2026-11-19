"""No-weight V5 feature-sidecar budget and reviewed schedule identity."""
import unittest

from models.second_architecture.nonspeech_probe_v1 import padding_channel_plan_v4 as v4
from models.second_architecture.nonspeech_probe_v1 import padding_channel_plan_v5 as v5


class V5BudgetTests(unittest.TestCase):
    def test_feature_budget_and_schedule_identity(self):
        self.assertEqual(v5.schedule(), v4.schedule())
        self.assertEqual(v5.schedule_sha256(),
                         "d939618689e989a2dfe249cc6d4f0d3b32a532c184991680b77f11b4af6a0ace")
        self.assertEqual(v5.FEATURE_CASES, 20)
        self.assertEqual(v5.FEATURE_BYTES, 19_200_000)
        self.assertEqual(v5.FEATURE_HEADER_ALLOWANCE_BYTES, 20_480)
        self.assertEqual(v5.worst_case_core_evidence_bytes(), 1_396_433_056)
        self.assertTrue(v5.resource_budget_fits())


if __name__ == "__main__":
    unittest.main()
