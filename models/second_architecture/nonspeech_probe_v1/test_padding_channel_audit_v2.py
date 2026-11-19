"""No-weight independent feature binding and saved-cap rejection tests."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from models.second_architecture.nonspeech_probe_v1 import padding_channel_audit_v2 as audit
from models.second_architecture.nonspeech_probe_v1 import padding_channel_worker_v2 as worker


class AuditV2Tests(unittest.TestCase):
    def test_exact_feature_sidecar_recomputed_and_all_call_binding(self):
        case = "d02_beep_train_low"
        expected = np.zeros((1, 80, 3000), np.float32)
        mask = [1] * 200 + [0] * 2800
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            ref = worker.save_array(output, f"input/{case}_features.npy",
                                    expected, worker.Budget(output))
            item = {"input_feature_ref": ref,
                    "input_feature_sha256_before": ref["tensor_sha256"],
                    "input_feature_sha256_after": ref["tensor_sha256"],
                    "input_geometry": {"attention_mask": mask}}
            known, seen = {}, set()
            audit.verify_input_binding(output, item, case, expected, mask, known, seen)
            audit.verify_input_binding(output, item, case, expected, mask, known, seen)
            self.assertEqual(len(known), 1)
            self.assertEqual(seen, {ref["path"]})
            with self.assertRaisesRegex(ValueError, "input feature or mask"):
                audit.verify_input_binding(output, {**item,
                    "input_feature_sha256_after": "0" * 64},
                    case, expected, mask, known, seen)
            with self.assertRaisesRegex(ValueError, "input feature or mask"):
                audit.verify_input_binding(output, {**item,
                    "input_geometry": {"attention_mask": mask[:-1] + [1]}},
                    case, expected, mask, known, seen)
            changed_expected = expected.copy()
            changed_expected[0, 0, 0] = 1.0
            with self.assertRaisesRegex(ValueError, "input feature or mask"):
                audit.verify_input_binding(output, item, case,
                                           changed_expected, mask, known, seen)

    def test_feature_bytes_tamper_rejected(self):
        case = "d02_beep_train_low"
        expected = np.zeros((1, 80, 3000), np.float32)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            ref = worker.save_array(output, f"input/{case}_features.npy",
                                    expected, worker.Budget(output))
            path = output / ref["path"]
            path.write_bytes(path.read_bytes()[:-1] + b"X")
            item = {"input_feature_ref": ref,
                    "input_feature_sha256_before": ref["tensor_sha256"],
                    "input_feature_sha256_after": ref["tensor_sha256"],
                    "input_geometry": {"attention_mask": [1] * 200 + [0] * 2800}}
            with self.assertRaisesRegex(ValueError, "byte hash"):
                audit.verify_input_binding(output, item, case, expected,
                                           item["input_geometry"]["attention_mask"],
                                           {}, set())


if __name__ == "__main__":
    unittest.main()
