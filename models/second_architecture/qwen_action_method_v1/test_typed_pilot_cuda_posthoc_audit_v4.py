"""No-weight fault tests for the one pilot_003 trailing-zero RSS correction."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from . import typed_pilot_cuda_backend_v4 as backend
from . import typed_pilot_cuda_posthoc_audit_v4 as audit


class PilotPosthocResourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log = Path(self.temp.name) / "samples.jsonl"
        self.samples = [{"sample_index": index, "elapsed_seconds": index / 10,
                         "tree_rss_bytes": 100 if index < 214 else 0,
                         "gpu_used_bytes": 50} for index in range(1, 215)]
        self.result = {"worker_resource_samples": 214,
                       "worker_elapsed_seconds": 22.0,
                       "worker_returncode": 0,
                       "worker_stop_reason": "normal_exit",
                       "worker_peak_tree_rss_bytes": 100,
                       "worker_peak_gpu_used_bytes": 50}
        self._save()

    def _save(self) -> None:
        self.log.write_text("".join(json.dumps(row) + "\n" for row in self.samples))
        self.result["worker_resource_sample_log_sha256"] = backend.sha(self.log)

    def _check(self) -> bool:
        return audit.verify_resource_samples(self.result, self.samples, self.log, 1000)

    def test_exactly_one_final_zero_after_positive_samples_passes(self) -> None:
        self.assertTrue(self._check())

    def test_zero_in_middle_rejected_even_with_valid_hash(self) -> None:
        self.samples[99]["tree_rss_bytes"] = 0
        self._save()
        with self.assertRaisesRegex(ValueError, "sampled resource log invalid"):
            self._check()

    def test_zero_peak_rejected(self) -> None:
        self.result["worker_peak_tree_rss_bytes"] = 0
        with self.assertRaisesRegex(ValueError, "sampled resource log invalid"):
            self._check()

    def test_hash_tamper_rejected(self) -> None:
        self.log.write_text(self.log.read_text().replace('"gpu_used_bytes": 50',
                                                    '"gpu_used_bytes": 51', 1))
        with self.assertRaisesRegex(ValueError, "sampled resource log invalid"):
            self._check()

    def test_nonterminal_zero_and_nonclean_exit_rejected(self) -> None:
        self.samples[-1]["tree_rss_bytes"] = 100
        self._save()
        with self.assertRaisesRegex(ValueError, "sampled resource log invalid"):
            self._check()
        self.samples[-1]["tree_rss_bytes"] = 0
        self.result["worker_returncode"] = 1
        self._save()
        with self.assertRaisesRegex(ValueError, "sampled resource log invalid"):
            self._check()


if __name__ == "__main__":
    unittest.main()
