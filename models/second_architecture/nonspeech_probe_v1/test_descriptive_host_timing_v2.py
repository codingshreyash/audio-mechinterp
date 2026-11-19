"""No-weight regression tests for the V2 post-host worker-exit fence."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import descriptive_parent_v1 as old_parent
from . import descriptive_parent_v2 as parent
from . import descriptive_worker_v1 as worker


class HostTimingSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        eastern = timezone(timedelta(hours=-5))
        self.start = datetime(2026, 11, 19, 8, 11, 0, tzinfo=eastern)
        self.exit = datetime(2026, 11, 19, 8, 11, 11, 300000, tzinfo=eastern)
        self.collected = datetime(2026, 11, 19, 8, 11, 11, 611000, tzinfo=eastern)
        self.after_call_finish = datetime(2026, 11, 19, 8, 11, 12, 87000,
                                          tzinfo=eastern)
        self.log = (
            "2026-11-19 08:00:00 -0500 Wake\tPrior event outside attempt\n"
            "2026-11-19 08:11:11 -0500 : Showing all currently held IOKit power assertions\n"
        )

    def test_exact_v1_second_boundary_race_is_fixed(self) -> None:
        with self.assertRaisesRegex(ValueError, "stale or truncated"):
            old_parent.host_gate._power_log_coverage(
                self.log, self.start, self.after_call_finish, self.collected)
        result = parent.validate_post_host(self.log, self.start, self.exit, self.collected)
        self.assertEqual(result["host_fence_end_utc"], self.exit.isoformat())
        self.assertAlmostEqual(result["host_probe_lag_seconds"], 0.311)

    def test_probe_lag_and_bad_event_remain_hard_failures(self) -> None:
        with self.assertRaisesRegex(ValueError, "probe lag exceeded"):
            parent.validate_post_host(self.log, self.start, self.exit,
                                      self.exit + timedelta(seconds=121))
        event_log = self.log.replace(
            "2026-11-19 08:11:11 -0500 : Showing",
            "2026-11-19 08:11:10 -0500 Sleep\tUnexpected transition\n"
            "2026-11-19 08:11:11 -0500 : Showing")
        with self.assertRaisesRegex(ValueError, "sleep, wake, or thermal"):
            parent.validate_post_host(event_log, self.start, self.exit, self.collected)
        with self.assertRaisesRegex(ValueError, "timestamp order"):
            parent.validate_post_host(self.log, self.start, self.exit,
                                      self.exit - timedelta(milliseconds=1))
        with self.assertRaisesRegex(ValueError, "stale or truncated"):
            parent.validate_post_host(self.log.replace("08:11:11 -0500 : Showing",
                                                       "08:11:09 -0500 : Showing"),
                                      self.start, self.exit, self.collected)

    def test_v2_preflight_hashes_prior_without_reclassifying_it(self) -> None:
        receipt = parent.preflight_receipt()
        self.assertEqual(receipt["prior_failed_attempt"]["status"],
                         "prior_invalid_preserved")
        self.assertEqual(receipt["prior_failed_attempt"]["files_sha256"],
                         parent.PRIOR_SHA256)
        self.assertEqual(receipt["attempt_path"], str(parent.V2_ATTEMPT.resolve()))
        self.assertEqual(receipt["max_post_probe_lag_seconds"], 120)
        self.assertEqual(receipt["v1_source_identity"], worker.source_identity())

    def test_without_new_independent_go_no_v2_attempt_is_admitted(self) -> None:
        self.assertFalse(parent.V2_ATTEMPT.exists())
        with self.assertRaisesRegex(ValueError, "review missing"):
            parent.require_v2_go(Path("/nonexistent/independent_v2_go.json"))
        self.assertFalse(parent.V2_ATTEMPT.exists())


if __name__ == "__main__":
    unittest.main()
