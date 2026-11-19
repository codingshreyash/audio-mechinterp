"""No-weight tests for two distinct remaining-model admissions."""
from __future__ import annotations

import copy
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from . import descriptive_parent_v2 as timing
from . import descriptive_remaining_parent_v1 as parent
from . import descriptive_worker_v1 as worker


class RemainingSourceTests(unittest.TestCase):
    def test_exact_names_distinct_paths_and_source_only_receipts(self) -> None:
        self.assertEqual(set(parent.MODEL_PATHS), {"whisper_english", "wav2vec2"})
        english = parent.preflight_receipt("whisper_english")
        wav = parent.preflight_receipt("wav2vec2")
        self.assertNotEqual(english["attempt_path"], wav["attempt_path"])
        self.assertNotEqual(english["review_path"], wav["review_path"])
        self.assertEqual(english["planned_cases"], wav["planned_cases"])
        self.assertEqual(english["planned_cases"], 20)
        self.assertEqual(english["planned_silence_repeats"], 2)
        self.assertEqual(english["case_ids"], wav["case_ids"])
        self.assertEqual(english["case_ids"],
                         [case["case_id"] for case in worker.trusted_cases()])
        self.assertNotEqual(english["model_assets"]["revision"],
                            wav["model_assets"]["revision"])
        self.assertEqual(english["status"], "source_only_no_model_load")
        self.assertEqual(wav["status"], "source_only_no_model_load")
        self.assertNotIn(english["attempt_path"],
                         (str(timing.V1_ATTEMPT), str(timing.V2_ATTEMPT)))
        self.assertNotIn(wav["attempt_path"],
                         (str(timing.V1_ATTEMPT), str(timing.V2_ATTEMPT)))

    def test_review_for_one_model_cannot_authorize_the_other(self) -> None:
        english = parent.preflight_receipt("whisper_english")
        wav = parent.preflight_receipt("wav2vec2")
        review = {"schema": "nonspeech_descriptive_remaining_model_review_v1",
                  "status": "GO", "model": "whisper_english",
                  "attempt_path": english["attempt_path"],
                  "preflight_sha256": "a" * 64,
                  "source_identity": english["source_identity"]}
        parent.validate_review_fields("whisper_english", review, english, "a" * 64)
        with self.assertRaisesRegex(ValueError, "does not bind this arm"):
            parent.validate_review_fields("wav2vec2", review, wav, "b" * 64)
        changed = copy.deepcopy(review)
        changed["attempt_path"] = wav["attempt_path"]
        with self.assertRaisesRegex(ValueError, "does not bind this arm"):
            parent.validate_review_fields("whisper_english", changed, english, "a" * 64)

    def test_admission_failure_occurs_before_host_or_attempt(self) -> None:
        for model in parent.MODEL_PATHS:
            with self.subTest(model=model):
                with mock.patch.object(parent, "require_model_go",
                                       side_effect=ValueError("fake review absent")):
                    with mock.patch.object(parent.old_parent, "host_probe") as probe:
                        with self.assertRaisesRegex(ValueError, "fake review absent"):
                            parent.run_once(model)
                        probe.assert_not_called()

    def test_remaining_parent_uses_corrected_exit_time_host_fence(self) -> None:
        eastern = timezone(timedelta(hours=-5))
        start = datetime(2026, 11, 19, 8, 11, 0, tzinfo=eastern)
        exited = datetime(2026, 11, 19, 8, 11, 11, 300000, tzinfo=eastern)
        collected = datetime(2026, 11, 19, 8, 11, 11, 611000, tzinfo=eastern)
        log = ("2026-11-19 08:00:00 -0500 Wake\tPrior event\n"
               "2026-11-19 08:11:11 -0500 : Showing all currently held IOKit power assertions\n")
        result = timing.validate_post_host(log, start, exited, collected)
        self.assertEqual(result["host_fence_end_utc"], exited.isoformat())
        self.assertLess(result["host_probe_lag_seconds"], 1.0)


if __name__ == "__main__":
    unittest.main()
