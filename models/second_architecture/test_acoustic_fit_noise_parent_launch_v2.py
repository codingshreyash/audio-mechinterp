"""No-model V2 parent admission and timeout checks."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acoustic_fit_noise_parent_launch_v2 as parent


class AcousticParentV2Tests(unittest.TestCase):
    def test_preserved_v1_failure_is_byte_bound(self):
        parent.verify_prior_failure()
        with mock.patch.object(parent.pilot, "sha_file", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "failed-attempt evidence changed"):
                parent.verify_prior_failure()

    def test_only_frozen_wav2vec2_attempt_002_is_admitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrong = Path(tmp) / "wav2vec2_attempt_002"
            with self.assertRaisesRegex(ValueError, "only the failed Wav2Vec2"):
                parent.execute("whisper", Path("receipt"), Path("worker_review"),
                               Path("parent_review"), parent.ATTEMPT)
            with self.assertRaisesRegex(ValueError, "frozen attempt_002 path"):
                parent.execute("wav2vec2", Path("receipt"), Path("worker_review"),
                               Path("parent_review"), wrong)
            self.assertFalse(wrong.exists())

    def test_v2_capture_and_review_schema_bindings(self):
        self.assertEqual(parent.WORKER.name, "acoustic_fit_noise_pilot_v2.py")
        self.assertEqual(parent.PROTOCOL.name, "ACOUSTIC_FIT_NOISE_PILOT_PROSPECTIVE_V2.md")
        with tempfile.TemporaryDirectory() as tmp:
            review = Path(tmp) / "review.json"
            review.write_text('{"schema":"acoustic_fit_noise_parent_source_go_v1","status":"GO"}')
            worker_review = Path(tmp) / "worker.json"
            worker_review.write_text("{}")
            with self.assertRaisesRegex(ValueError, "does not bind"):
                parent.require_launcher_go(review, {}, "pre", worker_review)

    def test_fake_subprocess_is_killed_on_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc, timed_out, elapsed = parent.run_bounded(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                root / "stdout", root / "stderr", timeout_seconds=0.05)
            self.assertTrue(timed_out)
            self.assertNotEqual(rc, 0)
            self.assertLess(elapsed, 2)


if __name__ == "__main__":
    unittest.main()
