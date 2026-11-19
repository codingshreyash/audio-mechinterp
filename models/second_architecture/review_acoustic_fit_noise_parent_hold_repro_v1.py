"""Read-only fake reproductions for the FIT acoustic parent launch review.

No candidate model, real host probe, or GPU is used.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import acoustic_fit_noise_parent_launch_v1 as parent


def forged_complete_matrix() -> bool:
    expected = {f"case_{i}": {arm: f"pcm_{i}_{arm}" for arm in parent.pilot.ARMS}
                for i in range(4)}
    rows = [{"type": "start", "model": "whisper"}]
    rows += [{"type": "case", "model": "whisper", "case": case,
              "arm": arm, "pcm_sha256": digest}
             for case, arms in expected.items() for arm, digest in arms.items()]
    rows += [{"type": "complete", "planned_cases": 20}]
    # The case rows have no complete output, state hashes, or clean-repeat parity.
    return parent.valid_worker_rows(rows, "whisper", expected)


def late_success() -> tuple[int | None, bool, float]:
    fake_process = mock.Mock(pid=999, returncode=0)
    fake_process.wait.return_value = 0
    with TemporaryDirectory() as temp:
        with (mock.patch.object(parent.subprocess, "Popen", return_value=fake_process),
              mock.patch.object(parent.time, "monotonic", side_effect=[0.0, 0.1, 1801.0])):
            return parent.run_bounded(["fake"], Path(temp) / "out",
                                      Path(temp) / "err", timeout_seconds=1800)


if __name__ == "__main__":
    accepted = forged_complete_matrix()
    code, timed_out, elapsed = late_success()
    print({"forged_matrix_accepted": accepted, "late_returncode": code,
           "late_timed_out": timed_out, "late_elapsed_seconds": elapsed})
    assert accepted is True
    assert code == 0 and timed_out is False and elapsed > parent.MAX_WORKER_SECONDS
