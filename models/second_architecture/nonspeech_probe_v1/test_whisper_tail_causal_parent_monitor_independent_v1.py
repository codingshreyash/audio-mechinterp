"""No-weight integration tests for concurrent host and worker monitoring."""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from . import whisper_tail_causal_parent_v1 as parent


class ParentMonitorIndependentTests(unittest.TestCase):
    def _run(self, probe):
        with tempfile.TemporaryDirectory(prefix="causal-host-monitor-") as raw:
            attempt = Path(raw)
            started = time.monotonic()
            samples = [{"monotonic_seconds": started}]
            with (attempt / "host_samples.jsonl").open("x") as host_out, \
                    patch.object(parent.host, "quick_probe", side_effect=probe), \
                    patch.object(parent.bounded, "process_tree_rss_bytes", return_value=1024**2):
                result = parent.run_bounded(
                    [sys.executable, "-c", "import time; time.sleep(5.2)"],
                    attempt, "whisper_english", nonce="0" * 32,
                    start=started, deadline=started + 300,
                    samples=samples, host_out=host_out)
            return result, samples

    def test_slow_probe_does_not_starve_heartbeat_or_rss(self):
        def slow_probe(*, initial=False):
            self.assertFalse(initial)
            time.sleep(2.5)  # Exceeds the worker's two-second heartbeat gap.
            return {"monotonic_seconds": time.monotonic(),
                    "battery_percent": 99, "lid_open": True, "thermal_ok": True}

        result, samples = self._run(slow_probe)
        self.assertGreaterEqual(result["parent_heartbeat_count"], 10)
        self.assertGreaterEqual(result["worker_rss_samples"], 10)
        self.assertGreaterEqual(len(samples), 2)

    def test_probe_failure_kills_worker(self):
        def failed_probe(*, initial=False):
            raise ValueError("unreadable host")

        result, _samples = self._run(failed_probe)
        self.assertEqual(result["worker_stop_reason"], "host_probe_failed")
        self.assertNotEqual(result["worker_returncode"], 0)


if __name__ == "__main__":
    unittest.main()
