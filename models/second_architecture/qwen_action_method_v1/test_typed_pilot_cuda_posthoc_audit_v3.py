"""No-weight fault checks for the posthoc wall/monotonic deadline proof."""
from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path

from . import typed_pilot_cuda_posthoc_audit_v3 as audit


class PosthocTimingTests(unittest.TestCase):
    def fixtures(self):
        start={"start_utc":"2026-11-19T13:18:05.265395+00:00"}
        pre={"collected_utc":"2026-11-19T13:18:05.300000+00:00"}
        result={"end_utc":"2026-11-19T13:18:33.310915+00:00",
                "worker_exit_utc":"2026-11-19T13:18:33.243732+00:00",
                "elapsed_seconds_before_result":28.045517345000007}
        post={"collected_utc":"2026-11-19T13:18:33.290000+00:00"}
        terminal={"terminal_utc":"2026-11-19T13:18:33.312483+00:00",
                  "elapsed_seconds_after_result":28.047071378000055}
        return start,result,terminal,pre,post

    def test_small_file_mtime_skew_is_not_a_deadline_signal(self):
        start,result,terminal,pre,post=self.fixtures()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"terminal.json"
            path.write_text("immutable test fixture")
            timestamp=audit._time(terminal["terminal_utc"]).timestamp()
            os.utime(path,(timestamp-0.0043,timestamp-0.0043))
            self.assertLess(path.stat().st_mtime,timestamp)
            evidence=audit.verify_parent_timing("load",start,result,terminal,pre,post)
            self.assertLess(evidence["max_clock_disagreement_seconds"],0.001)

    def test_rejects_wall_clock_shift(self):
        start,result,terminal,pre,post=self.fixtures()
        result=copy.deepcopy(result)
        terminal=copy.deepcopy(terminal)
        result["end_utc"]="2026-11-19T13:18:34.310915+00:00"
        terminal["terminal_utc"]="2026-11-19T13:18:34.312483+00:00"
        with self.assertRaisesRegex(ValueError,"wall/monotonic"):
            audit.verify_parent_timing("load",start,result,terminal,pre,post)

    def test_rejects_monotonic_overrun_and_reversed_host_order(self):
        start,result,terminal,pre,post=self.fixtures()
        late=copy.deepcopy(terminal)
        late["elapsed_seconds_after_result"]=180.001
        with self.assertRaisesRegex(ValueError,"wall/monotonic"):
            audit.verify_parent_timing("load",start,result,late,pre,post)
        reversed_post=copy.deepcopy(post)
        reversed_post["collected_utc"]="2026-11-19T13:18:34+00:00"
        with self.assertRaisesRegex(ValueError,"wall/monotonic"):
            audit.verify_parent_timing("load",start,result,terminal,pre,reversed_post)


if __name__ == "__main__":
    unittest.main()
