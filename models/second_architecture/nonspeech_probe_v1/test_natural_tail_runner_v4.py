"""No-weight V4 heartbeat exit-race and frozen-method regressions."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from . import natural_tail_parent_v4 as parent
from . import natural_tail_runner_v3 as old
from . import natural_tail_runner_v4 as runner
from . import natural_tail_saved_audit_v4 as audit


class NaturalTailV4NoWeightTests(unittest.TestCase):
    def test_fresh_paths_and_frozen_scientific_functions(self) -> None:
        for model in runner.MODELS:
            self.assertTrue(str(runner.ATTEMPTS[model]).endswith(f"{model}_attempt_v4"))
            self.assertNotEqual(old.ATTEMPTS[model], runner.ATTEMPTS[model])
            self.assertEqual(runner.schedule(model), audit._expected_plan(model))
            features,mask=object(),object()
            self.assertEqual(runner.generation_kwargs(model, features, mask),
                             old.generation_kwargs(model, features, mask))
        def functions(path: Path) -> dict:
            return {n.name:n for n in ast.parse(path.read_text()).body
                    if isinstance(n,ast.FunctionDef)}
        original, current = functions(Path(old.__file__)), functions(Path(runner.__file__))
        for name in ("schedule","verify_wavs","generation_kwargs","infer_one"):
            self.assertEqual(ast.dump(original[name]),ast.dump(current[name]),name)

    def test_complete_worker_output_requires_progress_records_and_sidecars(self) -> None:
        plan=[{"index":index} for index in range(1,161)]
        with tempfile.TemporaryDirectory() as temp:
            attempt=Path(temp)
            (attempt/"sidecars").mkdir()
            with (attempt/"progress.jsonl").open("w") as stream:
                for row in plan:
                    stream.write(json.dumps({"type":"attempt","index":row["index"]})+"\n")
                    stream.write(json.dumps({"type":"complete","index":row["index"]})+"\n")
            records=[]
            kinds=("mel","mask","first_raw_logits",
                   "first_processed_scores","first_decoder_state")
            for row in plan:
                refs={kind:{"file":f'{row["index"]:03d}_{kind}.npy'} for kind in kinds}
                for ref in refs.values():
                    (attempt/"sidecars"/ref["file"]).write_bytes(b"x")
                records.append({"type":"call","plan":row,"sidecars":refs})
            records.append({"type":"complete","calls":160})
            (attempt/"records.jsonl").write_text("".join(json.dumps(x)+"\n" for x in records))
            self.assertTrue(parent.complete_worker_output(attempt,plan))
            (attempt/"records.jsonl").write_text("".join(json.dumps(x)+"\n" for x in records[:-1]))
            self.assertFalse(parent.complete_worker_output(attempt,plan))
            (attempt/"records.jsonl").write_text("".join(json.dumps(x)+"\n" for x in records))
            (attempt/"sidecars"/"160_mel.npy").unlink()
            self.assertFalse(parent.complete_worker_output(attempt,plan))
            (attempt/"sidecars"/"160_mel.npy").write_bytes(b"x")
            (attempt/"progress.jsonl").write_text('{"type":"attempt","index":1}\n')
            self.assertFalse(parent.complete_worker_output(attempt,plan))

    def _fake_broken_pipe(self, *, code: int, complete: bool,
                          still_live: bool = False) -> tuple[dict,bool,bool]:
        class FakeProcess:
            pid=12345
            returncode=None
            polls=0
            def poll(self):
                self.polls += 1
                if self.polls >= 2 and not still_live and self.returncode is None:
                    self.returncode=code
                return self.returncode
            def wait(self,timeout):
                return self.returncode
        process=FakeProcess()
        def write(_fd,data):
            if data == b"H\n":raise BrokenPipeError("closed")
            return len(data)
        with tempfile.TemporaryDirectory() as temp:
            attempt=Path(temp)
            with mock.patch.object(parent.subprocess,"Popen",return_value=process), \
                 mock.patch.object(parent.os,"write",side_effect=write), \
                 mock.patch.object(parent.bounded,"process_tree_rss_bytes",return_value=123), \
                 mock.patch.object(parent,"evidence_bytes",return_value=0), \
                 mock.patch.object(parent,"complete_worker_output",return_value=complete) as output, \
                 mock.patch.object(parent.bounded,"kill_group",
                                   side_effect=lambda child:setattr(child,"returncode",-9)) as kill:
                result=parent.run_bounded([sys.executable,"-c","pass"],attempt,
                    nonce="0123456789abcdef0123456789abcdef",
                    deadline=time.monotonic()+600,seconds=5,plan=[{"index":1}])
                return result, output.called, kill.called

    def test_broken_pipe_clean_exit_with_complete_output_is_normal(self) -> None:
        result, checked, killed = self._fake_broken_pipe(code=0,complete=True)
        self.assertTrue(checked)
        self.assertFalse(killed)
        self.assertEqual(result["worker_stop_reason"],"normal_exit")
        self.assertEqual(result["worker_returncode"],0)

    def test_broken_pipe_partial_or_nonzero_exit_fails(self) -> None:
        for code,complete in ((0,False),(1,True)):
            result,_,_=self._fake_broken_pipe(code=code,complete=complete)
            self.assertEqual(result["worker_stop_reason"],"heartbeat_pipe_lost")

    def test_broken_pipe_still_live_after_sampled_grace_is_killed(self) -> None:
        result,checked,killed=self._fake_broken_pipe(code=0,complete=True,still_live=True)
        self.assertFalse(checked)
        self.assertTrue(killed)
        self.assertEqual(result["worker_stop_reason"],"heartbeat_pipe_lost")
        self.assertGreater(result["worker_rss_samples"],1)

    def test_direct_worker_without_live_pipe_rejects_before_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            attempt=Path(temp)
            with mock.patch.dict(runner.ATTEMPTS,{"whisper_english":attempt}), \
                 mock.patch.object(runner,"validate_go",side_effect=AssertionError(
                     "admission reached")), \
                 mock.patch.object(runner.old,"_snapshot_load",side_effect=AssertionError(
                     "weights reached")):
                with self.assertRaisesRegex(ValueError,"inherited parent pipe"):
                    runner.run("whisper_english",attempt,attempt/"fake_admission",
                               attempt/"fake_go",-1)

    def test_grace_sleep_clamps_clock_crossing(self) -> None:
        self.assertEqual(parent.grace_sleep_seconds(100.0,now=100.001),0.0)
        self.assertAlmostEqual(parent.grace_sleep_seconds(100.0,now=99.95),0.05)


if __name__ == "__main__":
    unittest.main()
