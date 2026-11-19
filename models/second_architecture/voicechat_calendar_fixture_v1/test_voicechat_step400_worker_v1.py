"""No-model tests for the pinned public-sample diagnostic child."""
from __future__ import annotations

import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from models.second_architecture.voicechat_calendar_fixture_v1 import (
    voicechat_step400_worker_v1 as worker,
)


class WorkerTests(unittest.TestCase):
    def test_exact_paths_rejects_consumed_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            nemo, ckpt, out = root / "nemo", root / "ckpt", root / "out"
            for path in (nemo, ckpt, out):
                path.mkdir()
            for name in worker.EXPECTED_FILES:
                (out / name).write_bytes(b"")
            with patch.object(worker.Path, "cwd", return_value=out):
                worker._exact_paths(nemo, ckpt, out)
                (out / "worker_completion.json").write_bytes(b"old")
                with self.assertRaisesRegex(ValueError, "consumed"):
                    worker._exact_paths(nemo, ckpt, out)

    def test_preparation_mirrors_official_default_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            nemo = Path(temp).resolve()
            template = nemo / "examples/speechlm2/function_calling/template.jinja"
            calls: list[tuple] = []
            model, signal, length, tokens, token_length = (object() for _ in range(5))
            default_tools = [{"function": {"name": "generate_random_number"}}]
            official = types.SimpleNamespace(
                DEFAULT_TEMPLATE=template, DEFAULT_SYSTEM_MESSAGE="default message",
                DEFAULT_TOOLS=default_tools,
                render_fc_system_prompt=lambda t, m, x:
                    calls.append(("render", t, m, x)) or "rendered prompt",
                build_model=lambda ckpt, device:
                    calls.append(("build", ckpt, device)) or model,
                load_wav_16k_mono=lambda path, device:
                    calls.append(("wav", path, device)) or (object(), signal, length),
                encode_system_prompt=lambda got_model, prompt, device:
                    calls.append(("tokens", got_model, prompt, device)) or
                    (tokens, token_length),
            )
            ckpt = nemo / "checkpoint"
            got = worker.prepare_official_inputs(official, ckpt, nemo)
            self.assertEqual(got[:5], (model, signal, length, tokens, token_length))
            self.assertEqual(calls, [
                ("render", template, "default message", default_tools),
                ("build", str(ckpt), "cuda"),
                ("wav", str(nemo / worker.sample.SAMPLE_WAV), "cuda"),
                ("tokens", model, "rendered prompt", "cuda"),
            ])
            self.assertEqual(got[5]["sample_id"], "sample_fc")
            self.assertEqual(got[5]["prompt_sha256"],
                             worker.digest(b"rendered prompt"))

    def test_wrong_default_tool_fails_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            nemo = Path(temp).resolve()
            official = types.SimpleNamespace(
                DEFAULT_TEMPLATE=nemo / "examples/speechlm2/function_calling/template.jinja",
                DEFAULT_SYSTEM_MESSAGE="x", DEFAULT_TOOLS=[],
                build_model=lambda *_a, **_k: self.fail("loaded model"))
            with self.assertRaisesRegex(ValueError, "default prompt or tool"):
                worker.prepare_official_inputs(official, nemo / "checkpoint", nemo)

    def test_worker_writes_one_bounded_completion_and_no_pass2(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            nemo, ckpt, out = root / "nemo", root / "ckpt", root / "out"
            for path in (nemo, ckpt, out):
                path.mkdir()
            for name in worker.EXPECTED_FILES:
                (out / name).write_bytes(b"")
            read_fd, write_fd = os.pipe()
            official = object()
            first_pass = object()
            helper = types.SimpleNamespace(run_offline_inference=first_pass)
            sent = [object() for _ in range(5)]
            identity = {"nemo_revision": "pinned"}

            def fake_pass1(**kwargs):
                self.assertIs(kwargs["offline_inference"], first_pass)
                self.assertEqual(tuple(kwargs[k] for k in
                                 ("model", "input_signal", "input_signal_lens",
                                  "prompt_tokens", "prompt_token_lens")), tuple(sent))
                kwargs["ledger"].write({"t": 1})
                for t in (50, 100, 200, 300, 400):
                    kwargs["ledger"].write({"t": t})
                return {"schema": "voicechat_step400_core_completion_v1",
                        "status": "diagnostic_early_stop_complete",
                        "ledger": kwargs["ledger"].close()}

            try:
                with patch.object(worker.Path, "cwd", return_value=out), \
                     patch.object(worker, "verify_pinned_public_sample",
                                  return_value=identity), \
                     patch.object(worker, "import_pinned_official",
                                  return_value=(official, helper)), \
                     patch.object(worker, "prepare_official_inputs",
                                  return_value=(*sent, {"sample_id": "sample_fc"})), \
                     patch.object(worker.probe, "run_pinned_pass1_only",
                                  side_effect=fake_pass1):
                    result = worker.run_worker(nemo_dir=nemo, checkpoint_dir=ckpt,
                                               output_dir=out, heartbeat_fd=write_fd)
                saved = json.loads((out / "worker_completion.json").read_text())
                self.assertEqual(saved, result)
                self.assertTrue(saved["no_second_pass"])
                self.assertEqual(saved["core"]["ledger"]["checkpoints"], 6)
                self.assertEqual(set(p.name for p in out.iterdir()),
                                 worker.EXPECTED_FILES |
                                 {"checkpoints.jsonl", "worker_completion.json"})
            finally:
                os.close(read_fd)
                os.close(write_fd)


if __name__ == "__main__":
    unittest.main()
