"""Fake, no-TTS V2 failure and completion evidence checks."""
import json
import shutil
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import audit_sources_v2 as audit
from . import listening_bundle_v2 as bundle
from . import render_sources_v2 as source


class FakeTTS:
    def __init__(self, _spec):
        pass

    def identity(self):
        return {"synthetic_test_only": True}

    def __call__(self, voice, _text, raw):
        samples = np.full(16000, 1000, dtype="<i2")
        source.tts._write_pcm(raw, samples)
        return ["FAKE_NO_TTS", voice]


class FakeThread:
    def is_alive(self):
        return False


class FakeMonitor:
    def __init__(self, output):
        self.output = output
        self.thread = FakeThread()
        self.samples = [{"utc": datetime.now(timezone.utc).isoformat(),
                         "tree_rss_bytes": 12345678}]

    def start(self):
        pass

    def finish(self):
        with (self.output / "rss_samples.jsonl").open("xb") as sink:
            sink.write((json.dumps(self.samples[0]) + "\n").encode())


def fake_probe(*, run_start=None):
    return {"collected_utc": datetime.now(timezone.utc).isoformat(),
            "outputs": {"power": "FAKE_NO_HOST"}}


class V2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.attempt = self.base / "attempt_001"
        self.bundle = self.base / "review_bundle_001"
        self.go = self.base / "GO.json"
        self.review = self.base / "review.md"
        self.go.write_text("{}")
        self.review.write_text("**Decision: GO**")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(source, "ATTEMPT", self.attempt))
        self.stack.enter_context(patch.object(source, "GO_PATH", self.go))
        self.stack.enter_context(patch.object(source, "REVIEW_PATH", self.review))
        self.stack.enter_context(patch.object(source, "validate_go",
                                              return_value={"sources": {"fake": "source"}}))
        self.stack.enter_context(patch.object(source.tts, "_RealTTS", FakeTTS))
        self.stack.enter_context(patch.object(source, "RssMonitor", FakeMonitor))
        self.stack.enter_context(patch.object(source.host, "probe", side_effect=fake_probe))
        self.stack.enter_context(patch.object(source.host, "archive"))
        self.stack.enter_context(patch.object(source.host, "validate_power"))
        self.stack.enter_context(patch.object(source.host, "validate_log"))
        self.stack.enter_context(patch.object(source.host, "load_archive", return_value="FAKE"))
        self.stack.enter_context(patch.object(bundle, "BUNDLE", self.bundle))
        source.render(self.attempt)

    def test_fake_18_source_reconstruction_and_bundle_copy_audit(self):
        result = audit.audit(self.attempt)
        self.assertEqual(result["cases"], 18)
        self.assertEqual(result["raw_and_converted_and_final_files"], 54)
        made = bundle.build(self.attempt, self.bundle)
        self.assertEqual(len(made["queues"]["A"]), 18)
        self.assertEqual(bundle.audit_bundle(self.attempt, self.bundle)["copied_wavs"], 36)
        with (self.bundle / "reviewer_A/q01.wav").open("ab") as out:
            out.write(b"changed")
        with self.assertRaisesRegex(ValueError, "blinded WAV copy changed"):
            bundle.audit_bundle(self.attempt, self.bundle)

    def test_failure_receipt_rejects_completed_manifest(self):
        (self.attempt / "failure.json").write_text('{"fake":true}')
        with self.assertRaisesRegex(ValueError, "failed attempt"):
            audit.audit(self.attempt)

    def test_missing_or_late_terminal_rejected(self):
        terminal = self.attempt / "terminal.json"
        saved = terminal.read_bytes()
        terminal.unlink()
        with self.assertRaises((FileNotFoundError, ValueError)):
            audit.audit(self.attempt)
        terminal.write_bytes(saved)
        value = json.loads(saved)
        value["elapsed_seconds_after_manifest"] = 901.0
        terminal.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "deadline"):
            audit.audit(self.attempt)

    def test_source_go_failure_prevents_attempt_creation(self):
        other = self.base / "another_attempt"
        with patch.object(source, "ATTEMPT", other), \
             patch.object(source, "validate_go", side_effect=ValueError("no reviewed GO")):
            with self.assertRaisesRegex(ValueError, "no reviewed GO"):
                source.render(other)
        self.assertFalse(other.exists())


if __name__ == "__main__":
    unittest.main()
