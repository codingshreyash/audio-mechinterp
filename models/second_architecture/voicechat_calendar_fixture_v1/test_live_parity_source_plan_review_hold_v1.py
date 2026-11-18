"""Independent source-review reproductions; no model or human QA is run."""

from __future__ import annotations

import unittest

from . import test_live_parity_source_plan_v1 as base_test


class SourcePlanReviewHoldReproductions(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = base_test.SourcePlanTests("test_success_emits_plan_and_never_models")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_rejects_selected_wav_changed_after_preload(self) -> None:
        def changed_preload(**_kwargs):
            self.fixture.wav.write_bytes(b"changed-after-initial-QA-check")
            return self.fixture._fake_preload()

        with self.assertRaises(ValueError):
            self.fixture._build(preload=changed_preload)

    def test_rejects_qa_manifest_changed_after_preload(self) -> None:
        def changed_preload(**_kwargs):
            self.fixture.manifest.write_bytes(b'{"changed_after_preflight":true}\n')
            return self.fixture._fake_preload()

        with self.assertRaises(ValueError):
            self.fixture._build(preload=changed_preload)

    def test_rejects_sample_wav_changed_after_preload(self) -> None:
        from . import live_parity_source_plan_v1 as plan

        def changed_preload(**_kwargs):
            (self.fixture.nemo / plan.SAMPLE_WAV).write_bytes(b"changed-sample")
            return self.fixture._fake_preload()

        with self.assertRaises(ValueError):
            self.fixture._build(preload=changed_preload)


if __name__ == "__main__":
    unittest.main()
