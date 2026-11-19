"""No-model boundary checks for the unsigned H200 freeze builder."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.second_architecture.voicechat_calendar_fixture_v1 import build_public_sample_freeze_v5_h200 as builder


GO = Path(__file__).resolve().parent / "VOICECHAT_PUBLIC_SAMPLE_SOURCE_REVIEW_GO_V5_H200_30G.json"


class H200FreezeBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name).resolve()
        self.args = argparse.Namespace(
            nemo_dir=root / "Speech", checkpoint_dir=root / "checkpoint",
            nano_hf_home=root / "nano", output_dir=root / "sample-output-005",
            source_review=GO, freeze=root / "freeze" / "freeze.json")
        source_hashes = json.loads(GO.read_text())["source_sha256"]
        self.expected = {
            "source_sha256": source_hashes, "gpu_policy": "h200-141gb",
            "native_ffmpeg_runtime": {
                "source": {"manifest_sha256": "a" * 64, "package_archives_count": 129},
                "recursive_lib_tree": {"entry_count": 1330}},
        }

    def test_writes_unsigned_h200_freeze_without_consuming_output(self) -> None:
        with patch.object(builder.sample, "expected_freeze_fields", return_value=self.expected), \
             patch.object(builder.sample, "check_h200_sample_runtime", return_value={
                 "gpu_policy": "h200-141gb", "disk_free_gib": 41.0}):
            receipt = builder.build(self.args)
        freeze = json.loads(self.args.freeze.read_text())
        self.assertEqual(freeze["caps"], {"wall_seconds": 1800, "vram_mib": 130000,
                                           "total_evidence_bytes": 1 << 30})
        self.assertEqual(freeze["expected"]["gpu_policy"], "h200-141gb")
        self.assertEqual(freeze["source_review"]["sha256"], builder.SOURCE_GO_SHA256)
        self.assertEqual(receipt["gpu_policy"], "h200-141gb")
        self.assertFalse(self.args.output_dir.exists())

    def test_rejects_consumed_output_before_host_checks(self) -> None:
        self.args.output_dir.mkdir()
        with patch.object(builder.sample, "expected_freeze_fields") as expected:
            with self.assertRaisesRegex(ValueError, "already consumed"):
                builder.build(self.args)
        expected.assert_not_called()
        self.assertFalse(self.args.freeze.exists())

    def test_rejects_stale_source_review_before_host_checks(self) -> None:
        stale = Path(self.tmp.name).resolve() / "stale-review.json"
        stale.write_bytes(GO.read_bytes() + b" ")
        self.args.source_review = stale
        with patch.object(builder.sample, "expected_freeze_fields") as expected:
            with self.assertRaisesRegex(ValueError, "not the independently reviewed"):
                builder.build(self.args)
        expected.assert_not_called()
        self.assertFalse(self.args.freeze.exists())


if __name__ == "__main__":
    unittest.main()
