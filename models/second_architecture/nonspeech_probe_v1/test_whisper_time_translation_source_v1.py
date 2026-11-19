"""No-model byte tests for the prospective time-translation stimuli."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from . import whisper_time_translation_source_v1 as source


class SourceTests(unittest.TestCase):
    def test_eight_equal_duration_layouts_reconstruct_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "source"
            rendered = source.render(root)
            self.assertEqual(source.verify(root), rendered)
            self.assertEqual(len(rendered["rows"]), 8)
            for offset in source.OFFSETS:
                rows = [r for r in rendered["rows"] if r["offset_samples"] == offset]
                self.assertEqual(len(rows), 4)
                self.assertEqual({r["frames"] for r in rows}, {source.FRAMES + offset})
                self.assertEqual(len({r["pcm_sha256"] for r in rows}), 4)

    def test_existing_path_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "source"
            source.render(root)
            with self.assertRaisesRegex(ValueError, "already consumed"):
                source.render(root)

    def test_modified_pcm_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "source"
            rendered = source.render(root)
            path = root / rendered["rows"][0]["wav"]
            raw = bytearray(path.read_bytes())
            raw[-1] ^= 1
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "PCM differs"):
                source.verify(root)


if __name__ == "__main__":
    unittest.main()
