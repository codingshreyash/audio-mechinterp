"""Source-only checks for the 12-case VoiceChat TTS output plan."""

from pathlib import Path
import unittest

from render_synthetic_v1 import VOICES, checked_inputs, output_paths


class RenderPathTests(unittest.TestCase):
    def test_all_render_paths_are_distinct(self) -> None:
        cases, _identity = checked_inputs()
        output = Path("/tmp/voicechat-render-path-check")
        all_paths = []
        for voice in VOICES:
            for case in cases:
                source, converted, final = output_paths(output, case["id"], voice)
                self.assertEqual(len({source, converted, final}), 3)
                self.assertEqual(final.suffix, ".wav")
                self.assertNotEqual(source, final)
                all_paths.extend((source, converted, final))
        self.assertEqual(len(all_paths), 36)
        self.assertEqual(len(set(all_paths)), 36)


if __name__ == "__main__":
    unittest.main()
