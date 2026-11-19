"""No-weight source checks; production bytes get a separate saved-result audit."""
import unittest

from . import listening_bundle, render_sources


class SourceTests(unittest.TestCase):
    def test_frozen_18_case_population_and_voice_split(self):
        rows = render_sources.expected_sources()
        self.assertEqual(len(rows), 18)
        self.assertEqual(len({r["id"] for r in rows}), 18)
        self.assertEqual({(r["split"], r["voice"]):
                          sum(x["split"] == r["split"] and x["voice"] == r["voice"]
                              for x in rows) for r in rows}, {
            ("development", "piper_en_US_amy_medium"): 6,
            ("locked", "macos_samantha_en_US"): 6,
            ("locked", "kokoro_af_bella"): 6})
        self.assertEqual(render_sources.source_identity()["frames"], 256000)

    def test_blind_html_never_discloses_source_words_or_labels(self):
        html = listening_bundle._html("A", ["q01.wav", "q02.wav"], "sample-binding")
        self.assertIn('src="${file}"', html)
        self.assertIn("q01.wav", html)
        self.assertNotIn("Design Review", html)
        self.assertNotIn("November nineteenth", html)
        self.assertNotIn("CANCEL", html)
        self.assertIn("bundle_binding", html)


if __name__ == "__main__":
    unittest.main()
