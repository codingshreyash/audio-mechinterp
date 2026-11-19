#!/usr/bin/env python3
"""No-model checks for the single-file descriptive viewer."""

import base64
import hashlib
import re
import unittest

import build_descriptive_v1 as builder


class ViewerBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = builder.build_data()

    def test_exact_census_and_separate_language_prefix(self):
        d = self.data
        self.assertEqual(len(d["cases"]), 20)
        self.assertEqual(d["summary"]["multilingual"], {"nonempty": 20, "different_from_silence": 4})
        self.assertEqual(d["summary"]["english"], {"nonempty": 20, "different_from_silence": 0})
        self.assertEqual(d["summary"]["ctc"], {"nonempty": 3, "different_from_silence": 3})
        self.assertEqual(d["cases"]["d02_beep_train_high"]["multilingual"]["transcript"], " Fuck")
        self.assertEqual(d["cases"]["d02_beep_train_high"]["multilingual"]["language_prefix"], "<|en|>")
        self.assertEqual(d["cases"]["d02_silence"]["multilingual"]["transcript"], " you")
        self.assertEqual(d["cases"]["d08_beep_train_high"]["ctc"]["transcript"], "AAAAAAAAAAAAAAAA")

    def test_embedded_wavs_match_manifest_hashes(self):
        for case_id, row in self.data["cases"].items():
            audio = row["asset"]["audio"]
            self.assertTrue(audio.startswith("data:audio/wav;base64,"), case_id)
            wav = base64.b64decode(audio.split(",", 1)[1], validate=True)
            self.assertEqual(hashlib.sha256(wav).hexdigest(), row["asset"]["wav_sha256"])

    def test_checked_in_html_is_deterministic_and_offline(self):
        built = builder.render(self.data)
        self.assertEqual((builder.HERE / "descriptive_v1.html").read_text(), built)
        self.assertNotIn("__EMBEDDED_EVIDENCE_JSON__", built)
        self.assertNotRegex(built, r"(?i)<(?:script|link|iframe)[^>]+(?:src|href)=[\"']https?://")
        self.assertEqual(len(re.findall(r"data:audio/wav;base64,", built)), 20)


if __name__ == "__main__":
    unittest.main()
