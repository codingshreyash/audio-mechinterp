"""Source-only checks for prospective VoiceChat ASR QA. No ASR model is loaded."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import asr_qa_v1 as qa


class ASRQASourceTests(unittest.TestCase):
    def test_exact_twelve_audio_inputs_and_separate_references(self):
        inputs, references = qa.validate_inputs()
        self.assertEqual(len(inputs), 12)
        self.assertEqual(len({x["opaque_id"] for x in inputs}), 12)
        self.assertEqual([(r["voice_id"], r["case_id"]) for r in references],
                         [(v, c) for v in qa.VOICES for c in qa.CASES])
        self.assertTrue(all(set(x) == {"opaque_id", "path", "wav_sha256",
                                       "pcm_int16_sha256", "frames"} for x in inputs))
        self.assertTrue(all("script" in x for x in references))
        self.assertTrue(all("script" not in x for x in inputs))

    def test_modified_final_hash_fails_closed(self):
        original = json.loads(qa.MANIFEST.read_text())
        original["rows"][0]["final_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as name:
            other = Path(name) / "manifest.json"
            other.write_text(json.dumps(original))
            with patch.object(qa, "MANIFEST", other), patch.dict(
                    qa.EXPECTED, {"manifest": qa.sha(other)}):
                with self.assertRaisesRegex(ValueError, "final WAV missing or hash changed"):
                    qa.validate_inputs()

    def test_missing_or_wrong_freeze_blocks_run(self):
        inputs, _ = qa.validate_inputs()
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            with patch.object(qa, "FREEZE", directory / "missing.json"):
                with self.assertRaisesRegex(ValueError, "freeze receipt absent"):
                    qa.run("whisper", directory / "output", inputs, {}, {}, None)
                self.assertFalse((directory / "output").exists())
            bad = directory / "bad.json"
            bad.write_text(json.dumps({"status": "independently_reviewed_voicechat_asr_qa_frozen",
                                       "runner_sha256": "0" * 64}))
            with patch.object(qa, "FREEZE", bad):
                with self.assertRaisesRegex(ValueError, "does not bind"):
                    qa.run("whisper", directory / "output", inputs, {}, {}, None)
                self.assertFalse((directory / "output").exists())

    def test_raw_log_full_denominator_and_reference_leak(self):
        inputs, _ = qa.validate_inputs()
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            rows = [{"ordinal": i, "opaque_id": item["opaque_id"],
                     "wav_sha256": item["wav_sha256"],
                     "pcm_int16_sha256": item["pcm_int16_sha256"],
                     "raw_transcript": "Test", "generated_token_ids": [1, 2],
                     "decoder_cap": 128, "ended_with_eos": True}
                    for i, item in enumerate(inputs, 1)]
            raw = output / "raw.jsonl"
            raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(qa.verify_raw(output, "whisper", inputs), qa.sha(raw))
            raw.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
            with self.assertRaisesRegex(ValueError, "denominator incomplete"):
                qa.verify_raw(output, "whisper", inputs)
            rows[0]["reference"] = "unallowed target"
            raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "field set changed"):
                qa.verify_raw(output, "whisper", inputs)

    def test_normalization_and_word_errors(self):
        wanted = qa.normalized_words("Confirm cancel Design Review.")
        heard = qa.normalized_words("CONFIRM, cancel design review")
        self.assertEqual(wanted, heard)
        self.assertEqual(qa.word_errors(wanted, heard), 0)
        self.assertEqual(qa.word_errors(wanted, qa.normalized_words("cancel design review")), 1)
        self.assertEqual(qa.word_errors(wanted, []), 4)


if __name__ == "__main__":
    unittest.main()
