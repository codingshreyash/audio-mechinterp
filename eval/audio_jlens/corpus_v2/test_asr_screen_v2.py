"""CPU preflight and synthetic ASR-log tests; no model inference."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import asr_screen_v2 as screen


def fake_inputs() -> list[dict]:
    return [{"opaque_id": f"{i:020x}", "wav_sha256": f"wav{i}",
             "pcm_int16_sha256": f"pcm{i}"} for i in (1, 2)]


def fake_row(item: dict, ordinal: int, model: str) -> dict:
    row = {"ordinal": ordinal, "opaque_id": item["opaque_id"],
           "wav_sha256": item["wav_sha256"],
           "pcm_int16_sha256": item["pcm_int16_sha256"],
           "raw_transcript": "I HEARD A WORD"}
    if model == "whisper":
        row.update(generated_token_ids=[7, 8, 9], decoder_cap=screen.MAX_NEW_TOKENS,
                   ended_with_eos=True)
    else:
        row.update(frame_argmax_token_ids=[1, 2, 2, 3], ctc_frame_count=4)
    return row


class ASRScreenContractTests(unittest.TestCase):
    def test_pinned_480_input_and_two_cached_model_preflight(self):
        pins, checked = screen.check_sources()
        reference_cases = json.loads((screen.HERE / "QA_CASES_PROSPECTIVE_V1.json").read_text())["cases"]
        asr_cases = json.loads(screen.INPUTS.read_text())["cases"]
        projection = ("opaque_id", "audio_path", "wav_sha256",
                      "pcm_int16_sha256", "frames")
        self.assertEqual(asr_cases,
                         sorted(({key: case[key] for key in projection}
                                 for case in reference_cases), key=lambda case: case["opaque_id"]))
        self.assertTrue(all(set(case) == set(projection) for case in asr_cases))
        self.assertEqual(len(checked["inputs"]), 480)
        self.assertEqual(len({c["opaque_id"] for c in checked["inputs"]}), 480)
        self.assertEqual(len({str(c["path"]) for c in checked["inputs"]}), 480)
        self.assertFalse(any("target_word" in c or "reference_script" in c
                             for c in checked["inputs"]))
        self.assertEqual(set(screen.check_model_pins(pins)), {"whisper", "wav2vec2"})

    def test_missing_freeze_rejects_any_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(screen, "FREEZE", Path(temp) / "missing.json"):
                with self.assertRaisesRegex(ValueError, "freeze receipt absent"):
                    screen.check_freeze()
                output = Path(temp) / "asr_output"
                with self.assertRaisesRegex(ValueError, "freeze receipt absent"):
                    screen.run("whisper", output, {}, [], {})
                self.assertFalse(output.exists())

    def test_missing_amendment_pin_rejects_source_preflight_and_run(self):
        with tempfile.TemporaryDirectory() as temp:
            freeze = Path(temp) / "freeze.json"
            receipt = {"status": "independently_reviewed_asr_screen_frozen",
                       "script_sha256": screen.sha(Path(screen.__file__))}
            receipt.update({f"{key}_sha256": screen.EXPECTED[key] for key in
                            ("manifest", "binding", "cases", "inputs", "pins", "protocol")})
            freeze.write_text(json.dumps(receipt))
            with mock.patch.object(screen, "FREEZE", freeze):
                with self.assertRaisesRegex(ValueError, "does not bind exact sources"):
                    screen.check_freeze()
                output = Path(temp) / "asr_output"
                with self.assertRaisesRegex(ValueError, "does not bind exact sources"):
                    screen.run("whisper", output, {}, [], {})
                self.assertFalse(output.exists())
                receipt["amendment_sha256"] = screen.EXPECTED["amendment"]
                freeze.write_text(json.dumps(receipt))
                self.assertEqual(screen.check_freeze(), receipt)
            with mock.patch.object(screen, "AMENDMENT", Path(temp) / "absent.md"):
                with self.assertRaisesRegex(ValueError, "frozen amendment SHA changed"):
                    screen.check_sources()
                with self.assertRaisesRegex(ValueError, "amendment SHA changed"):
                    screen.check_freeze()

    def test_raw_rows_are_complete_ordered_and_reference_free(self):
        expected = fake_inputs()
        for model in ("whisper", "wav2vec2"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "raw.jsonl"
                rows = [fake_row(item, ordinal, model)
                        for ordinal, item in enumerate(expected, 1)]
                def save():
                    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                save()
                self.assertEqual(screen.verify_raw_log(path, expected, model), screen.sha(path))
                rows[1]["opaque_id"] = rows[0]["opaque_id"]
                save()
                with self.assertRaisesRegex(ValueError, "identity/transcript"):
                    screen.verify_raw_log(path, expected, model)
                rows[1]["opaque_id"] = expected[1]["opaque_id"]
                rows[0]["target_word"] = "secret"
                save()
                with self.assertRaisesRegex(ValueError, "reference leaked"):
                    screen.verify_raw_log(path, expected, model)
                rows[0].pop("target_word")
                path.write_text(json.dumps(rows[0]) + "\n")
                with self.assertRaisesRegex(ValueError, "omitted a planned WAV"):
                    screen.verify_raw_log(path, expected, model)


if __name__ == "__main__":
    unittest.main()
