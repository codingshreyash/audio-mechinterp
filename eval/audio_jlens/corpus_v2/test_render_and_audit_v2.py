"""Synthetic CPU tests only; these do not render the proposed v2 population."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

import audit_wavs_v2 as auditor
import render_v2 as renderer


class CorpusV2PreparationTests(unittest.TestCase):
    def test_fixed_primary_population_and_inactive_reserves(self):
        spec, report, ordered = renderer.load_population()
        self.assertEqual((len(ordered), len(ordered) * 2), (240, 480))
        self.assertEqual(len(renderer.fragment_texts(spec, ordered)), 132)
        self.assertEqual(len(auditor.expected_population(spec, report)), 240)
        primary = {word for rows in spec["pairs_by_category"].values()
                   for pair in rows for word in pair}
        reserve = {word for rows in spec["ordered_reserve_pairs_by_category"].values()
                   for pair in rows for word in pair}
        self.assertEqual(len(primary), 120)
        self.assertEqual(len(reserve), 24)
        self.assertTrue(all(word in primary for row in ordered
                            for word in row["pair"]["words"]))
        self.assertFalse(any(word in reserve for row in ordered
                             for word in row["pair"]["words"]))
        self.assertEqual({split: len([row for row in ordered if row["split"] == split])
                          for split in renderer.SPLITS},
                         {"fit_pool": 144, "development": 48, "locked_test": 48})

    def test_render_blocked_without_independent_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(renderer, "FREEZE", Path(temp) / "missing_freeze.json"):
                with self.assertRaisesRegex(ValueError, "freeze receipt absent"):
                    renderer.freeze_receipt()

    def test_exact_fragment_reconstruction_detects_outside_and_word_slot_mutations(self):
        with tempfile.TemporaryDirectory(dir=auditor.ROOT, prefix="v2_audit_test_") as temp:
            output = Path(temp)
            (output / "clips").mkdir()
            pair = {"pair_id": "toy_01", "category": "animal",
                    "words": ["otter", "badger"]}
            template = {"id": "toy_template", "prefix": "I say", "suffix": "Which word?"}
            planned = {"split": "fit_pool", "pair": pair,
                       "template": template, "engine": "piper"}
            pieces = {"prefix_toy_template": 1600, "word_toy_01_otter": 800,
                      "word_toy_01_badger": 1000, "suffix_toy_template": 1200}
            fragments = {("piper", key): {"fragment_id": key,
                                          "voice": "en_US-amy-medium",
                                          "raw": {"file_sha256": "raw_" + key},
                                          "processed": {"file_sha256": "pcm_" + key,
                                                        "frames": frames}}
                         for key, frames in pieces.items()}
            source_pcm = {"prefix_toy_template": np.full(1600, 120, dtype=np.int16),
                          "word_toy_01_otter": np.full(800, 500, dtype=np.int16),
                          "word_toy_01_badger": np.full(1000, 750, dtype=np.int16),
                          "suffix_toy_template": np.full(1200, 240, dtype=np.int16)}
            start, end = 1600 + 1920, 1600 + 1920 + 1000
            expected_len = end + 1920 + 1200
            clips, outputs = {}, {}
            for word, word_frames, value in (("otter", 800, 500), ("badger", 1000, 750)):
                pcm = np.zeros(expected_len, dtype=np.int16)
                pcm[:1600] = 120
                pcm[start:start + word_frames] = value
                pcm[end + 1920:] = 240
                path = output / "clips" / f"piper_toy_01_toy_template_{word}.wav"
                sf.write(path, pcm, 16000, subtype="PCM_16", format="WAV")
                clips[word] = pcm
                outputs[word] = {
                    "path": str(path.relative_to(auditor.ROOT)),
                    "file_sha256": auditor.sha(path),
                    "pcm_int16_sha256": hashlib.sha256(pcm.astype("<i2").tobytes()).hexdigest(),
                    "frames": expected_len, "duration_seconds": expected_len / 16000,
                    "word_fragment_frames": word_frames,
                    "peak_int16": int(np.max(np.abs(pcm.astype(np.int32)))),
                    "spoken_script": f"I say {word}. Which word?",
                }
            row = {"split": "fit_pool", "category": "animal", "fit_role": None,
                   "engine": "piper", "voice": "en_US-amy-medium",
                   "pair_id": "toy_01", "phrasing_id": "toy_template",
                   "words": pair["words"], "sample_rate": 16000, "subtype": "PCM_16",
                   "outside_slot_pcm_identical": True, "inside_slot_pcm_differs": True,
                   "word_slot_start_sample": start, "word_slot_end_sample_exclusive": end,
                   "word_slot_start_seconds": start / 16000,
                   "word_slot_end_seconds": end / 16000,
                   "outputs": outputs,
                   "fragment_sha256": {key: {"raw": item["raw"]["file_sha256"],
                                             "processed": item["processed"]["file_sha256"]}
                                       for (_engine, key), item in fragments.items()}}
            with mock.patch.object(auditor, "processed_pcm",
                                   side_effect=lambda fragment, _output:
                                   source_pcm[fragment["fragment_id"]]):
                self.assertEqual(len(auditor.audit_pair(row, planned, fragments, output)[0]), 2)
                clips["otter"][4] += 1
                path = output / "clips/piper_toy_01_toy_template_otter.wav"
                sf.write(path, clips["otter"], 16000, subtype="PCM_16", format="WAV")
                # A refreshed manifest hash cannot hide an outside-slot edit.
                outputs["otter"]["file_sha256"] = auditor.sha(path)
                outputs["otter"]["pcm_int16_sha256"] = hashlib.sha256(
                    clips["otter"].astype("<i2").tobytes()).hexdigest()
                with self.assertRaisesRegex(ValueError, "exact processed-fragment assembly"):
                    auditor.audit_pair(row, planned, fragments, output)
                clips["otter"][4] -= 1
                clips["otter"][start + 100] += 1
                sf.write(path, clips["otter"], 16000, subtype="PCM_16", format="WAV")
                # A word-slot mutation preserves paired outside-slot equality.
                # Refreshed hashes still must not make it pass the auditor.
                outputs["otter"]["file_sha256"] = auditor.sha(path)
                outputs["otter"]["pcm_int16_sha256"] = hashlib.sha256(
                    clips["otter"].astype("<i2").tobytes()).hexdigest()
                with self.assertRaisesRegex(ValueError, "exact processed-fragment assembly"):
                    auditor.audit_pair(row, planned, fragments, output)


if __name__ == "__main__":
    unittest.main()
