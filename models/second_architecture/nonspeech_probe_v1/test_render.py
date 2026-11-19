"""No-model tests for the frozen non-speech waveform source."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from . import render as source


class SourceTests(unittest.TestCase):
    def test_exact_case_matrix_and_deterministic_pcm(self) -> None:
        spec = source.read_spec()
        first = source.cases(spec)
        second = source.cases(spec)
        self.assertEqual(len(first), 20)
        self.assertEqual([name for name, _ in first], [name for name, _ in second])
        for (name, pcm), (_, repeat) in zip(first, second):
            self.assertTrue(np.array_equal(pcm, repeat), name)
            self.assertEqual(pcm.dtype, np.dtype("int16"))
            self.assertIn(len(pcm), (32000, 128000))
            self.assertLessEqual(float(np.max(np.abs(pcm.astype(float) / 32768))), 0.8)
            if name.endswith("silence"):
                self.assertTrue(np.all(pcm == 0))
            if name.endswith("dither"):
                self.assertTrue(np.all(np.abs(pcm) <= 1))

    def test_wav_metadata_and_manifest_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "new_attempt"
            result = source.render(path)
            self.assertEqual(result["status"], "complete_source_only_no_model_inference")
            self.assertEqual(len(result["rows"]), 20)
            self.assertEqual(len(list(path.iterdir())), 22)
            manifest = json.loads((path / "manifest.json").read_text())
            self.assertEqual(manifest, result)
            for row in result["rows"]:
                wav_path = path / row["wav"]
                raw = wav_path.read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), row["wav_sha256"])
                with wave.open(str(wav_path), "rb") as sound:
                    self.assertEqual((sound.getnchannels(), sound.getframerate(),
                                      sound.getsampwidth()), (1, 16000, 2))
                    pcm = sound.readframes(sound.getnframes())
                    self.assertEqual(sound.getnframes(), row["frames"])
                self.assertEqual(hashlib.sha256(pcm).hexdigest(), row["pcm_sha256"])
            with self.assertRaises(ValueError):
                source.render(path)

    def test_band_noise_is_dominated_by_declared_band(self) -> None:
        spec = source.read_spec()
        for name, pcm in source.cases(spec):
            if "band_noise" not in name:
                continue
            spectrum = np.abs(np.fft.rfft(pcm.astype(np.float64))) ** 2
            freq = np.fft.rfftfreq(len(pcm), 1 / 16000)
            in_band = (freq >= 300) & (freq <= 3400)
            self.assertGreater(float(spectrum[in_band].sum() / spectrum.sum()), 0.999,
                               name)

    def test_spec_byte_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tampered = Path(temporary) / "spec.json"
            tampered.write_bytes(source.SPEC_PATH.read_bytes() + b" ")
            with mock.patch.object(source, "SPEC_PATH", tampered):
                with self.assertRaisesRegex(ValueError, "spec bytes changed"):
                    source.read_spec()


if __name__ == "__main__":
    unittest.main()
