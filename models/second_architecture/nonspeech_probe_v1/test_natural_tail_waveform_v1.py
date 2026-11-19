"""No-weight invariants for the frozen natural-tail source constructor."""

import unittest
import tempfile
import wave
from pathlib import Path

import numpy as np

from .natural_tail_waveform_v1 import (
    CONDITIONS, NOISE_PCM_AMPLITUDE, TAIL_SAMPLES, construct_pcm, noise_tail,
    render_new, trusted_sources,
)


class NaturalTailSourceTests(unittest.TestCase):
    def test_fixed_denominator_and_exact_prefix(self):
        original = np.array([0, 1, -200, 300, 0], dtype="<i2")
        self.assertEqual(len(CONDITIONS), 7)
        for condition in CONDITIONS:
            with self.subTest(condition=condition):
                built = construct_pcm(original, condition)
                np.testing.assert_array_equal(built[: len(original)], original)
                if condition == "base":
                    self.assertEqual(len(built), len(original))
                else:
                    samples = next(n for n in TAIL_SAMPLES if str(n) in condition)
                    self.assertEqual(len(built), len(original) + samples)

    def test_zero_and_seeded_noise_controls(self):
        original = np.array([7, -3], dtype="<i2")
        for samples in TAIL_SAMPLES:
            z = construct_pcm(original, f"z{samples}")[len(original):]
            self.assertTrue(np.all(z == 0))
            a = noise_tail(samples, 0)
            b = noise_tail(samples, 1)
            np.testing.assert_array_equal(a, noise_tail(samples, 0))
            self.assertTrue(np.all(np.abs(a) == NOISE_PCM_AMPLITUDE))
            self.assertTrue(np.all(np.abs(b) == NOISE_PCM_AMPLITUDE))
            self.assertFalse(np.array_equal(a, b))
            np.testing.assert_array_equal(
                construct_pcm(original, f"n{samples}_s0")[len(original):], a)

    def test_reject_unfrozen_conditions_and_bad_pcm(self):
        original = np.array([1, 2], dtype="<i2")
        for condition in ("z16", "n8000_s2", "n16000_s0", "not-a-condition"):
            with self.subTest(condition=condition), self.assertRaises(ValueError):
                construct_pcm(original, condition)
        with self.assertRaises(ValueError):
            construct_pcm(original.astype(np.float32), "base")

    def test_source_render_has_140_rows_and_preserves_every_prefix(self):
        sources = trusted_sources()
        self.assertEqual(len(sources), 20)
        originals = {case_id: pcm for case_id, pcm, _ in sources}
        with tempfile.TemporaryDirectory() as parent:
            manifest = render_new(Path(parent) / "new")
            self.assertEqual(len(manifest["rows"]), 140)
            self.assertEqual(len({(r["case_id"], r["condition"])
                                  for r in manifest["rows"]}), 140)
            for row in manifest["rows"]:
                with wave.open(str(Path(parent) / "new" / row["wav"]), "rb") as stream:
                    self.assertEqual((stream.getnchannels(), stream.getsampwidth(),
                                      stream.getframerate()), (1, 2, 16000))
                    raw = stream.readframes(stream.getnframes())
                pcm = np.frombuffer(raw, dtype="<i2")
                np.testing.assert_array_equal(
                    pcm[: len(originals[row["case_id"]])], originals[row["case_id"]])


if __name__ == "__main__":
    unittest.main()
