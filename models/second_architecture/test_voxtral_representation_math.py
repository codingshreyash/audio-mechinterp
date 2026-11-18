"""Independent receptive-field and partition checks; no model or MPS load."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from voxtral_representation_spread import (
    CONNECTOR_ROWS, MEL_FRAMES, MEL_HALF_WINDOW, MEL_HOP,
    nominal_word_connector_rows, partition_indices,
)


def brute_force_connector_rows(start: int, end: int) -> list[int]:
    hits = []
    for connector_row in range(CONNECTOR_ROWS):
        overlap = False
        for encoder_row in range(4 * connector_row, 4 * connector_row + 4):
            for conv2_input in range(2 * encoder_row - 1, 2 * encoder_row + 2):
                for mel_frame in range(conv2_input - 1, conv2_input + 2):
                    if not 0 <= mel_frame < MEL_FRAMES:
                        continue
                    center = mel_frame * MEL_HOP
                    if center - MEL_HALF_WINDOW < end and center + MEL_HALF_WINDOW > start:
                        overlap = True
        if overlap:
            hits.append(connector_row)
    return hits


class ShapeMathTests(unittest.TestCase):
    def test_flattened_encoder_row_count(self):
        self.assertEqual(MEL_FRAMES // 2, 1500)
        self.assertEqual((MEL_FRAMES // 2) // 4, CONNECTOR_ROWS)

    def test_word_window_matches_independent_conv_enumeration(self):
        for interval in ((18082, 26627), (0, 160), (479000, 479999)):
            with self.subTest(interval=interval):
                self.assertEqual(nominal_word_connector_rows(*interval), brute_force_connector_rows(*interval))
        self.assertEqual(nominal_word_connector_rows(18082, 26627), list(range(13, 22)))

    def test_partition_covers_every_row_once_including_padding(self):
        groups = partition_indices(CONNECTOR_ROWS, list(range(13, 22)), 39)
        combined = [index for group in groups.values() for index in group]
        self.assertEqual(sorted(combined), list(range(CONNECTOR_ROWS)))
        self.assertEqual(len(combined), len(set(combined)))
        self.assertEqual(len(groups["nominal_padding"]), 336)


if __name__ == "__main__":
    unittest.main()
