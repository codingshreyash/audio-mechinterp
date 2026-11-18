"""Metadata-only tests. No WAV, ASR, listener, or Qwen result is opened."""

import json
import unittest
from collections import Counter
from unittest import mock

import make_qa_queue_v1 as qa


class QaQueueTests(unittest.TestCase):
    def test_full_population_and_blind_nonadjacent_order(self):
        private, public = qa.build()
        case_by_id = {case["opaque_id"]: case for case in private["cases"]}
        order = public["opaque_ids_in_order"]
        self.assertEqual((len(case_by_id), len(order), len(set(order))), (480, 480, 480))
        self.assertEqual(Counter(case["split"] for case in private["cases"]), qa.SPLITS)
        self.assertEqual(set(order), set(case_by_id))
        self.assertTrue(all(case_by_id[a]["pair_id"] != case_by_id[b]["pair_id"]
                            for a, b in zip(order, order[1:])))
        self.assertEqual(qa.build(), (private, public))
        exposed = json.dumps(public)
        for forbidden in ("audio_path", "target_word", "paired_alternative",
                          "reference_script", "pair_id", "phrasing_id", "wav_sha256"):
            self.assertNotIn(forbidden, exposed)

    def test_changed_technical_binding_pin_stops_plan(self):
        with mock.patch.dict(qa.PINS, {"binding_sha256": "0" * 64}):
            with self.assertRaisesRegex(ValueError, "pinned metadata changed: binding"):
                qa.build()


if __name__ == "__main__":
    unittest.main()
