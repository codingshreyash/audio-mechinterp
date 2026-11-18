"""CPU-only checks for prospective development mask construction."""

import json
import unittest

import voxtral_dev_transfer_masks as masks


class DevelopmentMaskTests(unittest.TestCase):
    def test_exact_index_population_and_rebuild(self):
        masks.verify_sources()
        saved = json.loads(masks.OUTPUT.read_text())
        rebuilt = masks.make_manifest(json.loads(masks.INDEX.read_text()))
        self.assertEqual(saved, rebuilt)
        self.assertEqual(len(saved["records"]), 48)
        self.assertEqual(sum(len(r["words"]) for r in saved["records"]), 96)
        self.assertEqual({r["category"] for r in saved["records"]}, set(masks.CATEGORIES))

    def test_direct_overlap_and_disjoint_site_rows(self):
        saved = json.loads(masks.OUTPUT.read_text())
        for record in saved["records"]:
            start = record["word_slot_start_sample"]
            end = record["word_slot_end_sample_exclusive"]
            self.assertEqual(record["word_mel_frames_direct_overlap"], [
                i for i in range(3000)
                if i * masks.fit.MEL_HOP - masks.fit.MEL_HALF_WINDOW < end
                and i * masks.fit.MEL_HOP + masks.fit.MEL_HALF_WINDOW > start])
            for site in masks.SITES:
                data = record["site_maps"][site]
                groups = data["bins"]
                self.assertTrue(all(groups.values()))
                rows = [j for group in groups.values() for j in group]
                self.assertEqual(len(rows), len(set(rows)))
                whole = sorted(j for k in range(4) for j in groups[f"word_{k}"])
                self.assertEqual(data["whole_word_rows"], whole)
                self.assertEqual(len(data["padding_control_rows"]), len(whole))
                self.assertFalse(set(data["padding_control_rows"]) & set(rows))

    def test_pcm_mutation_is_rejected(self):
        index = json.loads(masks.INDEX.read_text())
        row = next(r for r in index["rows"] if r["split"] == "development")
        original = row["clips"][row["words"][0]]
        changed = {**original, "wav_sha256": "0" * 64}
        with self.assertRaises(ValueError):
            masks.read_wav(changed)


if __name__ == "__main__":
    unittest.main()
