"""CPU-only contract tests; no processor, model, ASR, or answer output."""

from __future__ import annotations

import copy
import unittest

import voxtral_v2_jlens_masks as masks


class VoxtralV2NominalMaskTests(unittest.TestCase):
    def test_full_frozen_population_and_waveform_binding(self):
        manifest, validation, spec = masks.read_inputs()
        planned = masks.planned_rows(validation, spec)
        self.assertEqual(len(planned), 240)
        made = masks.make_manifest(manifest, validation, spec)
        records = made["records"]
        self.assertEqual(len(records), 240)
        self.assertEqual(sum(len(r["clips"]) for r in records), 480)
        self.assertEqual(sum(r["fit_role"] == "basis" for r in records), 64)
        self.assertEqual(sum(r["fit_role"] == "map" for r in records), 80)
        self.assertEqual([sum(r["split"] == split for r in records)
                          for split in masks.SPLITS], [144, 48, 48])
        for row in records:
            self.assertEqual(len(row["word_mel_frames_direct_overlap"]) > 0, True)
            self.assertEqual(row["blank_fill_mel_frame"], 2999)
            for site in masks.SITES:
                support = row["site_maps"][site]
                self.assertEqual(len(support["bins"]), 6)
                self.assertEqual(len(support["whole_word_rows"]),
                                 len(support["padding_control_rows"]))

    def test_paired_pcm_invariant_catches_off_word_and_identical_word(self):
        left = bytearray(40)
        right = bytearray(40)
        left[10:14] = b"abcd"
        right[10:14] = b"wxyz"
        masks.require_pair_pcm(bytes(left), bytes(right), 5, 7)
        right[0] = 1
        with self.assertRaisesRegex(ValueError, "outside word"):
            masks.require_pair_pcm(bytes(left), bytes(right), 5, 7)
        with self.assertRaisesRegex(ValueError, "not inside word"):
            masks.require_pair_pcm(bytes(left), bytes(left), 5, 7)

    def test_role_and_manifest_claim_tampering_fail_closed(self):
        manifest, validation, spec = masks.read_inputs()
        wrong_roles = copy.deepcopy(validation)
        wrong_roles["assigned_pairs"]["fit_pool"][0]["fit_role"] = "map"
        with self.assertRaisesRegex(ValueError, "FIT-role balance"):
            masks.planned_rows(wrong_roles, spec)
        wrong_claim = copy.deepcopy(manifest)
        wrong_claim["pairs"][0]["outside_slot_pcm_identical"] = False
        with self.assertRaisesRegex(ValueError, "paired claims"):
            masks.make_manifest(wrong_claim, validation, spec)

    def test_old_shape_is_a_fail_closed_assumption(self):
        with self.assertRaisesRegex(ValueError, "3000-mel-frame"):
            masks.nominal_geometry(3000, 5000, 500000)
        with self.assertRaisesRegex(ValueError, "pre/post interval"):
            masks.nominal_geometry(100, 2100, 3000)
        intervals, mel, sites = masks.nominal_geometry(40000, 56000, 100000)
        self.assertEqual(set(intervals), {"pre", "post", "word_0", "word_1",
                                           "word_2", "word_3"})
        self.assertTrue(mel)
        self.assertEqual(set(sites), set(masks.SITES))


if __name__ == "__main__":
    unittest.main()
