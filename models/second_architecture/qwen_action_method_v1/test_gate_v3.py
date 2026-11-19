"""No-weight V3 gate tests, including the independent V2 false promotion."""
from __future__ import annotations

import copy
import unittest

import numpy as np

from . import gate_v3
from .endpoint import EOS_ID, load_spec

TOKENS = {"LIST": 22852, "ASK": 7384, "CANCEL": 88493}


class FakeTokenizer:
    eos_token_id = EOS_ID
    all_special_ids = [151644, EOS_ID]
    pieces = {22852: "LIST", 7384: "ASK", 88493: "CANCEL",
              104: " LIST", 151644: "<|im_start|>", EOS_ID: "<|im_end|>"}

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return "".join(self.pieces[item] for item in ids)


def fixture():
    spec = load_spec()
    fit = [row for row in spec["scripts"] if row["split"] == "fit_text"]
    dev = [row for row in spec["scripts"] if row["split"] == "development"]
    actual = np.array([[.51, .49, 0], [.51, .49, 0], [0, .51, .49],
                       [0, .51, .49], [.49, 0, .51], [.49, 0, .51]])
    wrong = np.array([[.49, .51, 0], [.49, .51, 0], [0, .49, .51],
                      [0, .49, .51], [.51, 0, .49], [.51, 0, .49]])
    typed, audio = [], []
    for idx, row in enumerate(dev):
        common = {"continuation_ids": [TOKENS[row["label"]], EOS_ID],
                  "actual_scores": actual[idx].tolist(),
                  "vjp_scores": wrong[idx].tolist()}
        typed.append({"case_id": f"{row['id']}__typed", **copy.deepcopy(common)})
        audio.append({"case_id": f"{row['id']}__piper_en_US_amy_medium",
                      **copy.deepcopy(common)})
    fit_records = [{"case_id": f"{row['id']}__typed", "scores": [0.0, 0.0, 0.0]}
                   for row in fit]
    return typed, audio, fit_records


class GateV3Tests(unittest.TestCase):
    def setUp(self):
        self.typed, self.audio, self.fit = fixture()
        self.tok = FakeTokenizer()

    def gate(self):
        return gate_v3.development_gate(self.typed, self.audio, self.fit, self.tok)

    def test_exact_v2_false_promotion_is_rejected_by_derived_argmax(self):
        result = self.gate()
        self.assertEqual(result["counts"]["typed_vjp"], 0)
        self.assertEqual(result["counts"]["audio_vjp"], 0)
        self.assertAlmostEqual(result["counts"]["typed_vjp_to_constant_mae"], .04)
        self.assertFalse(result["counts"]["pass"])

    def test_true_score_and_raw_complete_paths_can_pass(self):
        for rows in (self.typed, self.audio):
            for row in rows:
                row["vjp_scores"] = list(row["actual_scores"])
        result = self.gate()
        self.assertTrue(result["counts"]["pass"])
        self.assertEqual(result["counts"]["typed_canonical"], 6)
        self.assertEqual(result["counts"]["audio_canonical"], 6)
        self.assertEqual(result["constant_top"], ["LIST"] * 6)

    def test_asserted_label_and_flag_must_match_raw_endpoint(self):
        self.typed[0]["continuation_ids"] = []
        self.typed[0]["canonical_label_path"] = True
        with self.assertRaisesRegex(ValueError, "canonical_label_path contradicts"):
            self.gate()
        del self.typed[0]["canonical_label_path"]
        self.typed[0]["status"] = "complete"
        with self.assertRaisesRegex(ValueError, "status contradicts"):
            self.gate()
        del self.typed[0]["status"]
        result = self.gate()
        self.assertEqual(result["typed"][0]["endpoint"]["status"], "censored")
        self.assertFalse(result["counts"]["pass"])

    def test_malformed_path_cannot_claim_complete_or_canonical(self):
        self.audio[0]["continuation_ids"] = [EOS_ID, EOS_ID]
        self.audio[0]["label"] = "LIST"
        with self.assertRaisesRegex(ValueError, "label contradicts"):
            self.gate()
        del self.audio[0]["label"]
        result = self.gate()
        self.assertEqual(result["audio"][0]["endpoint"]["status"], "malformed")
        self.assertFalse(result["audio"][0]["endpoint"]["canonical_label_path"])

    def test_score_label_assertions_are_checked_against_matrices(self):
        self.typed[0]["predicted_label"] = "LIST"  # actual VJP argmax is ASK
        with self.assertRaisesRegex(ValueError, "predicted_label contradicts"):
            self.gate()
        del self.typed[0]["predicted_label"]
        self.audio[0]["actual_score_top"] = "CANCEL"
        with self.assertRaisesRegex(ValueError, "actual_score_top contradicts"):
            self.gate()

    def test_frozen_fit_and_development_id_order_is_required(self):
        self.typed[0], self.typed[1] = self.typed[1], self.typed[0]
        with self.assertRaisesRegex(ValueError, "manifest IDs/order"):
            self.gate()
        self.typed[0], self.typed[1] = self.typed[1], self.typed[0]
        self.fit[0], self.fit[1] = self.fit[1], self.fit[0]
        with self.assertRaisesRegex(ValueError, "FIT manifest IDs/order"):
            self.gate()

    def test_noncanonical_but_complete_alias_cannot_fake_canonical_gate(self):
        for rows in (self.typed, self.audio):
            for row in rows:
                row["vjp_scores"] = list(row["actual_scores"])
            rows[0]["continuation_ids"] = [104, EOS_ID]
            rows[1]["continuation_ids"] = [104, EOS_ID]
        result = self.gate()
        self.assertEqual(result["counts"]["typed_canonical"], 4)
        self.assertEqual(result["counts"]["audio_canonical"], 4)
        self.assertEqual(result["typed"][0]["endpoint"]["status"], "complete")
        self.assertFalse(result["counts"]["pass"])


if __name__ == "__main__":
    unittest.main()
