"""No-weight counterexamples for the V1 independent-review HOLD."""
from __future__ import annotations

import unittest

from . import endpoint_v2
from .endpoint import EOS_ID


class FakeTokenizer:
    eos_token_id = EOS_ID
    all_special_ids = [EOS_ID, 151644]
    pieces = {101: "LIST", 102: "ASK", 103: "CANCEL", 104: " CANCEL",
              105: " and LIST", 22852: "LIST", 7384: "ASK", 88493: "CANCEL",
              151644: "<|im_start|>", EOS_ID: "<|im_end|>"}

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return "".join(self.pieces[x] for x in ids)


class EndpointV2Tests(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()

    def test_valid_complete_labels_and_exact_token_parity(self):
        for token, expected in ((101, "LIST"), (102, "ASK"), (103, "CANCEL"),
                                (104, "CANCEL")):
            result = endpoint_v2.score_complete_answer([token, EOS_ID], self.tok)
            self.assertEqual((result["status"], result["label"], result["reason"]),
                             ("complete", expected, None))
            self.assertFalse(result["canonical_label_path"])
        a = endpoint_v2.score_complete_answer([103, EOS_ID], self.tok)
        b = endpoint_v2.score_complete_answer([104, EOS_ID], self.tok)
        canonical = endpoint_v2.score_complete_answer([88493, EOS_ID], self.tok)
        self.assertTrue(canonical["canonical_label_path"])
        self.assertTrue(endpoint_v2.exact_complete_parity(a, dict(a)))
        self.assertFalse(endpoint_v2.exact_complete_parity(a, b))

    def test_empty_and_cap_reached_stay_in_censored_denominator(self):
        empty = endpoint_v2.score_complete_answer([], self.tok)
        self.assertEqual((empty["status"], empty["reason"], empty["continuation_ids"]),
                         ("censored", "empty_continuation", []))
        cap = endpoint_v2.score_complete_answer([103], self.tok)
        self.assertEqual((cap["status"], cap["reason"]),
                         ("censored", "missing_terminal_eos"))
        self.assertFalse(endpoint_v2.exact_complete_parity(cap, cap))

    def test_early_or_repeated_eos_is_malformed(self):
        for ids in ([101, EOS_ID, 105], [101, EOS_ID, EOS_ID],
                    [EOS_ID, 101], [EOS_ID, EOS_ID]):
            result = endpoint_v2.score_complete_answer(ids, self.tok)
            self.assertEqual((result["status"], result["reason"]),
                             ("malformed", "early_or_repeated_eos"))
            self.assertEqual(result["continuation_ids"], list(ids))

    def test_unexpected_special_id_and_extra_words_are_malformed(self):
        for ids in ([151644, 103, EOS_ID], [103, 151644, EOS_ID],
                    [151644, 103], [103, 105, EOS_ID]):
            result = endpoint_v2.score_complete_answer(ids, self.tok)
            self.assertEqual(result["status"], "malformed")
        self.assertEqual(endpoint_v2.score_complete_answer(
            [151644, 103, EOS_ID], self.tok)["reason"], "unexpected_special_id")

    def test_invalid_ids_and_tokenizer_metadata_fail_closed(self):
        for ids in ([True, EOS_ID], [-1, EOS_ID], [103] * 13):
            with self.assertRaises(ValueError):
                endpoint_v2.score_complete_answer(ids, self.tok)
        bad = FakeTokenizer()
        bad.all_special_ids = []
        with self.assertRaisesRegex(ValueError, "omits EOS"):
            endpoint_v2.score_complete_answer([], bad)


if __name__ == "__main__":
    unittest.main()
