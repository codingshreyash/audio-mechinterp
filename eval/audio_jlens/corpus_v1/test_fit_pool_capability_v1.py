"""CPU-only controls for full-answer classification and frozen FIT enumeration."""

import unittest

from fit_pool_capability_v1 import (
    N_CASES, classify_answer, load_fit_cases, summarize,
)


class FitPoolCapabilityTests(unittest.TestCase):
    def test_answer_surfaces_eos_and_alias_are_distinct(self):
        aliases = [{"surface": "Bear", "token_id": 11}]
        exact = classify_answer(" Bear ", [11, 99], 99, "bear", aliases)
        self.assertTrue(exact["strict_word_plus_eos"])
        self.assertTrue(exact["first_token_in_expected_aliases"])
        punctuated = classify_answer("Bear.", [12, 99], 99, "bear", aliases)
        self.assertFalse(punctuated["strict_word_plus_eos"])
        self.assertTrue(punctuated["semantic_word_plus_eos"])
        self.assertFalse(punctuated["first_token_in_expected_aliases"])
        truncated = classify_answer("Bear", [11], 99, "bear", aliases)
        self.assertFalse(truncated["semantic_word_plus_eos"])
        extra = classify_answer("Bear Wolf", [11, 22, 99], 99, "bear", aliases)
        self.assertFalse(extra["semantic_one_word"])
        empty = classify_answer("", [], 99, "bear", aliases)
        self.assertFalse(empty["stopped_on_eos"])
        self.assertFalse(empty["first_token_in_expected_aliases"])

    def test_frozen_fit_pool_is_exact_and_disjoint(self):
        spec, manifest, cases = load_fit_cases()
        self.assertEqual(len(cases), N_CASES)
        self.assertEqual(sum(case["map_fit"] for case in cases), 48)
        self.assertEqual({case["template"] for case in cases},
                         {"fit_chosen", "fit_remember"})
        self.assertEqual({case["engine"] for case in cases}, {"piper", "kokoro"})
        self.assertEqual({case["category"] for case in cases},
                         {"animal", "color", "object", "nature"})
        self.assertEqual(len({case["audio_sha256"] for case in cases}), N_CASES)
        self.assertEqual(manifest["spec_sha256"],
                         "4c4cb6beafdd27509a073872f71dea004be686a6d8fafccbae410b8b11d7f265")
        self.assertEqual(spec["expected"]["map_fit_recipient_examples"], 48)

    def test_counts_keep_all_rows(self):
        rows = []
        for index in range(N_CASES):
            rows.append({
                "category": "animal", "engine": "piper", "template": "fit_chosen",
                "map_fit": False,
                "classification": {
                    "stopped_on_eos": index % 2 == 0,
                    "strict_word_plus_eos": index % 4 == 0,
                    "semantic_word_plus_eos": index % 2 == 0,
                    "first_token_in_expected_aliases": index % 3 == 0,
                },
            })
        result = summarize(rows)
        self.assertEqual(result["overall"]["planned"], N_CASES)
        self.assertEqual(result["overall"]["completed"], N_CASES)
        self.assertEqual(result["overall"]["ended_on_eos"], 96)
        self.assertEqual(result["overall"]["strict_word_plus_eos"], 48)


if __name__ == "__main__":
    unittest.main()
