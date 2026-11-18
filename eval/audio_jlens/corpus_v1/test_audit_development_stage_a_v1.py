"""CPU toy tests for the independent auditor; no development artifacts read."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_development_stage_a_v1 import (AuditError, EOS, recompute_bootstrap,
                                           recompute_site, recompute_task,
                                           require, verify_generation,
                                           verify_descriptive_groups)
from development_transport_metrics_v1 import summarize_primary


class ToyTokenizer:
    all_special_ids = [EOS, 10]
    def decode(self, ids, skip_special_tokens=True):
        table = {0: "cat", 1: "dog", EOS: "" if skip_special_tokens else "<eos>",
                 10: "" if skip_special_tokens else "<hidden>"}
        return "".join(table[i] for i in ids)


def toy_records():
    result = []
    for pair in range(12):
        for direction in range(8):
            alpha_rows = [{"alpha": alpha, "measured_v2_delta_score": 1.0,
                           "predicted_v2_delta_score": {
                               "map": .9, "zero": 0.0, "global": .5,
                               "six_bin": .6}}
                          for alpha in (.005, .01, .025)]
            result.append({
                "case_id": f"pair_{pair}/{direction}",
                "pair_id": f"pair_{pair}",
                "category": f"category_{pair//3}",
                "engine": "piper" if direction < 4 else "kokoro",
                "template": "template_0" if direction % 4 < 2 else "template_1",
                "clean": {"semantic_word_plus_eos": True,
                          "prefix_class": "expected",
                          "prefix_class_complete_answer_match": True},
                "sites": {site: {"absolute_anchor_improved": True,
                                 "alphas": alpha_rows}
                          for site in ("encoder31", "connector")},
            })
    return result


class AuditToyTests(unittest.TestCase):
    def test_recompute_gates_and_pair_bootstrap(self):
        records = toy_records()
        task = recompute_task(records)
        self.assertTrue(task["task_fidelity_gate_passed"])
        stats = recompute_site(records, "encoder31", .01)
        self.assertEqual(stats["determinate"], 288)
        self.assertTrue(stats["pass"])
        self.assertAlmostEqual(stats["mae"]["map"], .1)
        pair_order = [f"pair_{i}" for i in range(12)]
        boot = recompute_bootstrap(stats["pair_component"], pair_order)
        self.assertEqual(boot["sign"]["interval_95"], [1.0, 1.0])
        self.assertEqual(boot["clean"]["interval_95"], [1.0, 1.0])
        self.assertEqual(boot["map_zero_ratio"]["n_valid_replicates"], 10000)
        saved = summarize_primary(records, {"encoder31": .01, "connector": .01},
                                  bootstrap=False)
        verify_descriptive_groups(records, "encoder31", .01,
                                  saved["sites"]["encoder31"]["descriptive_groups"])

    def test_token_structure_cannot_hide_special(self):
        policy = {"words": {"cat": {"first_token_ids": [0]},
                            "dog": {"first_token_ids": [1]}}}
        generation = {
            "token_ids": [0, 10, EOS], "decoded_raw": "cat",
            "decoded_with_special_tokens": "cat<hidden><eos>",
            "classification": {"decoded_trimmed": "cat",
                               "valid_token_structure": False,
                               "nonterminal_special_token_ids": [10],
                               "early_eos": False,
                               "stopped_on_eos": True,
                               "strict_exact_word": True,
                               "semantic_one_word": True,
                               "first_token_id": 0,
                               "first_token_in_expected_aliases": True,
                               "expected_alias_token_ids": [0],
                               "strict_word_plus_eos": False,
                               "semantic_word_plus_eos": False},
            "semantic_word_plus_eos": False,
            "prefix_class": "expected",
            "prefix_class_complete_answer_match": False,
            "audio_source_hook_calls": 1,
        }
        verify_generation(generation, "cat", "dog", policy, ToyTokenizer(),
                          [{"token_id": 0}],
                          clean=False)
        generation["classification"]["semantic_word_plus_eos"] = True
        with self.assertRaises(AuditError):
            verify_generation(generation, "cat", "dog", policy, ToyTokenizer(),
                              [{"token_id": 0}],
                              clean=False)

    def test_require_raises_audit_error(self):
        require(True, "okay")
        with self.assertRaises(AuditError):
            require(False, "bad")


if __name__ == "__main__":
    unittest.main()
