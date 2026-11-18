"""CPU-only contract checks for the frozen development runner.

These are synthetic result rows. No Voxtral development or test output is read.
"""

import json
import tempfile
import unittest
from pathlib import Path

import voxtral_dev_transfer_v1 as dev


def planned_cases():
    pairs = [f"{category}_{i:02d}" for category in ("animal", "color", "nature", "object")
             for i in range(3)]
    return [f"{pair}|{engine}|{template}|{word}"
            for pair in pairs for engine in ("piper", "kokoro")
            for template in ("dev_listen", "dev_hear") for word in ("alpha", "beta")]


def generation(case, site, kind, label, contrast):
    pair, engine, template, word = case.split("|")
    other = "beta" if word == "alpha" else "alpha"
    return {"type": "generation", "pair_id": pair, "engine": engine,
            "phrasing_id": template, "recipient": word, "opposite": other,
            "site": site, "kind": kind, "classification": label,
            "generated_token_ids": {"recipient": [11, 2], "opposite": [12, 2],
                                    "other": [13, 2], "malformed": [14]}[label],
            "canonical_scores": {word: contrast, other: 0.0},
            "recipient_minus_opposite": contrast}


def synthetic_rows(cases):
    rows = []
    for case in cases:
        word = case.rsplit("|", 1)[-1]
        for name in (f"native_{word}", f"word_vs_outside_{word}", f"padding_swap_{word}"):
            rows.append(generation(case, None, f"feature:{name}", "recipient", 2.0))
        rows.append(generation(case, None, "blank_baseline", "opposite", -2.0))
        for site in dev.first.SITES:
            rows.append(generation(case, site, "passive:clean", "recipient", 2.0))
            rows.append(generation(case, site, "passive:blank", "opposite", -2.0))
            for arm in dev.first.ARMS:
                for name in dev.first.PATCHES:
                    primary = name == "whole"
                    label = ("recipient" if primary else "opposite") if arm == "restore" else (
                        "opposite" if primary else "recipient")
                    score = (2.0 if primary else -2.0) if arm == "restore" else (
                        -2.0 if primary else 2.0)
                    rows.append(generation(case, site, f"patch:{arm}:{name}", label, score))
            rows.append(generation(case, site, "opposite_word_control", "opposite", -2.0))
    return rows


class DevRunnerContractTests(unittest.TestCase):
    def test_frozen_complete_answer_policy(self):
        classify = dev.first.classify
        self.assertEqual(classify([7, 2], "Alpha", {2, 9}, 2, "alpha", "beta"), "recipient")
        self.assertEqual(classify([8, 2], "Beta", {2, 9}, 2, "alpha", "beta"), "opposite")
        self.assertEqual(classify([8, 2], "Gamma", {2, 9}, 2, "alpha", "beta"), "other")
        self.assertEqual(classify([7, 2], "Alpha.", {2, 9}, 2, "alpha", "beta"), "malformed_surface")
        self.assertEqual(classify([9, 7, 2], "Alpha", {2, 9}, 2, "alpha", "beta"), "malformed_special")
        self.assertEqual(classify([7], "Alpha", {2, 9}, 2, "alpha", "beta"), "missing_eos")

    def test_all_planned_slots_and_strict_gates(self):
        cases = planned_cases()
        rows = synthetic_rows(cases)
        self.assertEqual((len(cases), len(rows)), (96, 4800))
        result = dev.summarize(rows, cases)
        self.assertTrue(result["development_advance_gate"])
        self.assertEqual(result["clean_correct"], 96)
        self.assertEqual(result["restoration_eligible"], 96)
        for site in dev.first.SITES:
            for arm in dev.first.ARMS:
                self.assertEqual(result["sites"][site][arm]["whole_word_success_of_96"], 96)
                self.assertTrue(result["sites"][site][arm]["time_preference_gate"])
        # Full slot count is a contract, even when one row was omitted.
        with self.assertRaises(ValueError):
            dev.summarize(rows[:-1], cases)

    def test_malformed_control_does_not_gain_strict_time_credit(self):
        cases = planned_cases()
        rows = synthetic_rows(cases)
        for row in rows:
            if row["site"] in dev.first.SITES and row["kind"] == "patch:restore:pre":
                row["classification"] = "malformed"
        result = dev.summarize(rows, cases)
        for site in dev.first.SITES:
            restoration = result["sites"][site]["restore"]
            self.assertTrue(restoration["whole_word_effect_gate"])
            self.assertFalse(restoration["time_preference_gate"])
            self.assertEqual(restoration["score_comparisons"]["pre"]["valid_of_96"], 0)
            self.assertEqual(restoration["score_comparisons"]["pre"]["finite_score_only_pairs_of_96"], 96)
        self.assertTrue(result["development_advance_gate"])

    def test_missing_or_wrong_score_fails_closed(self):
        cases = planned_cases()
        rows = synthetic_rows(cases)
        rows[0]["recipient_minus_opposite"] = 10.0
        with self.assertRaisesRegex(ValueError, "candidate score"):
            dev.summarize(rows, cases)

    def test_missing_self_and_passive_parity_fail_closed(self):
        cases = planned_cases()
        rows = synthetic_rows(cases)
        self_row = next(r for r in rows if r["kind"] == "patch:restore:self")
        self_row["type"] = "invalid_condition"
        with self.assertRaisesRegex(ValueError, "Self exact-suffix parity"):
            dev.summarize(rows, cases)
        self_row["type"] = "generation"
        passive = next(r for r in rows if r["kind"] == "passive:clean")
        passive["generated_token_ids"] = [99, 2]
        with self.assertRaisesRegex(ValueError, "Passive exact-suffix parity"):
            dev.summarize(rows, cases)

    def test_resume_requires_terminal_elapsed_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            start = output / "invocation_one.start.json"
            start.write_text(json.dumps({"stamp": "one"}))
            with self.assertRaisesRegex(ValueError, "Unreceipted"):
                dev.resume_budget_seconds(output, 0)
            self.assertEqual(dev.resume_budget_seconds(output, 0, active_stamp="one"), 0.0)
            (output / "invocation_one.error.json").write_text(json.dumps({
                "cumulative_budget_seconds": 123.5, "attempt_file": None}))
            self.assertEqual(dev.resume_budget_seconds(output, 0), 123.5)
            (output / "invocation_two.start.json").write_text(json.dumps({"stamp": "two"}))
            with self.assertRaisesRegex(ValueError, "Unreceipted"):
                dev.resume_budget_seconds(output, 0)
            self.assertEqual(dev.resume_budget_seconds(output, 0, active_stamp="two"), 123.5)


if __name__ == "__main__":
    unittest.main()
