"""CPU-only checks for the eight-record FIT causal runner and denominators."""

import json
import tempfile
import unittest
from pathlib import Path

import voxtral_fit_v2_causal_full as assay


def toy_records():
    rows = []
    keys = []
    for pair in ("animal_08", "color_05", "nature_09", "object_05"):
        for engine in ("piper", "kokoro"):
            for word in ("worda", "wordb"):
                key = f"{pair}|{engine}|fit_chosen|{word}"
                keys.append(key)
                base = {"type": "generation", "pair_id": pair, "engine": engine,
                        "recipient": word, "site": None}
                rows.append({**base, "kind": "blank_baseline", "classification": "other",
                             "recipient_minus_opposite": -1.0})
                rows.append({**base, "kind": f"feature:native_{word}", "classification": "recipient",
                             "recipient_minus_opposite": 1.0})
                for site in assay.first.SITES:
                    rows.append({**base, "site": site, "kind": "opposite_word_control",
                                 "classification": "opposite", "recipient_minus_opposite": -1.0})
                    for arm in assay.first.ARMS:
                        whole_label = "recipient" if arm == "restore" else "other"
                        whole_score = 1.0 if arm == "restore" else -1.0
                        rows.append({**base, "site": site, "kind": f"patch:{arm}:whole",
                                     "classification": whole_label,
                                     "recipient_minus_opposite": whole_score})
                        for quarter in range(4):
                            rows.append({**base, "site": site, "kind": f"patch:{arm}:word_{quarter}",
                                         "classification": whole_label,
                                         "recipient_minus_opposite": whole_score})
                        for control in assay.CONTROL_NAMES:
                            rows.append({**base, "site": site, "kind": f"patch:{arm}:{control}",
                                         "classification": "other" if arm == "restore" else "recipient",
                                         "recipient_minus_opposite": 0.0})
    return rows, keys


class FullRunnerTests(unittest.TestCase):
    def test_all_planned_denominators(self):
        rows, keys = toy_records()
        result = assay.summarize(rows, keys)
        self.assertEqual(result["planned_directions"], 16)
        self.assertEqual(result["restoration_eligible"], 16)
        for site in assay.first.SITES:
            for arm in assay.first.ARMS:
                item = result["sites"][site][arm]
                self.assertEqual(item["whole_word_success_of_16"], 16)
                self.assertTrue(item["whole_word_effect_gate"])
                self.assertTrue(item["time_preference_gate"])

    def test_ineligible_blank_does_not_count_restoration(self):
        rows, keys = toy_records()
        rows[0]["classification"] = "recipient"
        result = assay.summarize(rows, keys)
        self.assertEqual(result["restoration_eligible"], 15)
        self.assertEqual(result["sites"]["encoder31"]["restore"]["whole_word_success_of_16"], 15)

    def test_invalid_control_stays_in_denominator_and_withholds_time_label(self):
        rows, keys = toy_records()
        item = next(r for r in rows if r["site"] == "encoder31" and
                    r["kind"] == "patch:restore:pre")
        item["type"] = "invalid_condition"
        item["classification"] = "invalid"
        item["recipient_minus_opposite"] = None
        result = assay.summarize(rows, keys)
        arm = result["sites"]["encoder31"]["restore"]
        self.assertEqual(arm["whole_word_success_of_16"], 16)
        self.assertEqual(arm["invalid_main_or_control_conditions"], 1)
        self.assertFalse(arm["time_preference_gate"])

    def test_checkpoint_resume_checks_attempt_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            assay.base.exclusive_json(output / "meta.json", {"test": True})
            record = {"pair_id": "animal_08", "engine": "piper", "phrasing_id": "fit_chosen"}
            attempt = output / "record_01_attempt_01.jsonl"
            attempt.write_text("".join(json.dumps({"type": "capture"}) + "\n" for _ in range(8)) +
                               "".join(json.dumps({"type": "generation"}) + "\n" for _ in range(100)))
            cp = {"ordinal": 1, "record_key": assay.record_key(record),
                  "invocation_stamp": "20261118T000000000000Z",
                  "run_meta_sha256": assay.base.sha(output / "meta.json"),
                  "attempt_file": attempt.name, "attempt_sha256": assay.base.sha(attempt),
                  "progress_records": 108, "capture_records": 8,
                  "generation_records": 100, "teacher_forced_forwards": 200,
                  "invalid_conditions": 0, "seconds": 1.0,
                  "cumulative_budget_seconds": 2.0}
            assay.base.exclusive_json(output / "record_01.checkpoint.json", cp)
            self.assertEqual(assay.completed_prefix(output, [record], {}), 1)
            attempt.write_text("changed\n")
            with self.assertRaises(ValueError):
                assay.completed_prefix(output, [record], {})

    def test_incomplete_attempt_requires_accounted_elapsed_debt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            partial = output / "record_01_attempt_01.jsonl"
            partial.write_text("{}\n")
            with self.assertRaises(ValueError):
                assay.resume_budget_seconds(output, 0)
            error = output / "invocation_20261118T000000000000Z.error.json"
            assay.base.exclusive_json(error, {"attempt_file": partial.name,
                                             "cumulative_budget_seconds": 12.5})
            self.assertEqual(assay.resume_budget_seconds(output, 0), 12.5)
            later = output / "invocation_20261118T000001000000Z.error.json"
            assay.base.exclusive_json(later, {"attempt_file": None,
                                             "cumulative_budget_seconds": 11.0})
            with self.assertRaises(ValueError):
                assay.resume_budget_seconds(output, 0)

    def test_kill_during_model_load_fails_closed_without_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            stamp = "20261118T000000000000Z"
            (output / f"invocation_{stamp}.memory.jsonl").write_text("")
            with self.assertRaises(ValueError):
                assay.resume_budget_seconds(output, 0)
            assay.base.exclusive_json(output / f"invocation_{stamp}.error.json",
                                      {"attempt_file": None, "cumulative_budget_seconds": 3.0})
            self.assertEqual(assay.resume_budget_seconds(output, 0), 3.0)


if __name__ == "__main__":
    unittest.main()
