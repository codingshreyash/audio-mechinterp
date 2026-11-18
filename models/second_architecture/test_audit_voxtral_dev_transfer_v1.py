"""CPU-only synthetic checks for the prospective development auditor."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import audit_voxtral_dev_transfer_v1 as audit


class FakeProcessor:
    def batch_decode(self, sequences, skip_special_tokens):
        assert len(sequences) == 1
        word = {11: "Bear", 12: "Wolf"}[sequences[0][0]]
        return [word if skip_special_tokens else word + "</s>"]


class DevelopmentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = json.loads(audit.P["masks"][0].read_text())["records"]

    def test_frozen_cpu_preflight_and_slot_grammar(self):
        report = audit.validate_only()
        self.assertEqual((report["records"], report["directions"], report["planned_progress_rows"]),
                         (48, 96, 5184))
        self.assertTrue(all(len(audit.planned_slots(r["words"])) == 108 for r in self.records))

    def test_suffix_classifier_is_strict(self):
        self.assertEqual(audit.classify([11, 2], "Bear", {2}, "bear", "wolf"), "recipient")
        self.assertEqual(audit.classify([11, 2], '"Bear"', {2}, "bear", "wolf"),
                         "malformed_surface")
        self.assertEqual(audit.classify([2, 11, 2], "Bear", {2}, "bear", "wolf"),
                         "malformed_special")
        self.assertEqual(audit.classify([11], "Bear", {2}, "bear", "wolf"), "missing_eos")

    def base_row(self):
        return {"type": "generation", "kind": "feature:native_bear", "site": None,
                "pair_id": "animal_00", "engine": "piper", "phrasing_id": "dev_listen",
                "recipient": "bear", "opposite": "wolf", "generated_token_ids": [11, 2],
                "raw_decoded": "Bear", "decoded_with_special_tokens": "Bear</s>",
                "terminal_eos": True, "nonterminal_special_token_ids": [],
                "classification": "recipient", "canonical_scores": {"bear": 1.0, "wolf": 0.0},
                "recipient_minus_opposite": 1.0, "seconds_including_scores": 0.1,
                "mps_driver_bytes_after": 12345, "hook_calls_including_two_scores": 0,
                "hook_events": [], "edited_rows_shape": None, "edited_rows_dtype": None}

    def test_generated_suffix_reclassification_catches_tampering(self):
        r = self.base_row()
        audit.verify_row(r, self.records[0], {}, FakeProcessor(), {2})
        r["classification"] = "opposite"
        with self.assertRaisesRegex(AssertionError, "suffix decode/classification"):
            audit.verify_row(r, self.records[0], {}, FakeProcessor(), {2})
        r["classification"] = "recipient"
        r["nonterminal_special_token_ids"] = [2]
        with self.assertRaisesRegex(AssertionError, "suffix decode/classification"):
            audit.verify_row(r, self.records[0], {}, FakeProcessor(), {2})

    def test_passive_parity_and_invalid_self_fail_closed(self):
        clean = self.base_row()
        passive = dict(clean, kind="passive:clean", site="encoder31",
                       generated_token_ids=[12, 2], raw_decoded="Wolf",
                       decoded_with_special_tokens="Wolf</s>", classification="opposite",
                       hook_calls_including_two_scores=3,
                       hook_events=[{"shape":[1,1500,1280],"dtype":"torch.bfloat16"}]*3)
        byslot = {("generation", None, "bear", "feature:native_bear"): clean}
        with self.assertRaisesRegex(AssertionError, "passive/self suffix-token parity"):
            audit.verify_row(passive, self.records[0], byslot, FakeProcessor(), {2})
        invalid = {"type":"invalid_condition", "kind":"patch:restore:self",
                   "site":"encoder31", "recipient":"bear", "opposite":"wolf",
                   "pair_id":"animal_00", "engine":"piper", "phrasing_id":"dev_listen",
                   "classification":"invalid", "recipient_minus_opposite":None,
                   "edit":{"valid":False, "mode":"self",
                           "rows":self.records[0]["site_maps"]["encoder31"]["whole_word_rows"]}}
        with self.assertRaisesRegex(AssertionError, "self patch invalid"):
            audit.verify_row(invalid, self.records[0], byslot, FakeProcessor(), {2})

    def synthetic_rows(self):
        rows, cases = [], []
        for record in self.records:
            pair, voice, template = (record[k] for k in ("pair_id","engine","phrasing_id"))
            for word in record["words"]:
                opposite = next(w for w in record["words"] if w != word)
                cases.append("|".join((pair,voice,template,word)))
                for key in audit.planned_slots(record["words"]):
                    if key[0] == "capture" or key[2] != word:
                        continue
                    if key[0] == "patch":
                        kind = f"patch:{key[3]}:{key[4]}"
                        site = key[1]
                        arm, patch = key[3], key[4]
                        label = ("recipient" if patch=="whole" else "other") if arm=="restore" else (
                            "other" if patch=="whole" else "recipient")
                        contrast = (1.0 if patch=="whole" else 0.0) if arm=="restore" else (
                            -1.0 if patch=="whole" else 0.0)
                    else:
                        site, kind = key[1], key[3]
                        label = ("malformed_surface" if kind in ("blank_baseline", "passive:blank")
                                 else "opposite" if kind=="opposite_word_control" else "recipient")
                        contrast = 0.0
                    rows.append({"type":"generation","pair_id":pair,"engine":voice,
                                 "phrasing_id":template,"recipient":word,"opposite":opposite,
                                 "site":site,"kind":kind,"classification":label,
                                 "recipient_minus_opposite":contrast})
        return rows,cases

    def test_strict_gate_counts_and_malformed_control_tamper(self):
        rows,cases = self.synthetic_rows()
        self.assertEqual((len(rows),len(cases)),(4800,96))
        result = audit.independent_summary(rows,cases)
        self.assertTrue(result["development_advance_gate"])
        self.assertEqual(result["sites"]["encoder31"]["restore"]["whole_word_success_of_96"],96)
        # The scores remain finite, but 20 malformed post-control answers must
        # fail the >=84 valid-complete-answer rule without removing cases.
        changed = 0
        for row in rows:
            if row["site"]=="encoder31" and row["kind"]=="patch:restore:post":
                row["classification"]="malformed_surface"
                changed += 1
                if changed==20: break
        self.assertEqual(changed,20)
        result = audit.independent_summary(rows,cases)
        self.assertFalse(result["sites"]["encoder31"]["restore"]["time_preference_gate"])
        self.assertEqual(result["sites"]["encoder31"]["restore"]
                         ["score_comparisons"]["post"]["valid_of_96"],76)
        self.assertEqual(result["sites"]["encoder31"]["restore"]
                         ["score_comparisons"]["post"]["finite_score_only_pairs_of_96"],96)

    def test_receipt_chain_rejects_missing_elapsed_debt(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            (root/"result.json").write_text('{"ok":true}')
            old="20261118T100000000000Z"
            new="20261118T100100000000Z"
            def start(stamp,prior):
                return {"stamp":stamp,"status":"started","runner_sha256":audit.P["runner"][1],
                        "protocol_sha256":audit.P["protocol"][1],
                        "masks_sha256":audit.P["masks"][1],
                        "wall_cap_seconds":audit.CAP_SECONDS,"prior_budget_seconds":prior}
            (root/f"invocation_{old}.start.json").write_text(json.dumps(start(old,0.0)))
            (root/f"invocation_{old}.error.json").write_text(json.dumps({
                "stamp":old,"status":"incomplete_invocation","cumulative_budget_seconds":3.0,
                "attempt_file":"record_01_attempt_01.jsonl","memory_log_sha256":None}))
            (root/"record_01_attempt_01.jsonl").write_text("partial\n")
            (root/f"invocation_{new}.start.json").write_text(json.dumps(start(new,3.0)))
            (root/f"invocation_{new}.complete.json").write_text(json.dumps({
                "stamp":new,"status":"complete","cumulative_budget_seconds":10.0,
                "result_sha256":audit.sha(root/"result.json")}))
            (root/"record_01_attempt_02.jsonl").write_text("done\n")
            checkpoints=[{"ordinal":1,"attempt_file":"record_01_attempt_02.jsonl",
                          "invocation_stamp":new,"cumulative_budget_seconds":8.0,
                          "mps_driver_observed_max_bytes":1234}]
            self.assertEqual(audit.verify_receipts(root,checkpoints)["cumulative_seconds"],10.0)
            (root/f"invocation_{new}.start.json").write_text(json.dumps(start(new,2.0)))
            with self.assertRaisesRegex(AssertionError,"invocation start/debt"):
                audit.verify_receipts(root,checkpoints)


if __name__ == "__main__":
    unittest.main()
