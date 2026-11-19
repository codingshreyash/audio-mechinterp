"""No-weight checks for the future Qwen action runner evidence contract."""
from __future__ import annotations

import hashlib
import json
import math
import unittest

import numpy as np

from . import runner_contract_v1 as contract
from .endpoint import LABELS
from .readout_v2 import EOS_TOKEN_ID, LABEL_TOKEN_IDS


class RunnerContractTests(unittest.TestCase):
    class Tokenizer:
        eos_token_id = EOS_TOKEN_ID
        all_special_ids = [EOS_TOKEN_ID]

        def decode(self, ids, skip_special_tokens=False):
            return "".join({LABEL_TOKEN_IDS[0]: "LIST",
                            LABEL_TOKEN_IDS[1]: "ASK",
                            LABEL_TOKEN_IDS[2]: "CANCEL"}.get(x, "?")
                           for x in ids)

    def test_real_source_preflight_is_read_only_and_still_holds_inference(self):
        receipt = contract.source_preflight()
        self.assertEqual(receipt["status"], "source_bytes_pass_candidate_inference_hold")
        self.assertEqual(receipt["human_reviews"], 0)
        self.assertIs(receipt["candidate_inference_authorized"], False)

    def test_plan_keeps_locked_cases_out_of_development(self):
        plan = contract.planned_case_ids()
        self.assertEqual([len(plan[k]) for k in (
            "typed_pilot", "fit_text", "development_typed",
            "development_audio", "locked_typed", "locked_audio")],
            [1, 18, 6, 6, 6, 12])
        self.assertTrue(all(x.startswith("D_") for x in plan["development_audio"]))
        self.assertFalse(set(plan["development_audio"]) & set(plan["locked_audio"]))

    def _triplet(self):
        vocab = EOS_TOKEN_ID + 1
        first = np.zeros(vocab, dtype=np.float32)
        second = np.zeros(vocab, dtype=np.float32)
        score = -2 * math.log(vocab)
        prompt_ids = [12, 34, 56]
        prompt_hash = hashlib.sha256(json.dumps(prompt_ids,
                                     separators=(",", ":")).encode()).hexdigest()
        return [{"label": label,
                 "target_ids": [token, EOS_TOKEN_ID],
                 "prompt_ids": prompt_ids.copy(),
                 "prompt_ids_sha256": prompt_hash,
                 "teacher_forced_input_ids": prompt_ids + [token],
                 "logit_positions": [2, 3],
                 "hook_calls": {site: 1 for site in contract.SITES},
                 "first_logits": first.copy(),
                 "after_label_logits": second.copy(),
                 "claimed_score": score,
                 "states": {site: np.ones(8, dtype=np.float32)
                            for site in contract.SITES},
                 "gradients": {site: np.ones(8, dtype=np.float32)
                               for site in contract.SITES}}
                for label, token in zip(LABELS, LABEL_TOKEN_IDS, strict=True)]

    def test_full_target_score_and_prompt_state_binding(self):
        rows = self._triplet()
        result = contract.validate_teacher_forced_triplet(rows, width=8)
        self.assertEqual(len(result["scores"]), 3)
        self.assertTrue(all(v == 0 for v in
                            result["cross_suffix_state_maxabs"].values()))
        rows[1]["states"][contract.SITES[0]][0] = 1.01
        with self.assertRaisesRegex(ValueError, "cross-suffix"):
            contract.validate_teacher_forced_triplet(rows, width=8)

    def test_target_and_hook_or_score_drift_fail(self):
        rows = self._triplet()
        rows[0]["target_ids"][0] = 3
        with self.assertRaisesRegex(ValueError, "target"):
            contract.validate_teacher_forced_triplet(rows, width=8)
        rows = self._triplet()
        rows[0]["hook_calls"][contract.SITES[0]] = 2
        with self.assertRaisesRegex(ValueError, "hook"):
            contract.validate_teacher_forced_triplet(rows, width=8)
        rows = self._triplet()
        rows[2]["claimed_score"] += 0.01
        with self.assertRaisesRegex(ValueError, "score differs"):
            contract.validate_teacher_forced_triplet(rows, width=8)
        rows = self._triplet()
        rows[1]["logit_positions"] = [3, 4]
        with self.assertRaisesRegex(ValueError, "logits alignment"):
            contract.validate_teacher_forced_triplet(rows, width=8)

    def test_complete_control_parity_uses_full_token_ids(self):
        def row(token):
            return {"status": "complete", "label": "LIST",
                    "continuation_ids": [token, EOS_TOKEN_ID]}
        original = row(LABEL_TOKEN_IDS[0])
        donor = {"status": "complete", "label": "CANCEL",
                 "continuation_ids": [LABEL_TOKEN_IDS[2], EOS_TOKEN_ID]}
        self.assertTrue(all(contract.validate_complete_controls(
            original, dict(original), dict(original), donor, donor,
            self.Tokenizer()).values()))
        with self.assertRaisesRegex(ValueError, "parity"):
            contract.validate_complete_controls(
                original, dict(original), donor, donor, donor,
                self.Tokenizer())
        with self.assertRaisesRegex(ValueError, "verified EOS-complete"):
            contract.validate_complete_controls(
                original, dict(original), dict(original),
                {"status": "complete", "label": "LIST",
                 "continuation_ids": donor["continuation_ids"]},
                donor, self.Tokenizer())

    def test_resource_limits_and_candidate_entrypoints(self):
        good = {"wall_seconds": 299.0,
                "peak_mps_driver_bytes": 23 * 1024**3,
                "peak_tree_rss_bytes": 29 * 1024**3}
        self.assertTrue(contract.validate_pilot_resources(good)[
            "host_audit_still_required"])
        for key, limit in (("wall_seconds", contract.PILOT_SECONDS + 1),
                           ("peak_mps_driver_bytes", contract.PILOT_MPS_LIMIT),
                           ("peak_tree_rss_bytes", contract.PILOT_RSS_LIMIT)):
            bad = {**good, key: limit}
            with self.assertRaisesRegex(ValueError, "limit exceeded"):
                contract.validate_pilot_resources(bad)
        with self.assertRaisesRegex(RuntimeError, "candidate inference HOLD"):
            contract.refuse_candidate_inference()


if __name__ == "__main__":
    unittest.main()
