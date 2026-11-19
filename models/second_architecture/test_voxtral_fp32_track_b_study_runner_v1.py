"""CPU fake checks only; no Voxtral checkpoint or Track B model inference."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_voxtral_track_b_answer_link_core_v1 import FakeProcessor, fixture  # noqa: E402
import voxtral_fp32_track_b_study_runner_v1 as study  # noqa: E402


class LedgerTests(unittest.TestCase):
    def test_ledger_rejects_overlap_duplicate_and_missing(self):
        tick = iter([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ledger = study.CallLedger(context_id="fake", now=lambda: next(tick))
        ledger.guard("before|generation|clean")
        with self.assertRaisesRegex(ValueError, "overlapping"):
            ledger.guard("before|generation|zero")
        ledger.guard("after|generation|clean")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            ledger.guard("before|generation|clean")
        with self.assertRaisesRegex(ValueError, "differ"):
            ledger.seal(expected_generation=2, expected_no_grad=0, expected_grad=0)
        self.assertEqual(ledger.seal(expected_generation=1, expected_no_grad=0,
                                     expected_grad=0)["counts"]["generation"], 1)
        with self.assertRaisesRegex(ValueError, "order/identity"):
            ledger.seal(expected_generation=1, expected_no_grad=0,
                        expected_grad=0, expected_labels=["generation|zero"])

    def test_one_context_complete_fake_ledger(self):
        model, batch, _source, donor = fixture()
        processor = FakeProcessor()
        context = {"index": 0, "context_id": "fake_pair|piper|banana->tomato",
                   "recipient": "banana", "donor": "tomato",
                   "audio_placeholder_count": 3, "projector_word_rows": [1],
                   "recipient_aliases": [[7, 2]], "donor_aliases": [[8, 2]]}
        # Natural donor is the opposite direction's clean generation; it is
        # referenced, not charged as an eighth generation in this context.
        donor_batch = {**batch, "input_features": donor[None]}
        donor_clean, donor_state = study.clean_generation_with_capture(
            model, processor, donor_batch, {**context, "recipient": "tomato",
                                          "donor": "banana"}, pad=2)
        ledger = study.CallLedger(context_id=context["context_id"])
        clean, source = study.clean_generation_with_capture(
            model, processor, batch, context, pad=2, call_guard=ledger.guard)
        evidence, selected = study.run_one_context(
            model, processor, context, batch, source, donor_state, clean,
            donor_clean, pad=2, call_guard=ledger.guard)
        calls = ledger.seal(expected_generation=7, expected_no_grad=12,
                            expected_grad=2,
                            expected_labels=study.expected_context_call_labels(context))
        self.assertEqual(len(calls["calls"]), 21)
        self.assertEqual(set(evidence["generations"]),
                         set(study.GENERATIONS) | {"natural_donor"})
        self.assertEqual(evidence["actual_counts"],
                         {"generations": 7, "no_grad_singleton_score_forwards": 12,
                          "grad_singleton_score_forwards": 2})
        self.assertEqual(set(selected), {"recipient_0", "donor_0"})
        self.assertEqual(evidence["decision"]["parity_failures"], [])
        self.assertEqual(evidence["generations"]["clean"]["suffix_ids"],
                         evidence["generations"]["zero"]["suffix_ids"])

    def test_cli_is_validate_only(self):
        self.assertFalse(hasattr(study, "load_model"))
        self.assertEqual(study.CONTEXT_COUNT, 16)
        self.assertEqual(study.GENERATIONS_PER_CONTEXT, 7)
        self.assertEqual(study.MAX_NO_GRAD_SCORES_PER_CONTEXT, 72)
        self.assertEqual(study.MAX_GRAD_SCORES_PER_CONTEXT, 12)


if __name__ == "__main__":
    unittest.main()
