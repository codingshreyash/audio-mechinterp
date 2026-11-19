"""No-weight fault tests for the narrow V2 GO-time provenance repair."""
from __future__ import annotations

import json
import copy
import unittest
from unittest import mock

from . import whisper_tail_causal_historical_audit_v2 as posthoc
from . import whisper_tail_causal_worker_v2 as worker


class HistoricalNoWeightTests(unittest.TestCase):
    def test_original_go_fails_only_on_later_edit_commit(self):
        with self.assertRaisesRegex(ValueError,
                                    "independent causal source/admission/review/GO order invalid"):
            worker.validate_go(posthoc.MODEL,worker.ADMISSIONS[posthoc.MODEL],
                               worker.GOS[posthoc.MODEL])
        receipt=posthoc.historical_receipt()
        self.assertEqual(receipt["original_go_commit"],posthoc.ORIGINAL_GO_COMMIT)
        self.assertEqual(receipt["v1_interpretation_sha256"],posthoc.ORIGINAL_MEMO_SHA)
        self.assertEqual(receipt["review_commit"],
                         posthoc.historical_commit(worker.REVIEW))

    def test_unexpected_original_go_error_remains_hold(self):
        with mock.patch.object(worker,"validate_go",side_effect=ValueError("source changed")):
            with self.assertRaisesRegex(ValueError,"different reason"):
                posthoc.historical_receipt()

    def test_historical_tree_byte_forgery_rejected(self):
        memo=posthoc.HERE/"WHISPER_TAIL_CAUSAL_INTERPRETATION_INDEPENDENT_V1.md"
        with mock.patch.object(posthoc,"historical_bytes",return_value=b"forged"):
            with self.assertRaisesRegex(ValueError,"GO-time artifact differs"):
                posthoc.check_artifact(memo,posthoc.ORIGINAL_MEMO_SHA)

    def test_current_byte_forgery_rejected_even_with_valid_go_tree(self):
        memo=posthoc.HERE/"WHISPER_TAIL_CAUSAL_INTERPRETATION_INDEPENDENT_V1.md"
        with mock.patch.object(posthoc,"current_bytes",return_value=b"forged"):
            with self.assertRaisesRegex(ValueError,"current or GO-time artifact differs"):
                posthoc.check_artifact(memo,posthoc.ORIGINAL_MEMO_SHA)

    def test_as_of_go_commit_differs_from_later_restoration(self):
        memo=posthoc.HERE/"WHISPER_TAIL_CAUSAL_INTERPRETATION_INDEPENDENT_V1.md"
        old=posthoc.historical_commit(memo)
        current=worker._commit(memo)
        self.assertNotEqual(old,current)
        self.assertTrue(worker._ancestor(old,posthoc.ORIGINAL_GO_COMMIT))
        self.assertFalse(worker._ancestor(current,posthoc.ORIGINAL_GO_COMMIT))
        self.assertEqual(posthoc.check_artifact(memo,posthoc.ORIGINAL_MEMO_SHA),old)

    def test_bad_go_time_source_order_rejected(self):
        original=posthoc.historical_commit
        memo=posthoc.HERE/"WHISPER_TAIL_CAUSAL_INTERPRETATION_INDEPENDENT_V1.md"
        def forged(path):
            return posthoc.ORIGINAL_GO_COMMIT if path==memo else original(path)
        with mock.patch.object(posthoc,"historical_commit",side_effect=forged):
            with self.assertRaisesRegex(ValueError,"GO-time source/admission/review lineage invalid"):
                posthoc.historical_receipt()

    def test_raw_attempt_hash_mismatch_rejected(self):
        current=posthoc.evidence_inventory()
        self.assertEqual(len(current),403)
        altered=copy.deepcopy(current)
        altered[0]["sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"raw attempt file/hash inventory changed"):
            posthoc.verify_inventory(altered,current)

    def test_saved_audit_substitution_is_scoped(self):
        saved=json.loads(posthoc.SAVED.read_text())
        candidate={"historical_receipt":{"original_admission":{"sentinel":"verified"}}}
        original=worker.validate_go
        original_sha=posthoc.sha
        def source_sha(path):
            return "a"*64 if path==posthoc.ADMISSION else original_sha(path)
        def replay_saved(model,admission_path,go_path):
            worker.validate_go(model,admission_path,go_path)
            return saved
        with mock.patch.object(posthoc,"validate_posthoc_go",return_value=candidate),\
             mock.patch.object(posthoc.saved,"audit",side_effect=replay_saved) as replay,\
             mock.patch.object(posthoc,"sha",side_effect=source_sha):
            result=posthoc.audit_saved()
        self.assertIs(worker.validate_go,original)
        self.assertEqual(result["status"],"PASS")
        replay.assert_called_once_with(posthoc.MODEL,worker.ADMISSIONS[posthoc.MODEL],
                                       worker.GOS[posthoc.MODEL])

    def test_saved_audit_failure_restores_original_go_checker(self):
        candidate={"historical_receipt":{"original_admission":{"sentinel":"verified"}}}
        original=worker.validate_go
        with mock.patch.object(posthoc,"validate_posthoc_go",return_value=candidate),\
             mock.patch.object(posthoc.saved,"audit",side_effect=ValueError("saved sidecar changed")):
            with self.assertRaisesRegex(ValueError,"saved sidecar changed"):
                posthoc.audit_saved()
        self.assertIs(worker.validate_go,original)

    def test_substituted_checker_accepts_only_original_call(self):
        candidate={"historical_receipt":{"original_admission":{"sentinel":"verified"}}}
        original=worker.validate_go
        def wrong_call(*_args):
            worker.validate_go(posthoc.MODEL,posthoc.ADMISSION,posthoc.GO)
        with mock.patch.object(posthoc,"validate_posthoc_go",return_value=candidate),\
             mock.patch.object(posthoc.saved,"audit",side_effect=wrong_call):
            with self.assertRaisesRegex(ValueError,"escaped original saved-audit call"):
                posthoc.audit_saved()
        self.assertIs(worker.validate_go,original)


if __name__=="__main__":unittest.main()
