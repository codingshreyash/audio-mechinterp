"""Synthetic auditor tests only; no ASR model or human response is simulated as evidence."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import audit_asr_listening_v1 as auditor
import listening_server_v1 as listening


def case(ident: str, target: str = "otter") -> dict:
    return {"opaque_id": ident, "pair_id": "synthetic_pair", "split": "development",
            "category": "synthetic", "engine": "piper", "template_id": "synthetic",
            "target_word": target, "paired_alternative": "other",
            "reference_script": f"Say {target} now.", "wav_sha256": "a" * 64,
            "pcm_int16_sha256": "b" * 64}


def asr_row(ident: str, text: str, model: str) -> dict:
    row = {"ordinal": 1, "opaque_id": ident, "wav_sha256": "a" * 64,
           "pcm_int16_sha256": "b" * 64, "raw_transcript": text}
    if model == "whisper":
        row.update(generated_token_ids=[1, 2, 3], decoder_cap=128, ended_with_eos=True)
    else:
        row.update(frame_argmax_token_ids=[1, 2, 3], ctc_frame_count=3)
    return row


def response(ident: str, stage: str, answer: str, clear: bool = True,
             confidence: int = 5) -> dict:
    return {"kind": "initial", "stage": stage,
            "reviewer_id": {"primary": "listener_one", "secondary": "listener_two",
                            "tertiary": "listener_three"}[stage],
            "opaque_id": ident, "middle_word": answer, "clear": clear,
            "confidence": confidence, "play_count": 1}


class AuditorTests(unittest.TestCase):
    def test_frozen_preflight_rebuilds_all_480_without_outcomes(self):
        cases, order, pins = auditor.frozen_cases()
        self.assertEqual((len(cases), len(order)), (480, 480))
        self.assertEqual(set(pins["models"]), {"whisper", "wav2vec2"})

    def test_tokenization_and_wer_are_literal(self):
        self.assertEqual(auditor.tokens("  O’NEILL, café 10!"),
                         ["o’neill", "café", "10"])
        self.assertEqual(auditor.tokens("tan"), ["tan"])
        self.assertNotEqual(auditor.tokens("tan"), auditor.tokens("10"))
        self.assertEqual(auditor.word_errors(["say", "otter", "now"],
                                             ["say", "other", "now"]), 1)

    def test_flagged_secondary_and_tertiary_gate(self):
        ident = "a" * 20
        cases = [case(ident)]
        asr = {"whisper": {ident: asr_row(ident, "Say otter now.", "whisper")},
               "wav2vec2": {ident: asr_row(ident, "SAY OTHER NOW", "wav2vec2")}}
        first = response(ident, "primary", "otter")
        second = response(ident, "secondary", "other")
        third = response(ident, "tertiary", "otter")
        pending = auditor.audit(cases, [ident], asr, [first, second])
        self.assertEqual(pending["status"], "stop")
        self.assertIn("missing_required_tertiary", pending["failed_cases"][0]["problems"])
        passed = auditor.audit(cases, [ident], asr, [first, second, third])
        self.assertEqual(passed["status"], "go")
        self.assertEqual(passed["quality_confirmed_count"], 1)
        self.assertEqual(passed["asr_disagreement_count"], 1)
        self.assertEqual(passed["aggregate"]["wav2vec2"]["all"]["word_errors"], 1)
        wrong_third = auditor.audit(cases, [ident], asr,
                                    [first, second, response(ident, "tertiary", "other")])
        self.assertEqual(wrong_third["status"], "stop")

    def test_unflagged_primary_pass_and_extra_secondary_stop(self):
        ident = "b" * 20
        cases = [case(ident)]
        asr = {key: {ident: asr_row(ident, "Say otter now.", key)}
               for key in ("whisper", "wav2vec2")}
        first = response(ident, "primary", "otter")
        self.assertEqual(auditor.audit(cases, [ident], asr, [first])["status"], "go")
        extra = auditor.audit(cases, [ident], asr,
                              [first, response(ident, "secondary", "otter")])
        self.assertEqual(extra["status"], "stop")
        self.assertIn("unflagged_extra_secondary", extra["failed_cases"][0]["problems"])
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            auditor.audit(cases, [ident], {"whisper": asr["whisper"], "wav2vec2": {}},
                          [first])

    def test_response_chain_external_hash_order_and_revisions(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "listening"
            provenance = {"private_sha256": auditor.EXPECTED["cases"],
                          "queue_sha256": auditor.EXPECTED["queue"],
                          "manifest_sha256": auditor.EXPECTED["manifest"],
                          "binding_sha256": auditor.EXPECTED["binding"]}
            log = listening.ResponseLog(run, provenance, create=True)
            a, b = "a" * 20, "b" * 20
            try:
                with self.assertRaisesRegex(ValueError, "browser user-agent/version"):
                    log.append(response(a, "primary", "otter"), {a, b},
                               browser_user_agent="")
                log.append(response(a, "primary", "otter"), {a, b},
                           browser_user_agent="SyntheticBrowser/1")
                log.append(response(b, "primary", "otter"), {a, b},
                           browser_user_agent="SyntheticBrowser/1")
                rows = auditor.verify_responses(log.path, auditor.sha(log.path),
                                                 [a, b], log.meta_path)
                self.assertEqual(len(rows), 2)
                with self.assertRaisesRegex(ValueError, "response SHA"):
                    auditor.verify_responses(log.path, "0" * 64, [a, b], log.meta_path)
                with self.assertRaisesRegex(ValueError, "queue order"):
                    auditor.verify_responses(log.path, auditor.sha(log.path),
                                             [b, a], log.meta_path)
                log.append({**response(a, "primary", "otter"),
                            "kind": "correction", "prior_sequence": 1,
                            "reason": "Typo in first answer"}, {a, b},
                           browser_user_agent="SyntheticBrowser/1")
                revised = auditor.verify_responses(log.path, auditor.sha(log.path),
                                                    [a, b], log.meta_path)
                self.assertEqual(revised[2]["prior_sequence"], 1)
                meta = json.loads(log.meta_path.read_text())
                meta["server_version"] = "changed-version"
                log.meta_path.write_text(json.dumps(meta))
                with self.assertRaisesRegex(ValueError, "provenance/source changed"):
                    auditor.verify_responses(log.path, auditor.sha(log.path),
                                             [a, b], log.meta_path)
                meta["server_version"] = "BlindListening/1"
                log.meta_path.write_text(json.dumps(meta))
                changed = revised[0].copy()
                changed["browser_user_agent"] = "direct_test_client_no_browser"
                changed.pop("record_sha256")
                changed["record_sha256"] = auditor.hashlib.sha256(
                    auditor.canonical(changed)).hexdigest()
                with log.path.open("wb") as stream:
                    stream.write(auditor.canonical(changed) + b"\n")
                with self.assertRaisesRegex(ValueError, "response values invalid"):
                    auditor.verify_responses(log.path, auditor.sha(log.path),
                                             [a, b], log.meta_path)
            finally:
                log.close()

    def test_asr_receipt_detects_missing_or_changed_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            ident = "c" * 20
            cases = [case(ident)]
            pins = json.loads(auditor.PINS.read_text())
            model = pins["models"]["whisper"]
            freeze = "f" * 64
            meta = {"status": "started_raw_asr_only", "model_key": "whisper",
                    "model_id": model["repo_id"], "model_revision": model["revision"],
                    "model_files_sha256": model["files_sha256"],
                    "script_sha256": auditor.EXPECTED["runner"],
                    "pins_sha256": auditor.EXPECTED["pins"], "freeze_sha256": freeze,
                    "technical_binding_sha256": auditor.EXPECTED["binding"],
                    "asr_inputs_sha256": auditor.EXPECTED["inputs"],
                    "provenance_cases_sha256": auditor.EXPECTED["cases"],
                    "planned_clips": 1, "device": "cpu", "threads": 4,
                    "generation": {"whisper": {"do_sample": False, "num_beams": 1,
                                                "max_new_tokens": 128,
                                                "processor_attention_mask": True},
                                   "wav2vec2": {"decoder": "frame_argmax_ctc_greedy"}}}
            (run / "meta.json").write_text(json.dumps(meta))
            raw = run / "raw.jsonl"
            raw.write_text(json.dumps(asr_row(ident, "Say otter now.", "whisper")) + "\n")
            done = {"status": "complete_raw_asr_no_reference_comparison",
                    "model_key": "whisper", "planned_clips": 1, "raw_rows": 1,
                    "meta_sha256": auditor.sha(run / "meta.json"),
                    "raw_sha256": auditor.sha(raw), "no_model_answer_inference": True}
            (run / "complete.json").write_text(json.dumps(done))
            saved = auditor.sha(run / "complete.json")
            self.assertEqual(len(auditor.verify_asr_run(run, "whisper", cases,
                                                         pins, freeze, saved)), 1)
            raw.write_text("")
            with self.assertRaisesRegex(ValueError, "completion/hash mismatch"):
                auditor.verify_asr_run(run, "whisper", cases, pins, freeze, saved)
            leaked = asr_row(ident, "Say otter now.", "whisper")
            leaked["target_word"] = "otter"
            raw.write_text(json.dumps(leaked) + "\n")
            done["raw_sha256"] = auditor.sha(raw)
            (run / "complete.json").write_text(json.dumps(done))
            with self.assertRaisesRegex(ValueError, "raw row 1 changed"):
                auditor.verify_asr_run(run, "whisper", cases, pins, freeze,
                                       auditor.sha(run / "complete.json"))

    def test_incomplete_evidence_writes_explicit_stop_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "audit.json"
            with mock.patch.object(sys, "argv", ["audit_asr_listening_v1.py",
                                                 "--output", str(output)]):
                auditor.main()
            receipt = json.loads(output.read_text())
            self.assertEqual(receipt["status"], "stop")
            self.assertIn("external receipt hashes required", receipt["gate_errors"][0])


if __name__ == "__main__":
    unittest.main()
