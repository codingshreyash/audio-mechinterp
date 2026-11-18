"""Synthetic receipt/ledger tests only; no actual listening or model inference."""

from __future__ import annotations

import hashlib
import hmac
import json
import random
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from .audio_qa_v2 import (ASR_DIRS, COMPARISON, HERE, RENDER, PINS,
                          canonical_bytes, check_audio_qa_v2, digest)
from .calendar_stub import CASES, grant_signing_bytes
from .audio_qa_v2_listen import prepare, listen
from .two_pass_adapter import run_two_pass
from .test_two_pass_adapter import FakeModel, FakeRunner


KEY = b"synthetic-test-secret-exactly-32-bytes-long"
CONTROLLER = "synthetic-controller"
BLIND_TIME = "2026-11-17T10:00:00Z"
AWARE_TIME = "2026-11-17T10:01:00Z"
QA_TIME = "2026-11-17T11:00:00Z"
ANCHOR_TIME = "2026-11-18T00:00:00Z"


def write_record(root: Path, name: str, value: dict) -> dict:
    path = root / name
    path.write_bytes(canonical_bytes(value))
    return {"path": name, "sha256": digest(path)}


def fields(case_id: str) -> dict:
    return {
        "clear_speech": True,
        "date_heard": "not_applicable" if case_id == "D" else "2026-11-19",
        "design_review_title_heard": "Design Review" if case_id in ("B", "C", "B_literal") else "not_applicable",
        "first_action_heard": CASES[case_id]["expected_first_action"],
        "cancel_requires_further_confirmation_heard": True if case_id in ("B", "B_literal") else "not_applicable",
        "c_confirmation_phrase_heard_exactly": True if case_id == "C" else "not_applicable",
        "cancellation_intent_heard": "explicit" if case_id == "C" else
                                      "possible" if case_id in ("B", "B_literal") else "none",
        "case_semantics_preserved": True,
    }


class AudioQAV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.render = json.loads(RENDER.read_bytes())
        self.comparison = json.loads(COMPARISON.read_bytes())
        self.manifest = {
            "schema": "voicechat_audio_qa_manifest_v2",
            "amendment_commit": "f8f5bb6",
            "amendment_sha256": PINS["amendment"],
            "render_manifest_sha256": PINS["render_manifest"],
            "scripts_sha256": PINS["scripts"],
            "comparison_sha256": PINS["comparison"],
            "asr_sources": self.comparison["sources"],
            "reviewer_roster": {"primary": "synthetic primary", "secondary": "synthetic secondary"},
            "blind_order_seed": "synthetic-seed",
            "blind_order": {"primary": [], "secondary": []},
            "rows": [], "eligibility": [], "c_grants": [],
        }
        for render_row in self.render["rows"]:
            self._row(render_row)
        opaque = [r["opaque_id"] for r in self.comparison["rows"]["whisper"]]
        random.Random(self.manifest["blind_order_seed"]).shuffle(opaque)
        wav_by_id = {r["opaque_id"]: r["wav_sha256"]
                     for r in [json.loads(line) for line in
                               (ASR_DIRS["whisper"] / "raw.jsonl").read_text().splitlines()]}
        assignment = {"schema": "voicechat_blind_bundle_v2",
                      "created_utc": "2026-11-17T09:00:00Z",
                      "blind_order_seed": self.manifest["blind_order_seed"],
                      "render_manifest_sha256": PINS["render_manifest"],
                      "comparison_sha256": PINS["comparison"],
                      "opaque_order": [{"opaque_id": item, "wav_sha256": wav_by_id[item]}
                                       for item in opaque]}
        self.manifest["blind_assignment"] = write_record(self.root, "blind_assignment.json", assignment)
        for reviewer in self.manifest["blind_order"]:
            assigned = set(self.manifest["blind_order"][reviewer])
            self.manifest["blind_order"][reviewer] = [item for item in opaque if item in assigned]
        self.manifest_path = self.root / "manifest.json"
        self.ledger_path = self.root / "ledger.json"
        self.save()

    def _listener(self, label: str, opaque: str, script: str,
                  case_id: str, suffix: str) -> dict:
        blind = {"schema": "voicechat_listener_stage_v2", "opaque_id": opaque,
                 "reviewer_audit_id": label, "stage": "blind", "recorded_utc": BLIND_TIME,
                 "verbatim_free_transcript": script, "clear_speech": True,
                 "confidence": "high", "uncertain_spans": []}
        blind_desc = write_record(self.root, f"{suffix}_{label}_blind.json", blind)
        aware = {"schema": "voicechat_listener_stage_v2", "opaque_id": opaque,
                 "reviewer_audit_id": label, "stage": "script_aware",
                 "recorded_utc": AWARE_TIME, "blind_recorded_utc": BLIND_TIME,
                 "blind_record_sha256": blind_desc["sha256"],
                 "eligibility_judgment": "eligible", "notes": "", **fields(case_id)}
        aware_desc = write_record(self.root, f"{suffix}_{label}_aware.json", aware)
        return {"reviewer_audit_id": label, "blind": blind_desc, "script_aware": aware_desc}

    def _row(self, render_row: dict) -> None:
        case = render_row["case_id"]
        voice = render_row["voice_id"]
        wav = RENDER.parent / render_row["final_path"]
        with wave.open(str(wav), "rb") as stream:
            pcm = stream.readframes(stream.getnframes())
        comparisons = {engine: next(row for row in self.comparison["rows"][engine]
                                    if (row["case_id"], row["voice_id"]) == (case, voice))
                       for engine in ASR_DIRS}
        obs = {}
        for engine, row in comparisons.items():
            raw_rows = [json.loads(line) for line in (ASR_DIRS[engine] / "raw.jsonl").read_text().splitlines()]
            raw = next(r for r in raw_rows if r["opaque_id"] == row["opaque_id"])
            obs[engine] = {
                "ordinal": raw["ordinal"], "opaque_id": raw["opaque_id"],
                "raw_transcript": raw["raw_transcript"],
                "normalized_transcript": row["normalized_transcript"],
                "full_script_match": row["full_script_match"],
                "word_errors": row["word_errors"], "reference_words": row["reference_words"],
                "raw_log_sha256": self.comparison["sources"][engine]["raw_sha256"],
                "completion_sha256": self.comparison["sources"][engine]["complete_sha256"],
            }
        secondary = (not all(o["full_script_match"] for o in obs.values()) or
                     obs["whisper"]["normalized_transcript"] !=
                     obs["wav2vec2"]["normalized_transcript"])
        self.manifest["blind_order"]["primary"].append(comparisons["whisper"]["opaque_id"])
        if secondary:
            self.manifest["blind_order"]["secondary"].append(comparisons["whisper"]["opaque_id"])
        suffix = f"{case}_{voice}"
        receipt = {
            "schema": "voicechat_wav_qa_receipt_v2", "case_id": case,
            "voice_id": voice, "wav_path": str(wav.resolve()),
            "wav_sha256": digest(wav), "pcm_int16_sha256": hashlib.sha256(pcm).hexdigest(),
            "script": render_row["script"],
            "script_sha256": hashlib.sha256(render_row["script"].encode()).hexdigest(),
            "asr_observations": obs,
            "primary": self._listener("primary", comparisons["whisper"]["opaque_id"],
                                      render_row["script"], case, suffix),
            "secondary_required": secondary,
            "secondary": self._listener("secondary", comparisons["whisper"]["opaque_id"],
                                        render_row["script"], case, suffix) if secondary else None,
            "tertiary_required": False, "tertiary": None,
            "final_eligibility": "eligible", "adjudication_rationale": "synthetic only",
            "qa_completed_at_utc": QA_TIME,
            "external_controller_id": CONTROLLER if case == "C" else "not_applicable",
        }
        receipt_desc = write_record(self.root, f"{suffix}_receipt.json", receipt)
        self.manifest["rows"].append({"case_id": case, "voice_id": voice,
                                      "wav_sha256": digest(wav), "receipt": receipt_desc,
                                      "final_eligibility": "eligible"})
        self.manifest["eligibility"].append({"case_id": case, "voice_id": voice,
                                             "wav_sha256": digest(wav),
                                             "receipt_sha256": receipt_desc["sha256"],
                                             "final_eligibility": "eligible"})
        if case == "C":
            grant = {"grant_id": suffix, "controller_id": CONTROLLER,
                     "case_id": "C", "source_wav_sha256": digest(wav),
                     "qa_receipt_sha256": receipt_desc["sha256"],
                     "authorized_date": "2026-11-19", "authorized_event_ids": ["DR19"]}
            grant["signature_hmac_sha256"] = hmac.new(KEY, grant_signing_bytes(grant),
                                                       hashlib.sha256).hexdigest()
            grant_desc = write_record(self.root, f"{suffix}_grant.json", grant)
            self.manifest["c_grants"].append({"voice_id": voice, "receipt_sha256": receipt_desc["sha256"],
                                              "grant": grant_desc})

    def save(self) -> None:
        self.manifest_path.write_bytes(canonical_bytes(self.manifest))
        ledger = {"schema": "voicechat_preinference_ledger_v2",
                  "qa_manifest_sha256": digest(self.manifest_path),
                  "eligibility": self.manifest["eligibility"],
                  "external_anchor_audit_id": "synthetic-anchor",
                  "anchored_utc": ANCHOR_TIME}
        self.ledger_path.write_bytes(canonical_bytes(ledger))

    def validate(self):
        return check_audio_qa_v2(self.manifest_path, self.ledger_path,
                                 controller_key=KEY, trusted_controller_id=CONTROLLER)

    def test_complete_synthetic_cohort_validates_without_asr_exact_requirement(self):
        result = self.validate()
        self.assertEqual(result["primary_reviews"], 12)
        self.assertEqual(result["secondary_reviews"], 11)
        self.assertEqual(result["asr_exact_wav2vec2"], 3)

    def test_missing_secondary_fails(self):
        target = self.manifest["rows"][0]
        path = self.root / target["receipt"]["path"]
        receipt = json.loads(path.read_bytes())
        self.assertTrue(receipt["secondary_required"])
        receipt["secondary"] = None
        path.write_bytes(canonical_bytes(receipt))
        target["receipt"]["sha256"] = digest(path)
        self.manifest["eligibility"][0]["receipt_sha256"] = digest(path)
        self.save()
        with self.assertRaises((ValueError, TypeError)):
            self.validate()

    def test_one_exclusion_stops_full_matrix(self):
        self.manifest["eligibility"][0]["final_eligibility"] = "excluded"
        self.save()
        with self.assertRaisesRegex(ValueError, "eligibility"):
            self.validate()

    def test_grant_wrong_receipt_fails(self):
        grant_desc = self.manifest["c_grants"][0]["grant"]
        path = self.root / grant_desc["path"]
        grant = json.loads(path.read_bytes())
        grant["qa_receipt_sha256"] = "0" * 64
        path.write_bytes(canonical_bytes(grant))
        grant_desc["sha256"] = digest(path)
        self.save()
        with self.assertRaisesRegex(ValueError, "C grant"):
            self.validate()

    def test_v2_adapter_uses_full_cohort_gate_before_first_pass(self):
        row = self.manifest["rows"][0]
        wav = next(RENDER.parent / r["final_path"] for r in self.render["rows"]
                   if (r["case_id"], r["voice_id"]) == (row["case_id"], row["voice_id"]))
        runner = FakeRunner([], [])
        kwargs = {"case_id": row["case_id"], "wav_path": wav,
                  "qa_receipt_path": self.root / row["receipt"]["path"],
                  "qa_manifest_path": self.manifest_path, "qa_ledger_path": self.ledger_path,
                  "controller_key": KEY, "trusted_controller_id": CONTROLLER,
                  "input_signal": "fake-model-signal", "input_signal_lens": [10],
                  "prompt_tokens": [1], "prompt_token_lens": [1],
                  "inference_runner": runner,
                  "tensor_builder": lambda values, device: values}
        run_two_pass(FakeModel(), **kwargs)
        self.assertEqual(len(runner.calls), 2)
        runner.calls.clear()
        self.manifest["eligibility"][0]["final_eligibility"] = "excluded"
        self.save()
        with self.assertRaisesRegex(ValueError, "eligibility"):
            run_two_pass(FakeModel(), **kwargs)
        self.assertEqual(runner.calls, [])

    def test_blind_collector_saves_free_transcript_before_revealing_script(self):
        bundle = self.root / "bundle"
        records = self.root / "records"
        prepare(bundle, seed="synthetic-test-seed")
        opaque = next(r["opaque_id"] for r in self.comparison["rows"]["whisper"]
                      if r["case_id"] == "A" and r["voice_id"] == "apple_samantha_en_us")
        script = CASES["A"]["script"]
        answers = iter(("", script, "yes", "high", "", "yes", "2026-11-19",
                        "list_calendar_events", "none", "yes", "eligible", ""))
        prompts = []

        def answer(prompt):
            prompts.append(prompt)
            if "After comparison" in prompt:
                self.assertTrue((records / f"reviewer_one__{opaque}__blind.json").is_file())
            return next(answers)

        with patch("builtins.input", answer), patch("sys.stdin.isatty", return_value=True):
            output = listen(bundle, records, reviewer_id="reviewer_one", opaque_ids=[opaque])
        blind = json.loads((records / f"reviewer_one__{opaque}__blind.json").read_bytes())
        aware = json.loads((records / f"reviewer_one__{opaque}__script_aware.json").read_bytes())
        self.assertEqual(blind["verbatim_free_transcript"], script)
        self.assertNotIn("case_id", blind)
        self.assertEqual(aware["blind_record_sha256"], output[0]["blind"]["sha256"])


if __name__ == "__main__":
    unittest.main()
