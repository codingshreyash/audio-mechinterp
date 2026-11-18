"""CPU-only fake gates for the no-model VoiceChat parity source plan."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import live_parity_source_plan_v1 as plan


class ResponseRuleTests(unittest.TestCase):
    def test_no_call_has_exact_frozen_bytes_and_zero_injection(self):
        calls = [{"decode_audio": False}, {"decode_audio": True}]
        data, no_call = plan.fixed_response_bytes(
            {"external_tool_execution": False, "replay": None}, calls)
        self.assertTrue(no_call)
        self.assertEqual(data, plan.NO_CALL_BYTES)
        self.assertEqual(hashlib.sha256(data).hexdigest(), plan.NO_CALL_SHA256)
        calls[1]["function_calls"] = [1]
        with self.assertRaisesRegex(ValueError, "No-call adapter injected"):
            plan.fixed_response_bytes(
                {"external_tool_execution": False, "replay": None}, calls)

    def test_parsed_response_keeps_insertion_order_and_requires_full_injection(self):
        calls = [{"decode_audio": False},
                 {"decode_audio": True, **{field: [1] for field in plan.INJECTION_FIELDS}}]
        trace = {"external_tool_execution": False,
                 "replay": {"response_selector": {
                     "tool_name": "list_calendar_events",
                     "response": {"status": "ok", "events": [], "executed": False}}}}
        data, no_call = plan.fixed_response_bytes(trace, calls)
        self.assertFalse(no_call)
        self.assertEqual(data, b'{"tool_name": "list_calendar_events", "response": '
                               b'{"status": "ok", "events": [], "executed": false}}\n')
        del calls[1]["function_response_steps"]
        with self.assertRaisesRegex(ValueError, "complete second-pass injection"):
            plan.fixed_response_bytes(trace, calls)

    def test_rejects_untrusted_trace_and_wrong_pass_order(self):
        calls = [{"decode_audio": True}, {"decode_audio": False}]
        with self.assertRaisesRegex(ValueError, "capture or pass order"):
            plan.fixed_response_bytes({"external_tool_execution": False, "replay": None}, calls)
        calls = [{"decode_audio": False}, {"decode_audio": True}]
        with self.assertRaisesRegex(ValueError, "capture or pass order"):
            plan.fixed_response_bytes({"external_tool_execution": True, "replay": None}, calls)
        with self.assertRaisesRegex(ValueError, "capture or pass order"):
            plan.fixed_response_bytes({"external_tool_execution": False}, calls)


class SourcePlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.nemo = self.root / "Speech"
        for relative in (plan.SAMPLE_SCRIPT, plan.SAMPLE_WAV, plan.SAMPLE_RESPONSE,
                         plan.TEMPLATE, plan.HELPER):
            file = self.nemo / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(relative.encode())
        self.wav = self.root / "B_piper.wav"
        self.wav.write_bytes(b"fake-source-only-wav")
        self.receipt = self.root / "receipt.json"
        self.receipt.write_bytes(b'{"test_only":true}\n')
        self.manifest = self.root / "manifest.json"
        self.manifest.write_bytes(b'{"test_only":true}\n')
        self.ledger = self.root / "ledger.json"
        self.ledger.write_text('{"external_anchor_audit_id":"fake-test-anchor"}\n')
        self.render = self.root / "render.json"
        self.render.write_text(json.dumps({"rows": [{
            "case_id": plan.CASE_ID, "voice_id": plan.VOICE_ID,
            "final_sha256": plan.sha256(self.wav),
        }]}))
        self.kwargs = {
            "nemo_dir": self.nemo, "checkpoint_dir": self.root / "checkpoint",
            "nano_home": self.root / "nano", "qa_manifest": self.manifest,
            "qa_ledger": self.ledger, "qa_receipt": self.receipt,
            "wav_path": self.wav, "controller_id": "fake-controller",
            "gpu_policy": "h100", "seed": 1729,
        }
        self.key_patch = patch.dict(os.environ, {"VOICECHAT_CONTROLLER_KEY_HEX": "11" * 32})
        self.key_patch.start()
        self.addCleanup(self.key_patch.stop)
        self.render_patch = patch.object(plan, "RENDER", self.render)
        self.render_patch.start()
        self.addCleanup(self.render_patch.stop)

    def _fake_qa(self, *_args, **_kwargs):
        return {"human_eligible": 12, "qa_manifest_sha256": "a" * 64,
                "qa_ledger_sha256": "b" * 64}

    def _fake_receipt(self, *_args, **_kwargs):
        raw = self.receipt.read_bytes()
        return {"voice_id": plan.VOICE_ID}, raw, hashlib.sha256(raw).hexdigest()

    def _fake_preload(self, **_kwargs):
        return {"reported": {"status": "local_preload_preflight_passed_no_model_or_network"},
                "stdout_sha256": "c" * 64}

    def _build(self, **changes):
        kwargs = {**self.kwargs, "qa_checker": self._fake_qa,
                  "preload": self._fake_preload, "source_gate": lambda: None,
                  **changes}
        with patch.object(plan, "read_bound_qa_receipt", self._fake_receipt):
            return plan.build_source_plan(**kwargs)

    def test_success_emits_plan_and_never_models(self):
        result = self._build()
        self.assertEqual(result["status"], "synthetic_test_only_no_model_work")
        self.assertTrue(result["synthetic_test_injection"])
        self.assertEqual(result["arm_order"], ["official_sample", *plan.ARMS])
        self.assertEqual(result["selected"]["wav_sha256"], plan.sha256(self.wav))
        self.assertEqual(result["response_rule"]["no_call_placeholder_sha256"],
                         plan.NO_CALL_SHA256)
        self.assertFalse(result["external_tool_execution"])
        self.assertIn("four_fresh_processes_with_identical_model_rng_state",
                      result["unimplemented_live_gates"])
        self.assertEqual(json.loads(plan.canonical_bytes(result))["schema"],
                         "voicechat_live_parity_source_plan_v1")

    def test_qa_failure_precedes_preload(self):
        calls = []
        def fail_qa(*_args, **_kwargs):
            calls.append("qa")
            raise ValueError("human QA absent")
        def preload(**_kwargs):
            calls.append("preload")
            return self._fake_preload()
        with self.assertRaisesRegex(ValueError, "human QA absent"):
            self._build(qa_checker=fail_qa, preload=preload)
        self.assertEqual(calls, ["qa"])

    def test_wrong_selected_wav_precedes_preload(self):
        other = self.root / "other.wav"
        other.write_bytes(b"other-fake-wav")
        called = []
        def preload(**_kwargs):
            called.append(True)
            return self._fake_preload()
        with self.assertRaisesRegex(ValueError, "Fixed case-B Piper WAV"):
            self._build(wav_path=other, preload=preload)
        self.assertEqual(called, [])

    def test_missing_sample_source_holds(self):
        (self.nemo / plan.SAMPLE_WAV).unlink()
        with self.assertRaisesRegex(ValueError, "sample/source file absent"):
            self._build()

    def test_bad_preload_status_holds(self):
        with self.assertRaisesRegex(ValueError, "Pinned preload receipt differs"):
            self._build(preload=lambda **_kwargs: {"reported": {"status": "failed"}})

    def test_source_gate_detects_changed_protocol(self):
        fake = self.root / "protocol.md"
        fake.write_text("changed protocol")
        with patch.object(plan, "PROTOCOL", fake):
            with self.assertRaisesRegex(ValueError, "Reviewed VoiceChat parity protocol"):
                plan._source_gate()


if __name__ == "__main__":
    unittest.main()
