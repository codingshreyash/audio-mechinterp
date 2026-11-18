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
                         plan.TEMPLATE, plan.HELPER, *plan.OTHER_PINNED_SOURCE):
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
            "final_path": self.wav.name,
            "final_sha256": plan.sha256(self.wav),
        }]}))
        (self.root / self.wav.name).write_bytes(self.wav.read_bytes())
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_bytes(b"fake-config")
        (checkpoint / "model.safetensors").write_bytes(b"fake-weights")
        nano_ref = (self.root / "nano" / "hub" /
                    "models--nvidia--NVIDIA-Nemotron-Nano-9B-v2" / "refs" / "main")
        nano_ref.parent.mkdir(parents=True)
        nano_ref.write_bytes(b"fake-nano-ref")
        nano_snapshot = nano_ref.parent.parent / "snapshots" / \
            "6533e8de2c68e4536bf7c411d7a3ce5734111476"
        nano_snapshot.mkdir(parents=True)
        for name in plan.NANO_METADATA_FILES:
            (nano_snapshot / name).write_bytes(name.encode())
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
        return {"human_eligible": 12, "qa_manifest_sha256": plan.sha256(self.manifest),
                "qa_ledger_sha256": plan.sha256(self.ledger)}

    def _fake_receipt(self, *_args, **_kwargs):
        raw = self.receipt.read_bytes()
        return {"voice_id": plan.VOICE_ID}, raw, hashlib.sha256(raw).hexdigest()

    def _fake_phase(self, phase):
        report = {
            "status": f"local_{phase}_preflight_passed_no_model_or_network",
            "nemo_commit": "fake-nemo", "voicechat_commit": "fake-voicechat",
            "config_sha256": plan.sha256(self.root / "checkpoint" / "config.json"),
            "weight_sha256": plan.sha256(self.root / "checkpoint" / "model.safetensors"),
            "qa_manifest_sha256": plan.sha256(self.manifest),
            "qa_ledger_sha256": plan.sha256(self.ledger),
            "c_grant_sha256_by_voice": {"fake": "fake"},
            "nano_repo_id": "fake-nano", "nano_revision": "fake-revision",
            "nano_hf_home": str(self.root / "nano"), "gpu_policy": "h100",
            "gpu": "fake-h100", "source_files_verified": 5,
            "nano_files_verified": 6,
        }
        if phase == "ready":
            report["nano_offline_resolver"] = {"status": "fake-offline-resolver"}
        else:
            report["nano_preload_environment"] = "offline_pinned_cache_no_bytecode"
        instant = "2026-11-18T00:00:00+00:00" if phase == "ready" else \
                  "2026-11-18T00:00:01+00:00"
        return {"phase": phase, "started_utc": instant, "finished_utc": instant,
                "reported": report, "stdout_sha256": "c" * 64}

    def _fake_ready(self, **_kwargs):
        return self._fake_phase("ready")

    def _fake_preload(self, **_kwargs):
        return self._fake_phase("preload")

    def _build(self, **changes):
        kwargs = {**self.kwargs, "qa_checker": self._fake_qa,
                  "ready": self._fake_ready, "preload": self._fake_preload,
                  "source_gate": lambda: None,
                  **changes}
        with patch.object(plan, "read_bound_qa_receipt", self._fake_receipt):
            return plan.build_source_plan(**kwargs)

    def test_success_emits_plan_and_never_models(self):
        result = self._build()
        self.assertEqual(result["status"], "synthetic_test_only_no_model_work")
        self.assertTrue(result["synthetic_test_injection"])
        self.assertEqual(result["arm_order"], ["official_sample", *plan.ARMS])
        self.assertEqual(result["selected"]["wav_sha256"], plan.sha256(self.wav))
        self.assertEqual(result["ready"]["phase"], "ready")
        self.assertEqual(result["preload"]["phase"], "preload")
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

    def test_ready_must_precede_preload_and_bind_identity(self):
        calls = []
        def ready(**_kwargs):
            calls.append("ready")
            return self._fake_ready()
        def preload(**_kwargs):
            calls.append("preload")
            return self._fake_preload()
        self._build(ready=ready, preload=preload)
        self.assertEqual(calls, ["ready", "preload"])
        def wrong_identity(**_kwargs):
            result = self._fake_preload()
            result["reported"]["nemo_commit"] = "changed"
            return result
        with self.assertRaisesRegex(ValueError, "staged identity differs"):
            self._build(preload=wrong_identity)

    def test_changed_input_during_ready_prevents_preload(self):
        calls = []
        def changed_ready(**_kwargs):
            self.receipt.write_bytes(b"mutated-after-qa")
            return self._fake_ready()
        def preload(**_kwargs):
            calls.append("preload")
            return self._fake_preload()
        with self.assertRaisesRegex(ValueError, "changed during ready"):
            self._build(ready=changed_ready, preload=preload)
        self.assertEqual(calls, [])

    def test_changed_checkpoint_after_preload_holds(self):
        def changed_preload(**_kwargs):
            (self.root / "checkpoint" / "config.json").write_bytes(b"mutated-config")
            return self._fake_preload()
        with self.assertRaisesRegex(ValueError, "changed during preload"):
            self._build(preload=changed_preload)

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
