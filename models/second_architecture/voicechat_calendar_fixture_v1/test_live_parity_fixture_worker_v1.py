"""CPU-only contract tests for the gated fixture worker; no model or QA records."""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from .live_parity_fixture_worker_v1 import (
    run_fixture_arm, run_production_adapter_once,
)
from .live_parity_saved_capture_v1 import load_capture
from .live_parity_source_plan_v1 import CASE_ID, NO_CALL_BYTES, VOICE_ID, canonical_bytes
from .live_parity_production_admission_v1 import child_public_environment


class FakeRecorder:
    def __init__(self, model: object):
        self.model = model
        self.passes: dict[str, object] = {}

    def wrap_runner(self, _base):
        def run(model, **kwargs):
            assert model is self.model
            label = "pass1" if not self.passes else "pass2"
            result = {"tokens_text": torch.tensor([[1, 2]]),
                      "audio": torch.tensor([0.25]), "text": label}
            self.passes[label] = {"pass_label": label, "result_channels_unspliced": result}
            return result
        return run

    def bind_adapter_trace(self, trace):
        assert set(self.passes) == {"pass1", "pass2"}
        assert trace["external_tool_execution"] is False


class FixtureWorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.arm_dir = self.root / "arm"
        self.arm_dir.mkdir()
        self.events: list[str] = []
        self.model = object()
        self.official = SimpleNamespace(
            load_wav_16k_mono=self._load,
            render_fc_system_prompt=self._render,
            encode_system_prompt=self._encode,
        )
        paths = {}
        for name, data in {
            "wav_path": b"fake wav, never modeled",
            "qa_receipt_path": b"no human receipt, fake adapter never reads it",
            "qa_manifest_path": b"no manifest, fake adapter never reads it",
            "qa_ledger_path": b"no ledger, fake adapter never reads it",
            "template_path": b"fake template",
            "system_message_path": b"System",
            "tools_path": b"[]",
            "response_path": NO_CALL_BYTES,
        }.items():
            path = self.root / name
            path.write_bytes(data)
            paths[name] = str(path)
        self.request = {
            "arm": "adapter", "case_id": CASE_ID, "voice_id": VOICE_ID,
            "seed": 17, "capture_byte_cap": 100000,
            "input_hashes": {key: "0" * 64 for key in
                             ("selected_wav", "qa_receipt", "qa_manifest",
                              "qa_ledger")},
            "response_sha256": hashlib.sha256(NO_CALL_BYTES).hexdigest(),
            "checkpoint_dir": str(self.root / "absent-checkpoint"),
            "device": "cpu", **paths,
        }

    def _gate(self, request):
        self.events.append("gate")
        return {"status": "synthetic_test_only_no_model_work",
                "input_hashes": request["input_hashes"], "seed": request["seed"]}

    def _official(self):
        self.events.append("official")
        return self.official

    def _qa(self, _request):
        self.events.append("qa")
        return {name: "0" * 64 for name in
                ("wav_sha256", "qa_receipt_sha256", "qa_manifest_sha256",
                 "qa_ledger_sha256")}

    def _model(self, official, checkpoint_dir, device):
        assert official is self.official
        assert checkpoint_dir == Path(self.request["checkpoint_dir"])
        assert device == "cpu"
        self.events.append("model")
        return self.model

    def _load(self, *args, **kwargs):
        self.events.append("load_wav")
        return (torch.tensor([0.0]), torch.tensor([[0.0]]), torch.tensor([1]))

    def _render(self, *args):
        self.events.append("render")
        return "exact fake prompt"

    def _encode(self, *args, **kwargs):
        self.events.append("encode")
        return torch.tensor([[1]]), torch.tensor([1])

    def _adapter(self, model, **kwargs):
        self.events.append("adapter")
        assert model is self.model
        child = kwargs["output_dir"]
        assert not child.exists()
        child.mkdir()
        observer = kwargs["inference_runner"]
        first = observer(model, decode_audio=False, input_signal=torch.tensor([1]))
        second = observer(model, decode_audio=True, input_signal=torch.tensor([1]))
        return {"pass1": {"raw_result": first},
                "pass2": {"raw_result": second},
                "replay": None, "external_tool_execution": False}

    def _stock(self, official, model, **kwargs):
        self.events.append("stock")
        assert official is self.official and model is self.model
        assert kwargs["api_response"]["response"]["executed"] is False
        return {"pass1": {"kwargs": {"decode_audio": False},
                          "result": {"tokens_text": torch.tensor([[1]])}},
                "pass2": {"kwargs": {"decode_audio": True},
                          "result": {"audio": torch.tensor([0.5])}},
                "tts_wrapped": kwargs["capture_tts"]}

    def _run(self):
        return run_fixture_arm(
            self.request, arm_dir=self.arm_dir, admission_gate=self._gate,
            qa_precheck=self._qa,
            official_factory=self._official, model_factory=self._model,
            recorder_factory=FakeRecorder, adapter_run=self._adapter,
            stock_capture=self._stock)

    def test_default_gate_blocks_before_any_factory_or_output(self):
        with self.assertRaisesRegex(RuntimeError, "admission not installed"):
            run_fixture_arm(self.request, arm_dir=self.arm_dir,
                            official_factory=self._official,
                            model_factory=self._model)
        self.assertEqual(self.events, [])
        self.assertEqual(list(self.arm_dir.iterdir()), [])

    def test_adapter_observer_captures_exact_two_passes_in_fresh_child(self):
        receipt = self._run()
        self.assertEqual(self.events, ["gate", "qa", "official", "model", "adapter"])
        self.assertTrue((self.arm_dir / "adapter_high_level").is_dir())
        self.assertEqual(receipt["status"], "synthetic_test_only_no_model_work")
        saved = load_capture(self.arm_dir / "raw_capture", byte_cap=100000,
                             expected_manifest_sha256=receipt["capture"]["manifest_sha256"],
                             trusted_root=self.arm_dir)
        self.assertEqual([x["decode_audio"] for x in saved["calls"]], [False, True])
        self.assertEqual(set(saved["passive_frames"]), {"pass1", "pass2"})
        self.assertTrue(torch.equal(saved["trace"]["pass2"]["raw_result"]["audio"],
                                    torch.tensor([0.25])))

    def test_stock_arms_use_inert_exact_response_and_tts_flag(self):
        for arm, wrapped in (("stock_a", True), ("stock_b", True),
                             ("stock_no_tts", False)):
            with self.subTest(arm=arm):
                self.arm_dir = self.root / arm
                self.arm_dir.mkdir()
                self.request["arm"] = arm
                self.events.clear()
                receipt = self._run()
                self.assertEqual(self.events[:4], ["gate", "qa", "official", "model"])
                self.assertEqual(self.events[4:], ["load_wav", "render", "encode", "stock"])
                saved = load_capture(self.arm_dir / "raw_capture", byte_cap=100000,
                                     expected_manifest_sha256=receipt["capture"]["manifest_sha256"],
                                     trusted_root=self.arm_dir)
                self.assertEqual(saved["stock"]["tts_wrapped"], wrapped)
                self.assertEqual(saved["system_message_exact"], b"System")

    def test_rejects_non_inert_stock_response_before_load(self):
        self.request["arm"] = "stock_a"
        Path(self.request["response_path"]).write_bytes(json.dumps({
            "tool_name": "cancel_calendar_events",
            "response": {"executed": True}}).encode() + b"\n")
        with self.assertRaisesRegex(ValueError, "not inert"):
            self._run()
        self.assertEqual(self.events, ["gate", "qa"])

    def test_rejects_unbound_gate_and_reused_arm(self):
        def wrong_gate(_request):
            self.events.append("gate")
            return {"status": "production_ready", "seed": 17,
                    "input_hashes": self.request["input_hashes"]}
        with self.assertRaisesRegex(ValueError, "admission remains closed"):
            run_fixture_arm(self.request, arm_dir=self.arm_dir,
                            admission_gate=wrong_gate,
                            qa_precheck=self._qa,
                            official_factory=self._official,
                            model_factory=self._model)
        self.assertEqual(self.events, ["gate"])
        self.events.clear()
        self._run()
        with self.assertRaisesRegex(ValueError, "exclusive prepared root"):
            self._run()

    def test_qa_precheck_failure_blocks_official_and_model_factories(self):
        def denied(_request):
            self.events.append("qa")
            raise ValueError("no independently reviewed 12-WAV cohort")
        with self.assertRaisesRegex(ValueError, "12-WAV cohort"):
            run_fixture_arm(self.request, arm_dir=self.arm_dir,
                            admission_gate=self._gate, qa_precheck=denied,
                            official_factory=self._official,
                            model_factory=self._model)
        self.assertEqual(self.events, ["gate", "qa"])
        self.assertEqual(list(self.arm_dir.iterdir()), [])

    def test_qa_binding_and_response_hash_fail_before_model_factory(self):
        self.request["input_hashes"]["selected_wav"] = "1" * 64
        with self.assertRaisesRegex(ValueError, "source-plan bound hashes"):
            self._run()
        self.assertEqual(self.events, ["gate", "qa"])
        self.events.clear()
        self.request["input_hashes"]["selected_wav"] = "0" * 64
        self.request["arm"] = "stock_a"
        self.request["response_sha256"] = "1" * 64
        with self.assertRaisesRegex(ValueError, "expected response hash"):
            self._run()
        self.assertEqual(self.events, ["gate", "qa"])

    def test_production_wrapper_checks_admission_before_one_arm_and_posthash(self):
        plan_path = self.root / "source_plan.json"
        plan = {"seed": 17, "input_hashes": self.request["input_hashes"],
                "controller_id": "synthetic-test"}
        plan_path.write_bytes(canonical_bytes(plan))
        output = self.root / "run_root"
        output.mkdir()
        paths = {name: self.root / name for name in
                 ("nemo", "checkpoint", "nano", "qa_manifest", "qa_ledger",
                  "qa_receipt", "wav")}
        paths["output"] = output
        admission = {"status": "fixture_admitted_for_fresh_process_source_only",
                     "seed": 17, "input_hashes": plan["input_hashes"],
                     "source_plan_sha256": "a" * 64, "freeze_sha256": "b" * 64,
                     "official_sample_receipt_sha256": "d" * 64,
                     "source_sha256": {"synthetic": "c" * 64},
                     "child_public_environment": child_public_environment(),
                     "caps": {"arm_output_bytes": 100000}}
        def admit(**_kwargs):
            self.events.append("admit")
            self.assertIn("VOICECHAT_CONTROLLER_KEY_HEX", os.environ)
            return admission
        def arm_run(request, *, arm_dir, admission_gate):
            self.events.append("arm")
            self.assertNotIn("VOICECHAT_CONTROLLER_KEY_HEX", os.environ)
            self.assertNotIn("VOICECHAT_FIXTURE_FREEZE_REVIEW_KEY_HEX", os.environ)
            self.assertEqual(request["arm"], "adapter")
            self.assertEqual(request["source_plan_sha256"], "a" * 64)
            self.assertEqual(admission_gate(request), admission)
            self.assertTrue(arm_dir.is_dir())
            return {"status": admission["status"]}
        def rehash(_plan, _paths):
            self.events.append("rehash")
            return plan["input_hashes"]
        with patch.dict(os.environ, {"VOICECHAT_CONTROLLER_KEY_HEX": "c" * 64,
                                  "VOICECHAT_FIXTURE_FREEZE_REVIEW_KEY_HEX": "k" * 64}):
            with patch("models.second_architecture.voicechat_calendar_fixture_v1.live_parity_fixture_worker_v1.resource.setrlimit"):
                result = run_production_adapter_once(
                    plan_path=plan_path, freeze_path=self.root / "freeze",
                    review_path=self.root / "review", paths=paths,
                    freeze_review_key=b"k" * 32, controller_key=b"c" * 32,
                    admit=admit, sample_verify=lambda _admission: "d" * 64,
                    arm_run=arm_run, rehash=rehash,
                    source_digest=lambda: admission["source_sha256"])
        self.assertEqual(result["status"], admission["status"])
        self.assertEqual(self.events, ["admit", "arm", "rehash"])

    def test_production_wrapper_scrubs_keys_even_if_admission_fails(self):
        def fail(**_kwargs):
            self.assertIn("VOICECHAT_CONTROLLER_KEY_HEX", os.environ)
            raise ValueError("synthetic admission failure")
        with patch.dict(os.environ, {"VOICECHAT_CONTROLLER_KEY_HEX": "c" * 64,
                                  "VOICECHAT_FIXTURE_FREEZE_REVIEW_KEY_HEX": "k" * 64}):
            with self.assertRaisesRegex(ValueError, "synthetic admission failure"):
                run_production_adapter_once(
                    plan_path=self.root / "absent-plan",
                    freeze_path=self.root / "absent-freeze",
                    review_path=self.root / "absent-review", paths={},
                    freeze_review_key=b"k" * 32, controller_key=b"c" * 32,
                    admit=fail)
            self.assertNotIn("VOICECHAT_CONTROLLER_KEY_HEX", os.environ)
            self.assertNotIn("VOICECHAT_FIXTURE_FREEZE_REVIEW_KEY_HEX", os.environ)


if __name__ == "__main__":
    unittest.main()
