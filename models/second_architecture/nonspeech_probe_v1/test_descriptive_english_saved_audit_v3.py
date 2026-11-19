"""No-weight fake-token tests for checkpoint-aware saved Whisper EOS audit."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from . import descriptive_english_saved_audit_v3 as audit
from . import descriptive_remaining_audit_v2 as old_audit
from . import descriptive_worker_v1 as scoring


class FakeTokenizer:
    def __init__(self, eos: int):
        self.eos_token_id = eos
        self.pad_token_id = eos

    def decode(self, _ids, *, skip_special_tokens: bool) -> str:
        return "hello" if skip_special_tokens else "<|sot|>hello<|eot|>"


def fake_output(model: str) -> dict:
    contract = audit.whisper_contract(model)
    multilingual = model == "whisper_multilingual"
    prefix = ([contract["decoder_start_token_id"], 50263,
               contract["task_to_id"]["transcribe"],
               contract["no_timestamps_token_id"]] if multilingual else
              [contract["decoder_start_token_id"],
               contract["no_timestamps_token_id"]])
    generated = [345, contract["eos_token_id"]]
    logits = []
    states = []
    for index, token in enumerate(generated):
        vector = torch.full((1, contract["vocab_size"]), -10.0, dtype=torch.float32)
        vector[0, token] = 5.0
        logits.append(vector)
        states.append((torch.ones((1, len(prefix) if index == 0 else 1,
                                   contract["d_model"]), dtype=torch.float32),))
    result = SimpleNamespace(sequences=torch.tensor([prefix + generated]),
                             scores=tuple(logits), logits=tuple(logits),
                             decoder_hidden_states=tuple(states))
    detection = None
    if multilingual:
        detection = {"candidate_count": len(contract["lang_to_id"]),
                     "raw_logits_sha256": "0" * 64,
                     "chosen_language_ids": [50263],
                     "language_logits": {name: (10.0 if token == 50263 else 0.0)
                                         for name, token in contract["lang_to_id"].items()}}
    return scoring.score_whisper(result, FakeTokenizer(contract["eos_token_id"]),
                                 detection)


class CheckpointAwareAuditTests(unittest.TestCase):
    def test_pinned_english_and_multilingual_eos_prefix_geometry(self) -> None:
        english = audit.whisper_contract("whisper_english")
        multilingual = audit.whisper_contract("whisper_multilingual")
        self.assertEqual((english["eos_token_id"], english["pad_token_id"]),
                         (50256, 50256))
        self.assertEqual((multilingual["eos_token_id"], multilingual["pad_token_id"]),
                         (50257, 50257))
        self.assertEqual(english["vocab_size"], 51864)
        self.assertEqual(multilingual["vocab_size"], 51865)
        self.assertEqual(english["no_timestamps_token_id"], 50362)
        self.assertEqual(multilingual["no_timestamps_token_id"], 50363)

    def test_english_ended_50256_passes_v3_and_fails_old_audit(self) -> None:
        output = fake_output("whisper_english")
        self.assertEqual(output["sequence_ids"][:4], [50257, 50362, 345, 50256])
        self.assertTrue(output["eos_seen"])
        audit.check_whisper(output, "whisper_english")
        with self.assertRaisesRegex(ValueError, "EOS/censor accounting invalid"):
            old_audit.check_whisper(output, False)
        tampered = copy.deepcopy(output)
        tampered["eos_seen"] = False
        with self.assertRaisesRegex(ValueError, "EOS/censor accounting invalid"):
            audit.check_whisper(tampered, "whisper_english")
        tampered = copy.deepcopy(output)
        tampered["prefix_ids"][1] = 50363
        tampered["sequence_ids"][1] = 50363
        with self.assertRaisesRegex(ValueError, "decoder prefix invalid"):
            audit.check_whisper(tampered, "whisper_english")

    def test_multilingual_50257_control_and_language_choice(self) -> None:
        output = fake_output("whisper_multilingual")
        audit.check_whisper(output, "whisper_multilingual")
        tampered = copy.deepcopy(output)
        tampered["language_detection"]["language_logits"]["<|en|>"] = 11.0
        with self.assertRaisesRegex(ValueError, "not the argmax"):
            audit.check_whisper(tampered, "whisper_multilingual")

    def test_no_go_blocks_saved_attempt_read(self) -> None:
        with mock.patch.object(audit, "require_v3_go",
                               side_effect=ValueError("fake V3 GO absent")):
            with self.assertRaisesRegex(ValueError, "fake V3 GO absent"):
                audit.audit_attempt()

    def test_preflight_hashes_immutable_attempt_without_output_parse(self) -> None:
        receipt = audit.preflight_receipt()
        self.assertEqual(receipt["model"], "whisper_english")
        self.assertEqual(receipt["attempt_files_sha256"], audit.ATTEMPT_SHA256)
        self.assertEqual(receipt["status"],
                         "source_only_no_model_load_no_output_inspection")
        self.assertEqual(receipt["whisper_contract"]["eos_token_id"], 50256)
        self.assertEqual(receipt["multilingual_control_contract"]["eos_token_id"], 50257)


if __name__ == "__main__":
    unittest.main()
