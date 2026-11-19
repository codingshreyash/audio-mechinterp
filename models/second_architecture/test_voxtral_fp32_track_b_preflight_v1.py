"""CPU/no-model tests for the fail-closed Track B source preflight."""
from __future__ import annotations

import dataclasses
import base64
import gzip
import json
import plistlib
import sys
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import voxtral_fp32_track_b_preflight_v1 as target
from word_disjoint_v2 import candidate_common as common


class FakeTokenizer:
    all_special_ids = [2, 24]

    def __init__(self, ids):
        self.ids = ids

    def apply_chat_template(self, *_args, **_kwargs):
        return {"input_ids": self.ids}


class SourcePreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases, _ = common.fixture_cases()
        cls.manifest = json.loads(target.MANIFEST.read_text())
        cls.inventory = json.loads(target.INVENTORY.read_text())

    def test_frozen_anchors_and_complete_selected_pcm(self):
        self.assertEqual(len(target.verify_static_anchors()), len(target.SOURCE_HASHES))
        selected, receipts = target.selected_cases(self.cases, self.manifest,
                                                   self.inventory, common.load_pcm)
        self.assertEqual(len(selected), 16)
        self.assertEqual(len(receipts), 16)
        self.assertEqual(len({(c.pair_id, c.engine, c.word) for c in selected}), 16)

    def test_tampered_v3_anchor_fails(self):
        changed = dict(target.SOURCE_HASHES)
        changed[target.V3] = "0" * 64
        with patch.object(target, "SOURCE_HASHES", changed), self.assertRaisesRegex(ValueError, "Frozen anchor"):
            target.verify_static_anchors()

    def test_selected_wav_or_inventory_tamper_fails(self):
        changed = [dataclasses.replace(c, wav_sha256="0" * 64) if c.key ==
                   "pair_00_piper_banana" else c for c in self.cases]
        with self.assertRaisesRegex(ValueError, "Selected WAV"):
            target.selected_cases(changed, self.manifest, self.inventory, common.load_pcm)
        inventory = dict(self.inventory)
        inventory["final_wavs"] = [dict(row) for row in self.inventory["final_wavs"]]
        inventory["final_wavs"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Selected WAV"):
            target.selected_cases(self.cases, self.manifest, inventory, common.load_pcm)

    def test_pcm_word_window_is_half_open_and_nonempty(self):
        self.assertEqual(target.projector_rows((19382, 37029), 64398, 375), list(range(15, 29)))
        self.assertEqual(target.projector_rows((10964, 20622), 36606, 375), list(range(9, 16)))
        with self.assertRaisesRegex(ValueError, "Empty"):
            target.projector_rows((0, 1), 1000, 375)
        with self.assertRaisesRegex(ValueError, "geometry"):
            target.projector_rows((10, 1001), 1000, 375)

    def test_contextual_alias_must_extend_exact_prompt(self):
        tokenizer = FakeTokenizer([1, 4, 22])
        self.assertEqual(target.exact_alias_ids(tokenizer, [{"role": "user"}], [1, 4], "word"), [22, 2])
        with self.assertRaisesRegex(ValueError, "exact continuation"):
            target.exact_alias_ids(tokenizer, [{"role": "user"}], [1, 5], "word")
        with self.assertRaisesRegex(ValueError, "special token"):
            target.exact_alias_ids(FakeTokenizer([1, 4, 24]), [{"role": "user"}], [1, 4], "word")

    def test_sequential_and_batched_resource_ledgers_are_distinct(self):
        prepared = {"aliases": {"banana": [{"token_ids_with_eos": [1, 2]}] * 6},
                    "batches": [{"input_ids": [1] * 405, "audio_placeholder_count": 375}]}
        snapshot = {"weight_shards": {"a": {"size_bytes": 4_000_000_000},
                                       "b": {"size_bytes": 4_000_000_000}}}
        with patch.object(target, "fp32_parameter_payload_bytes", return_value=15_000_000_000):
            ledger = target.resource_estimate(prepared, snapshot)
        self.assertEqual(ledger["v5_sequential_b1_sizing"]["batch_size"], 1)
        self.assertEqual(ledger["v3_alias_batch_sizing"]["batch_size"], 6)
        self.assertLess(ledger["v5_sequential_b1_sizing"]["planning_total_bytes"],
                        ledger["v3_alias_batch_sizing"]["planning_total_bytes"])
        self.assertEqual(ledger["v5_sequential_workload"]["max_core_projector_calls"], 1456)
        self.assertFalse(ledger["can_certify_mps_peak_without_model"])
        self.assertGreater(ledger["fp32_parameter_payload_floor_bytes"], 0)
        self.assertLessEqual(ledger["fp32_parameter_payload_floor_bytes"],
                             ledger["fp32_weight_planning_bytes"])

    def test_frozen_v6_zero_geometry_uses_max_selected_frames(self):
        selected, wavs = target.selected_cases(self.cases, self.manifest,
                                              self.inventory, common.load_pcm)
        prepared = target.prepare_processor(selected)
        geometry = target.synthetic_zero_geometry(prepared, wavs)
        self.assertEqual(geometry["first_componentwise_satisfying_frames"], 64398)
        self.assertEqual(geometry["input_features_shape"], [1, 128, 3000])
        self.assertEqual(geometry["audio_placeholder_count"], 375)
        self.assertEqual(geometry["longest_alias"]["token_ids_with_eos"][-1], 2)
        self.assertEqual(geometry["pcm_sha256"], "acab5d38b351aced4029d7df83243229a82ad72008a90f3eb6c28176bbf459e4")

    def test_missing_human_qa_remains_a_hold(self):
        status = target.qa_status(self.cases)
        self.assertFalse(status["passed"])
        self.assertIn("absent", status["reason"])

    def test_host_receipt_never_claims_future_eligibility(self):
        import voxtral_fp32_nonbeaver_linearity_continuation_gate_v1 as gate
        fake = {"collected_utc": "2026-11-19T00:00:00+00:00", "commands": {"power": ["pmset"]},
                "outputs": {"power": "AC Power", "power_log": "safe"}}
        with patch.object(gate, "_host_probe", return_value=fake):
            receipt = target.host_status()
        self.assertTrue(receipt["eligible_at_collection"])
        self.assertTrue(receipt["fresh_host_probe_required_at_model_launch"])
        self.assertEqual(len(receipt["output_sha256"]["power_log"]), 64)
        self.assertEqual(gzip.decompress(base64.b64decode(receipt["power_log_gzip_base64"])), b"safe")

    def test_host_malformed_historical_assertion_preserves_raw_bytes(self):
        import voxtral_fp32_nonbeaver_linearity_continuation_gate_v1 as gate
        raw_log = b"   pid 42(WindowServer): old \xff assertion\n"
        outputs = {"power": b"Now drawing from 'AC Power'",
                   "clamshell": plistlib.dumps([{"AppleClamshellState": False}]),
                   "battery": plistlib.dumps([{"ExternalConnected": True}]),
                   "power_log": raw_log}
        def fake_run(command, **_kwargs):
            name = next(name for name, argv in gate.HOST_COMMANDS.items() if command == argv)
            return SimpleNamespace(stdout=outputs[name])
        bad_decode = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")
        with patch.object(gate, "_host_probe", side_effect=bad_decode), \
             patch.object(target.subprocess, "run", side_effect=fake_run), \
             patch.object(gate, "_power_log_coverage", return_value=None), \
             patch.object(gate, "_power_events", return_value=[]):
            receipt = target.host_status()
        self.assertEqual(gzip.decompress(base64.b64decode(receipt["power_log_gzip_base64"])), raw_log)
        self.assertEqual(receipt["power_log_utf8_replacement_count"], 1)


if __name__ == "__main__":
    unittest.main()
