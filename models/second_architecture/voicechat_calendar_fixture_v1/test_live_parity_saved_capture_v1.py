"""No-model tests of lossless, hash-checked VoiceChat parity captures."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import live_parity_saved_capture_v1 as saved
from .official_adapter_parity import compare_stock_adapter


class SavedCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "capture"

    def _sample(self):
        return {
            "pass1": {"kwargs": {"prompt_tokens": torch.tensor([[1, 2]],
                                                               dtype=torch.int64)},
                      "result": {"tokens_text": torch.tensor([[3, 4]],
                                                               dtype=torch.int64),
                                 "text": ["I will check."], "tokens_len": [2]}},
            "pass2": {"result": {"audio": torch.tensor([[0.1, -0.2, 0.3]],
                                                          dtype=torch.float32),
                                 "audio_len": torch.tensor([3])},
                      "codes": [torch.tensor([[5, 6]], dtype=torch.int64)]},
            "replay": {"response_utf8": b'{"executed": false}',
                       "position": (0, 1), "none": None},
            "numpy_scalar": np.float32(0.25),
            "bfloat": torch.tensor([1.25], dtype=torch.bfloat16),
        }

    def test_roundtrip_preserves_raw_tensor_values_dtypes_shapes_and_bytes(self):
        original = self._sample()
        receipt = saved.save_capture(original, self.out, byte_cap=65536)
        loaded = saved.load_capture(
            self.out, byte_cap=65536,
            expected_manifest_sha256=receipt["manifest_sha256"])
        self.assertEqual(receipt["tensor_count"], 7)
        self.assertEqual(loaded["replay"]["response_utf8"],
                         original["replay"]["response_utf8"])
        self.assertEqual(loaded["replay"]["position"], (0, 1))
        self.assertEqual(loaded["numpy_scalar"].shape, ())
        self.assertEqual(loaded["numpy_scalar"].dtype, np.float32)
        self.assertEqual(loaded["bfloat"].dtype, torch.bfloat16)
        for phase, key in (("pass1", "tokens_text"), ("pass2", "audio")):
            source = original[phase]["result"][key]
            replay = loaded[phase]["result"][key]
            self.assertEqual(replay.dtype, source.dtype)
            self.assertEqual(tuple(replay.shape), tuple(source.shape))
            self.assertTrue(torch.equal(replay, source))
        self.assertEqual(receipt["total_bytes"], sum(
            p.stat().st_size for p in self.out.iterdir()))

    def test_changed_tensor_or_manifest_is_rejected(self):
        receipt = saved.save_capture(self._sample(), self.out, byte_cap=65536)
        tensor = next(self.out.glob("tensor_*.npy"))
        with tensor.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "tensor hash differs"):
            saved.load_capture(self.out, byte_cap=65536,
                               expected_manifest_sha256=receipt["manifest_sha256"])
        tensor.write_bytes(tensor.read_bytes()[:-7])
        (self.out / saved.MANIFEST).write_bytes(b"{}\n")
        with self.assertRaisesRegex(ValueError, "manifest hash differs"):
            saved.load_capture(self.out, byte_cap=65536,
                               expected_manifest_sha256=receipt["manifest_sha256"])

    def test_extra_file_or_symlink_is_rejected(self):
        receipt = saved.save_capture(self._sample(), self.out, byte_cap=65536)
        extra = self.out / "extra.bin"
        extra.write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "artifact set differs"):
            saved.load_capture(self.out, byte_cap=65536,
                               expected_manifest_sha256=receipt["manifest_sha256"])
        extra.unlink()
        extra.symlink_to(self.out / saved.MANIFEST)
        with self.assertRaisesRegex(ValueError, "linked artifact"):
            saved.load_capture(self.out, byte_cap=65536,
                               expected_manifest_sha256=receipt["manifest_sha256"])

    def test_linked_capture_root_and_parent_are_rejected(self):
        receipt = saved.save_capture(self._sample(), self.out, byte_cap=65536,
                                     trusted_root=self.root)
        alias = self.root / "alias"
        alias.symlink_to(self.out, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "directory is a symlink"):
            saved.load_capture(alias, byte_cap=65536,
                               expected_manifest_sha256=receipt["manifest_sha256"])
        actual_parent = self.root / "actual_parent"
        nested = actual_parent / "capture"
        actual_parent.mkdir()
        second = saved.save_capture({"a": np.array([1, 2], dtype=np.int16)},
                                    nested, byte_cap=4096,
                                    trusted_root=self.root)
        parent_alias = self.root / "parent_alias"
        parent_alias.symlink_to(actual_parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "linked ancestor"):
            saved.load_capture(parent_alias / "capture", byte_cap=4096,
                               expected_manifest_sha256=second["manifest_sha256"],
                               trusted_root=self.root)

    def test_unsupported_object_and_bounded_write_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "type unsupported"):
            saved.save_capture({"bad": object()}, self.out, byte_cap=65536)
        self.assertTrue(self.out.is_dir())
        with self.assertRaises(FileExistsError):
            saved.save_capture({}, self.out, byte_cap=65536)
        with self.assertRaisesRegex(ValueError, "before write"):
            saved.save_capture({"large": torch.zeros(9000, dtype=torch.uint8)},
                               self.root / "large", byte_cap=8192)

    def test_oversized_artifact_is_rejected_before_open(self):
        receipt = saved.save_capture(self._sample(), self.out, byte_cap=65536)
        tensor = next(self.out.glob("tensor_*.npy"))
        tensor.write_bytes(b"x" * 100000)
        original = Path.open
        def guard(path, mode="r", *args, **kwargs):
            if path == tensor and mode == "rb":
                raise AssertionError("Oversized tensor read")
            return original(path, mode, *args, **kwargs)
        with patch.object(Path, "open", guard):
            with self.assertRaisesRegex(ValueError, "Capture exceeds byte cap"):
                saved.load_capture(self.out, byte_cap=65536,
                                   expected_manifest_sha256=receipt["manifest_sha256"])

    def test_comparator_consumes_reloaded_raw_capture_not_memory_object(self):
        common = {"input_signal": torch.tensor([[0.1, 0.2]]),
                  "input_signal_lens": torch.tensor([2]),
                  "prompt_tokens": torch.tensor([[1, 2]], dtype=torch.int64),
                  "prompt_token_lens": torch.tensor([2])}
        first_kwargs = {**common, "decode_audio": False}
        second_kwargs = {**common, "decode_audio": True}
        first = {"tokens_text": torch.tensor([[1]], dtype=torch.int64),
                 "tokens_len": torch.tensor([1]), "text": ["checking"]}
        second = {"tokens_text": torch.tensor([[2]], dtype=torch.int64),
                  "tokens_len": torch.tensor([1]), "text": ["done"],
                  "audio": torch.tensor([[0.1, -0.2]], dtype=torch.float32),
                  "audio_len": torch.tensor([2])}
        code = [torch.tensor([[7]], dtype=torch.int64)]
        response = {"tool_name": "list_calendar_events",
                    "response": {"status": "no_call", "executed": False}}
        def stock(wrapped):
            return {"pass1": {"kwargs": first_kwargs, "result": first,
                              "codes": []},
                    "pass2": {"kwargs": second_kwargs, "result": second,
                              "codes": code if wrapped else []},
                    "fc_output": {"tool_calls": []}, "call_step": -1,
                    "response_step": 0, "response_token_ids": [],
                    "api_response": response, "tts_wrapped": wrapped}
        payload = {
            "stocks": [stock(True), stock(True), stock(False)],
            "adapter_trace": {"external_tool_execution": False,
                              "replay": None,
                              "pass1": {"raw_result": first,
                                        "codec_codes_per_step": [],
                                        "stock_first_parse": {"calls": []}},
                              "pass2": {"raw_result": second,
                                        "codec_codes_per_step": code}},
            "adapter_calls": [first_kwargs, second_kwargs],
        }
        receipt = saved.save_capture(payload, self.out, byte_cap=65536)
        # The comparison below reads only restored, hash-checked artifacts.
        payload["stocks"][0]["pass2"]["result"]["audio"][0, 0] = 99.0
        restored = saved.load_capture(
            self.out, byte_cap=65536,
            expected_manifest_sha256=receipt["manifest_sha256"])
        result = compare_stock_adapter(*restored["stocks"],
                                       restored["adapter_trace"],
                                       restored["adapter_calls"])
        self.assertEqual(result["status"],
                         "source_comparison_passed_pending_live_provenance_review")
        self.assertEqual(result["stock_adapter_max_abs_pcm"], 0.0)


if __name__ == "__main__":
    unittest.main()
