"""Synthetic-only arm-supervisor tests; no model, human, anchor or controller records."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from . import live_parity_launcher_v1 as launch


class LauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.hashes = {
            "selected_wav": "a" * 64,
            "checkpoint_weight": "b" * 64,
            "checkpoint_config": "c" * 64,
            "nemo:examples/speechlm2/sample_audio/sample_fc.wav": "d" * 64,
        }
        self.plan = {
            "schema": "voicechat_live_parity_source_plan_v1",
            "status": "synthetic_test_only_no_model_work",
            "synthetic_test_injection": True,
            "selected": {"case_id": "B", "voice_id": "piper_amy_en_us",
                         "wav_sha256": self.hashes["selected_wav"]},
            "input_hashes": self.hashes, "seed": 1729,
            "arm_order": list(launch.ORDER), "fresh_process_per_arm": True,
            "external_tool_execution": False,
            "required_comparisons": {"exact_discrete_and_replay": True,
                                     "stock_repeat_required": True,
                                     "tts_no_wrapper_control": True,
                                     "raw_unspliced_pcm_max_abs": 1e-4},
        }
        self.caps = {"wall_seconds": 5, "vram_bytes": 1024,
                     "output_bytes": 8192}
        self.seen: list[str] = []

    def _worker(self, arm: str, directory: Path, response_path: Path | None):
        self.seen.append(arm)
        if arm == "official_sample":
            (directory / "sample_raw.bin").write_bytes(b"sample-only")
        else:
            for name in launch.REQUIRED_FIXTURE_ARTIFACTS:
                (directory / name).write_bytes((arm + name).encode())
        provenance = {
            "source_plan_sha256": hashlib.sha256(
                launch.canonical_bytes(self.plan)).hexdigest(),
            "input_hashes": self.hashes, "seed": self.plan["seed"],
            "checkpoint_weight_sha256": self.hashes["checkpoint_weight"],
            "checkpoint_config_sha256": self.hashes["checkpoint_config"],
            "process_id": arm,
        }
        if arm != "official_sample":
            provenance.update({"wav_sha256": self.hashes["selected_wav"],
                               "prompt_token_sha256": "e" * 64})
        if arm.startswith("stock_"):
            self.assertIsNotNone(response_path)
            provenance["response_sha256"] = hashlib.sha256(
                response_path.read_bytes()).hexdigest()
        capture = ({"trace": {"external_tool_execution": False, "replay": None,
                              "pass1": {"raw_result": {}},
                              "pass2": {"raw_result": {}}},
                    "calls": [{"decode_audio": False}, {"decode_audio": True}]}
                   if arm == "adapter" else {"stock": {
                       "arm": arm,
                       "pass1": {"kwargs": {}, "result": {}},
                       "pass2": {"kwargs": {}, "result": {}}}}
                   if arm.startswith("stock_") else {})
        return {"returncode": 0, "peak_vram_bytes": 1,
                "provenance": provenance, "capture": capture}

    @staticmethod
    def _compare(stock_a, stock_b, no_tts, trace, calls, *, waveform_tolerance):
        assert [stock_a["arm"], stock_b["arm"], no_tts["arm"]] == [
            "stock_a", "stock_b", "stock_no_tts"]
        assert trace["replay"] is None and len(calls) == 2
        assert waveform_tolerance == 1e-4
        return {"status": "source_comparison_passed_pending_live_provenance_review"}

    def _run(self, **changes):
        return launch.run_synthetic_protocol(
            plan=self.plan, output_dir=self.root / "run",
            worker=changes.get("worker", self._worker),
            rehash_inputs=changes.get("rehash_inputs", lambda: self.hashes),
            caps=changes.get("caps", self.caps),
            compare=changes.get("compare", self._compare),
        )

    def test_fixed_order_placeholder_and_no_rerun(self):
        result = self._run()
        self.assertEqual(self.seen, list(launch.ORDER))
        self.assertEqual(result["status"], "synthetic_test_only_no_model_work")
        self.assertTrue(result["no_call"])
        self.assertEqual((self.root / "run/api_response.json").read_bytes(),
                         launch.fixed_response_bytes(
                             {"external_tool_execution": False, "replay": None},
                             [{"decode_audio": False}, {"decode_audio": True}])[0])
        self.assertEqual(set(result["arm_receipts"]), set(launch.ORDER))
        with self.assertRaises(FileExistsError):
            self._run()

    def test_rejects_non_synthetic_or_wrong_selection_before_worker(self):
        self.plan["status"] = "source_plan_only_no_model_work"
        with self.assertRaisesRegex(ValueError, "explicitly synthetic"):
            self._run()
        self.assertEqual(self.seen, [])
        self.plan["status"] = "synthetic_test_only_no_model_work"
        self.plan["selected"]["voice_id"] = "other"
        with self.assertRaisesRegex(ValueError, "Fixed case-B"):
            self._run()
        self.assertEqual(self.seen, [])

    def test_rejects_changed_input_before_sample(self):
        with self.assertRaisesRegex(ValueError, "changed before sample"):
            self._run(rehash_inputs=lambda: {**self.hashes, "selected_wav": "0" * 64})
        self.assertEqual(self.seen, [])

    def test_in_place_plan_and_hash_mutation_after_sample_is_rejected(self):
        def worker(arm, directory, response_path):
            result = self._worker(arm, directory, response_path)
            if arm == "official_sample":
                self.hashes["checkpoint_weight"] = "0" * 64
                self.plan["seed"] = 7
            return result
        with self.assertRaisesRegex(ValueError, "source/checkpoint/seed provenance differs"):
            self._run(worker=worker)
        self.assertEqual(self.seen, ["official_sample"])

    def test_missing_raw_capture_stops_without_stock(self):
        def worker(arm, directory, response_path):
            result = self._worker(arm, directory, response_path)
            if arm == "adapter":
                (directory / "pass1_raw.bin").unlink()
            return result
        with self.assertRaisesRegex(ValueError, "raw artifact missing"):
            self._run(worker=worker)
        self.assertEqual(self.seen, ["official_sample", "adapter"])
        self.assertTrue((self.root / "run/adapter/failure.json").is_file())

    def test_partial_replay_injection_stops_before_stock(self):
        def worker(arm, directory, response_path):
            result = self._worker(arm, directory, response_path)
            if arm == "adapter":
                result["capture"]["trace"]["replay"] = {
                    "response_selector": {"tool_name": "list_calendar_events",
                                          "response": {"executed": False}}}
            return result
        with self.assertRaisesRegex(ValueError, "complete second-pass injection"):
            self._run(worker=worker)
        self.assertEqual(self.seen, ["official_sample", "adapter"])

    def test_changed_response_file_stops_next_stock(self):
        def worker(arm, directory, response_path):
            result = self._worker(arm, directory, response_path)
            if arm == "stock_a":
                response_path.write_bytes(b"changed")
            return result
        with self.assertRaisesRegex(ValueError, "changed the frozen response"):
            self._run(worker=worker)
        self.assertEqual(self.seen, ["official_sample", "adapter", "stock_a"])

    def test_missing_pass_capture_stops(self):
        def worker(arm, directory, response_path):
            result = self._worker(arm, directory, response_path)
            if arm == "adapter":
                del result["capture"]["trace"]["pass1"]
            return result
        with self.assertRaisesRegex(ValueError, "Adapter pass capture missing"):
            self._run(worker=worker)
        self.assertEqual(self.seen, ["official_sample", "adapter"])

    def test_no_call_stock_injection_is_rejected(self):
        def worker(arm, directory, response_path):
            result = self._worker(arm, directory, response_path)
            if arm == "stock_a":
                result["capture"]["stock"]["pass2"]["kwargs"]["function_calls"] = [1]
            return result
        with self.assertRaisesRegex(ValueError, "replay injection differs"):
            self._run(worker=worker)
        self.assertEqual(self.seen, ["official_sample", "adapter", "stock_a"])

    def test_process_seed_prompt_and_output_caps(self):
        for kind, pattern in (("seed", "source/checkpoint/seed"),
                              ("pid", "fresh process"),
                              ("prompt", "prompt token hash differs"),
                              ("bytes", "output cap")):
            with self.subTest(kind=kind):
                run_dir = self.root / "run"
                if run_dir.exists():
                    import shutil
                    shutil.rmtree(run_dir)
                self.seen.clear()
                def worker(arm, directory, response_path):
                    result = self._worker(arm, directory, response_path)
                    if arm == "adapter" and kind == "seed":
                        result["provenance"]["seed"] = 7
                    if arm == "adapter" and kind == "pid":
                        result["provenance"]["process_id"] = "official_sample"
                    if arm == "stock_a" and kind == "prompt":
                        result["provenance"]["prompt_token_sha256"] = "f" * 64
                    if arm == "adapter" and kind == "bytes":
                        (directory / "large.bin").write_bytes(b"x" * 9000)
                    return result
                with self.assertRaisesRegex(ValueError, pattern):
                    self._run(worker=worker)

    def test_stock_instability_is_not_a_parity_pass(self):
        def unstable(*_args, **_kwargs):
            raise ValueError("stock repeat audio unstable")
        with self.assertRaisesRegex(ValueError, "stock repeat"):
            self._run(compare=unstable)
        self.assertFalse((self.root / "run/synthetic_comparison.json").exists())


if __name__ == "__main__":
    unittest.main()
