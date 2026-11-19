"""No-model adversarial fixtures for the independent saved-result audit."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from .voicechat_step400_saved_auditor_v1 import (
    PINNED_SAMPLES, PINNED_SAMPLE_BYTES, PINNED_SOURCES, audit_saved_attempt,
)
from . import voicechat_step400_supervisor_v1 as parent

IDENTITY = {
    "nemo_revision": "097dfe9e2f55baf653b83035868bdc89849f1b47",
    "voicechat_revision": "443794ea956ef0065f001967ffd00e77f519cb39",
    "nano_revision": "6533e8de2c68e4536bf7c411d7a3ce5734111476",
    "source_sha256": PINNED_SOURCES, "sample_sha256": PINNED_SAMPLES,
    "sample_bytes": PINNED_SAMPLE_BYTES,
    "config_sha256": "2e0b67b56fefeb4815e436d66ad84f2f94be192b1d543054c810ed239a28cff5",
    "weight_sha256": "d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b",
    "weight_bytes": 44_382_749_892,
    "template_sha256": PINNED_SOURCES[
        "examples/speechlm2/function_calling/template.jinja"],
    "system_message_sha256": "1" * 64, "tools_sha256": "2" * 64,
    "prompt_sha256": "3" * 64, "sample_id": "sample_fc",
}
STEPS = [1, 50, 100, 200, 300, 400]


def encoded(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def carried() -> dict:
    return {"complete": True, "unknown": [], "unknown_count": 0, "node_count": 2,
            "tensor_rows": [{"family": "stt", "path": "stt[entry:0]", "shape": [8],
                             "dtype": "torch.float32", "device": "cuda:0",
                             "logical_bytes": 32, "storage_bytes": 32}],
            "families": {"stt": {"tensor_count": 1, "logical_bytes": 32,
                                 "unique_storage_bytes": 32,
                                 "unique_cuda_storage_bytes": 32},
                         "tts": {"tensor_count": 0, "logical_bytes": 0,
                                 "unique_storage_bytes": 0,
                                 "unique_cuda_storage_bytes": 0}},
            "shared_storage_bytes": 0, "global_unique_storage_bytes": 32,
            "global_unique_cuda_storage_bytes": 32}


def cuda() -> dict:
    return {"allocated_bytes": 100, "reserved_bytes": 200,
            "max_allocated_bytes": 150, "max_reserved_bytes": 220}


def checkpoints() -> list[dict]:
    result = []
    for i, t in enumerate(STEPS):
        start = 1.01 + i * .018
        snapshot = start + .01
        sample = snapshot + .001
        row = {"schema": "voicechat_step400_checkpoint_v1", "t": t, "T": 859,
               "interval_start_monotonic": start, "snapshot_monotonic": snapshot,
               "nvml_lag_seconds": sample - snapshot, "device_id": "0",
               "cuda": cuda(),
               "nvml": {"used_bytes": 1024, "model_pid_present": True,
                        "sample_monotonic": sample, "query_seconds": .001},
               "carried": carried()}
        if i == 0:
            row["pre_step_1_baseline"] = {
                "snapshot_monotonic": 1.001, "reset_monotonic": start,
                "cuda": cuda(), "T": 859,
                "nvml": {"used_bytes": 1024, "model_pid_present": True,
                         "sample_monotonic": 1.002, "query_seconds": .001}}
        result.append(row)
    return result


def poll() -> list[dict]:
    times = [(0.9, .99), (.99, 1.08), (1.08, 1.17), (1.17, 1.26)]
    return [{"sample_begin_monotonic": start, "sample_end_monotonic": end,
             "completed_sample_gap_seconds": end - start,
             "device_mib": 123, "process_mib": 100,
             "model_pid_present": True} for start, end in times]


def write_fixture(root: Path, *, rows: list[dict] | None = None,
                  polls: list[dict] | None = None) -> None:
    root.mkdir()
    rows = checkpoints() if rows is None else rows
    polls = poll() if polls is None else polls
    (root / "checkpoints.jsonl").write_bytes(b"".join(map(encoded, rows)))
    (root / "gpu_poll.jsonl").write_bytes(b"".join(map(encoded, polls)))
    (root / "stdout.log").write_bytes(b"")
    (root / "stderr.log").write_bytes(b"")
    ledger = (root / "checkpoints.jsonl").read_bytes()
    core = {"schema": "voicechat_step400_core_completion_v1",
            "status": "diagnostic_early_stop_complete", "inventory_complete": True,
            "checkpoint_steps": STEPS,
            "ledger": {"path": "checkpoints.jsonl", "sha256": hashlib.sha256(ledger).hexdigest(),
                       "bytes": len(ledger), "checkpoints": 6},
            "no_model_behavior_claim": True}
    worker = {"schema": "voicechat_step400_worker_completion_v1",
              "status": "diagnostic_early_stop_complete", "identity": copy.deepcopy(IDENTITY),
              "core": core, "no_second_pass": True}
    (root / "worker_completion.json").write_bytes(encoded(worker))
    reseal_parent(root)


def reseal_parent(root: Path, **overrides) -> None:
    files = {p.name: p.read_bytes() for p in root.iterdir()
             if p.is_file() and p.name != "parent_result.json"}
    parent = {"schema": "voicechat_step400_parent_candidate_v1",
              "status": "bounded_child_complete_pending_saved_audit", "reason": None,
              "returncode": 0, "model_pid_seen": True, "heartbeat_count": 1,
              "truncated_evidence": [], "elapsed_monotonic_seconds": 1.0,
              "retained_evidence_bytes_before_parent_receipt": sum(map(len, files.values())),
              "evidence_inventory_before_parent_receipt": {
                  hashlib.sha256(name.encode()).hexdigest():
                      {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                  for name, data in files.items()},
              "peak_device_mib": 123, "peak_process_mib": 100,
              "sample_count": 4, "largest_sample_gap_seconds": .09}
    parent["output_dir"] = str(root)
    parent.update(overrides)
    (root / "parent_result.json").write_bytes(encoded(parent))


def rewrite_ledger(root: Path, rows: list[dict]) -> None:
    data = b"".join(map(encoded, rows))
    (root / "checkpoints.jsonl").write_bytes(data)
    worker = json.loads((root / "worker_completion.json").read_bytes())
    worker["core"]["ledger"].update(sha256=hashlib.sha256(data).hexdigest(),
                                     bytes=len(data))
    (root / "worker_completion.json").write_bytes(encoded(worker))
    reseal_parent(root)


class SavedAuditorTests(unittest.TestCase):
    def test_real_probe_worker_parent_fake_process_to_saved_audit(self):
        script = textwrap.dedent('''
            import os, time, types
            from pathlib import Path
            from unittest.mock import patch
            import torch
            from models.second_architecture.voicechat_calendar_fixture_v1 import voicechat_step400_worker_v1 as worker
            from models.second_architecture.voicechat_calendar_fixture_v1 import voicechat_step400_probe_core_v1 as probe
            from models.second_architecture.voicechat_calendar_fixture_v1.test_voicechat_step400_saved_auditor_v1 import IDENTITY

            class STT:
                def _step_inference(self, t, state, *args, **kwargs):
                    return None
            class TTS:
                def infer_codes_one_step(self, **kwargs):
                    return (None, None)
            model = types.SimpleNamespace(stt_model=STT(), tts_model=TTS())
            def offline(current_model, **kwargs):
                assert current_model is model and kwargs['decode_audio'] is False
                state = {'T': 859}
                for t in range(1, 401):
                    current_model.stt_model._step_inference(t, state, None, None)
                    current_model.tts_model.infer_codes_one_step(**{
                        key: (True if key == 'ignore_eos_flag_stop' else None)
                        for key in probe.TTS_KWARGS})
            def nvml(pid):
                return {'used_bytes': 1024, 'model_pid_present': True,
                        'sample_monotonic': time.monotonic(), 'query_seconds': 0.0}
            fake_cuda = {
                'synchronize': lambda device: None,
                'memory_allocated': lambda device: 100,
                'memory_reserved': lambda device: 200,
                'max_memory_allocated': lambda device: 150,
                'max_memory_reserved': lambda device: 220,
                'reset_peak_memory_stats': lambda device: None,
            }
            root = Path.cwd().parent
            with patch.object(worker, 'verify_pinned_public_sample', return_value=dict(IDENTITY)), \\
                 patch.object(worker, 'import_pinned_official', return_value=(object(), types.SimpleNamespace(run_offline_inference=offline))), \\
                 patch.object(worker, 'prepare_official_inputs', return_value=(model, None, None, None, None, {})), \\
                 patch.object(probe, 'nvml_process_sample', side_effect=nvml), \\
                 patch.multiple(torch.cuda, **fake_cuda):
                worker.run_worker(nemo_dir=root / 'nemo', checkpoint_dir=root / 'staged',
                                  output_dir=Path.cwd(),
                                  heartbeat_fd=int(os.environ['VOICECHAT_DIAG_HEARTBEAT_FD']))
            time.sleep(.25)
        ''')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root / "nemo").mkdir()
            (root / "staged").mkdir()
            environment = os.environ.copy()
            repo = Path(__file__).resolve().parents[3]
            environment["PYTHONPATH"] = str(repo)
            parent_result = parent.supervise_candidate(
                command=[sys.executable, "-c", script], environment=environment,
                output_dir=root / "attempt", staged_checkpoint_dir=root / "staged",
                gpu_sample=lambda _pid: parent.GpuSample(1, 1, True),
                disk_free_bytes=lambda _path: 31 * 1024**3)
            self.assertEqual(parent_result["status"],
                             "bounded_child_complete_pending_saved_audit", parent_result)
            audited = audit_saved_attempt(root / "attempt", expected_identity=IDENTITY)
            self.assertEqual(audited["status"], "PASS", audited)

    def test_complete_six_checkpoint_fixture_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "attempt"
            write_fixture(root)
            result = audit_saved_attempt(root, expected_identity=IDENTITY)
            self.assertEqual(result["status"], "PASS", result)
            self.assertEqual([r["t"] for r in result["checkpoints"]], STEPS)

    def test_any_missing_or_extra_file_fails(self):
        for mutation in (lambda root: (root / "checkpoints.jsonl").unlink(),
                         lambda root: (root / "unexpected").write_bytes(b"x")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                root = Path(folder) / "attempt"
                write_fixture(root)
                mutation(root)
                self.assertEqual(audit_saved_attempt(root, expected_identity=IDENTITY)["status"],
                                 "NO_GO")

    def test_external_failure_receipt_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "attempt"
            write_fixture(root)
            (Path(folder) / "attempt.parent_failure.json").write_text("{}")
            self.assertEqual(audit_saved_attempt(root, expected_identity=IDENTITY)["status"],
                             "NO_GO")

    def test_changed_file_hash_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "attempt"
            write_fixture(root)
            (root / "stdout.log").write_bytes(b"modified")
            self.assertEqual(audit_saved_attempt(root, expected_identity=IDENTITY)["status"],
                             "NO_GO")

    def test_incomplete_checkpoint_even_with_resealed_parent_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "attempt"
            write_fixture(root, rows=checkpoints()[:-1])
            self.assertEqual(audit_saved_attempt(root, expected_identity=IDENTITY)["status"],
                             "NO_GO")

    def test_union_mismatch_even_with_rehashed_ledger_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "attempt"
            write_fixture(root)
            rows = checkpoints()
            rows[3]["carried"]["global_unique_storage_bytes"] = 33
            data = b"".join(map(encoded, rows))
            (root / "checkpoints.jsonl").write_bytes(data)
            worker = json.loads((root / "worker_completion.json").read_bytes())
            worker["core"]["ledger"]["sha256"] = hashlib.sha256(data).hexdigest()
            worker["core"]["ledger"]["bytes"] = len(data)
            (root / "worker_completion.json").write_bytes(encoded(worker))
            reseal_parent(root)
            self.assertIn("union", audit_saved_attempt(root, expected_identity=IDENTITY)["reason"])

    def test_poll_gap_even_with_resealed_parent_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "attempt"
            write_fixture(root)
            rows = poll()
            rows[1]["sample_end_monotonic"] = 1.2
            rows[1]["completed_sample_gap_seconds"] = .21
            (root / "gpu_poll.jsonl").write_bytes(b"".join(map(encoded, rows)))
            reseal_parent(root, largest_sample_gap_seconds=.21)
            self.assertEqual(audit_saved_attempt(root, expected_identity=IDENTITY)["status"],
                             "NO_GO")

    def test_missing_baseline_even_with_rehashed_ledger_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "attempt"
            write_fixture(root)
            rows = checkpoints()
            del rows[0]["pre_step_1_baseline"]
            data = b"".join(map(encoded, rows))
            (root / "checkpoints.jsonl").write_bytes(data)
            worker = json.loads((root / "worker_completion.json").read_bytes())
            worker["core"]["ledger"].update(sha256=hashlib.sha256(data).hexdigest(),
                                             bytes=len(data))
            (root / "worker_completion.json").write_bytes(encoded(worker))
            reseal_parent(root)
            self.assertEqual(audit_saved_attempt(root, expected_identity=IDENTITY)["status"],
                             "NO_GO")

    def test_identity_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "attempt"
            write_fixture(root)
            other = {**IDENTITY, "prompt_sha256": "0" * 64}
            self.assertEqual(audit_saved_attempt(root, expected_identity=other)["status"], "NO_GO")

    def test_resealed_impossible_measurements_fail(self):
        def long_query(rows):
            rows[1]["nvml"]["query_seconds"] = 100.0

        def oversized_cuda(rows):
            huge = 140_000 * 1024**2
            rows[1]["cuda"].update(allocated_bytes=huge,
                                   reserved_bytes=huge,
                                   max_allocated_bytes=huge,
                                   max_reserved_bytes=huge)

        def oversized_carried(rows):
            huge = 1024**4
            state = rows[1]["carried"]
            state["tensor_rows"][0]["storage_bytes"] = huge
            state["families"]["stt"]["unique_storage_bytes"] = huge
            state["families"]["stt"]["unique_cuda_storage_bytes"] = huge
            state["global_unique_storage_bytes"] = huge
            state["global_unique_cuda_storage_bytes"] = huge

        def float_step(rows):
            rows[2]["t"] = 100.0

        for mutation in (long_query, oversized_cuda, oversized_carried, float_step):
            with self.subTest(mutation=mutation.__name__), tempfile.TemporaryDirectory() as folder:
                root = Path(folder) / "attempt"
                write_fixture(root)
                rows = checkpoints()
                mutation(rows)
                rewrite_ledger(root, rows)
                result = audit_saved_attempt(root, expected_identity=IDENTITY)
                self.assertEqual(result["status"], "NO_GO", result)

    def test_parent_elapsed_shorter_than_poll_span_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "attempt"
            write_fixture(root)
            reseal_parent(root, elapsed_monotonic_seconds=.001)
            self.assertEqual(audit_saved_attempt(root, expected_identity=IDENTITY)["status"],
                             "NO_GO")

    def test_content_in_tensor_path_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "attempt"
            write_fixture(root)
            rows = checkpoints()
            rows[0]["carried"]["tensor_rows"][0]["path"] = "stt[secret prompt]"
            data = b"".join(map(encoded, rows))
            (root / "checkpoints.jsonl").write_bytes(data)
            worker = json.loads((root / "worker_completion.json").read_bytes())
            worker["core"]["ledger"].update(sha256=hashlib.sha256(data).hexdigest(),
                                             bytes=len(data))
            (root / "worker_completion.json").write_bytes(encoded(worker))
            reseal_parent(root)
            self.assertIn("content", audit_saved_attempt(root, expected_identity=IDENTITY)["reason"])

    def test_duplicate_json_key_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "attempt"
            write_fixture(root)
            parent_path = root / "parent_result.json"
            original = parent_path.read_bytes()
            parent_path.write_bytes(original[:-2] + b',"status":"bounded_child_complete_pending_saved_audit"}\n')
            self.assertEqual(audit_saved_attempt(root, expected_identity=IDENTITY)["status"],
                             "NO_GO")

    def test_symlinked_log_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "attempt"
            write_fixture(root)
            (root / "stdout.log").unlink()
            (root / "stdout.log").symlink_to(root / "stderr.log")
            self.assertEqual(audit_saved_attempt(root, expected_identity=IDENTITY)["status"],
                             "NO_GO")

    def test_raw_content_field_in_carried_fails_even_after_rehash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "attempt"
            write_fixture(root)
            rows = checkpoints()
            rows[0]["carried"]["decoded_text"] = "secret"
            data = b"".join(map(encoded, rows))
            (root / "checkpoints.jsonl").write_bytes(data)
            worker = json.loads((root / "worker_completion.json").read_bytes())
            worker["core"]["ledger"].update(sha256=hashlib.sha256(data).hexdigest(),
                                             bytes=len(data))
            (root / "worker_completion.json").write_bytes(encoded(worker))
            reseal_parent(root)
            self.assertEqual(audit_saved_attempt(root, expected_identity=IDENTITY)["status"],
                             "NO_GO")


if __name__ == "__main__":
    unittest.main()
