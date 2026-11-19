"""No candidate weights: fake phase hooks, edits, and source-only schedule tests."""
from __future__ import annotations

import types
import json
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import causal_plan_v1 as plan
from . import causal_worker_v2 as worker


class WorkerSourceTests(unittest.TestCase):
    def test_actual_pinned_42_record_ledger_and_repeat_tampering(self) -> None:
        """Exercise the exact ledger shape that stopped attempt 001, without weights."""
        cases = {item["case_id"]: item for item in worker.old.trusted_cases()}
        outputs = worker.load_valid_v2_outputs(cases)
        self.assertEqual(tuple(outputs), tuple(cases))
        self.assertEqual(len(outputs), 20)
        rows = [json.loads(line) for line in worker.DESCRIPTIVE_V2_RECORDS.read_text().splitlines()]
        case_rows = [row for row in rows if row["type"] == "case"]
        self.assertEqual(len(rows), 42)
        self.assertEqual(len(case_rows), 22)
        self.assertEqual([(row["case_id"], row["repeat"]) for row in case_rows[20:]],
                         [("d02_silence", True), ("d08_silence", True)])
        for repeated in case_rows[20:]:
            self.assertEqual(repeated["output"], outputs[repeated["case_id"]])
            self.assertEqual(repeated["wav_sha256"], cases[repeated["case_id"]]["wav_sha256"])
            self.assertEqual(repeated["pcm_sha256"], cases[repeated["case_id"]]["pcm_sha256"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mutated_records.jsonl"

            def reject(changed: list[dict], why: str) -> None:
                path.write_text("".join(json.dumps(row) + "\n" for row in changed))
                with self.assertRaisesRegex(ValueError, why):
                    worker.load_valid_v2_outputs(cases, path, worker.sha_file(path))

            changed = json.loads(json.dumps(rows))
            repeated = next(row for row in changed if row.get("run_id") == "d02_silence__repeat")
            repeated["output"]["sequence_ids"][-1] += 1
            reject(changed, "repeat output changed")
            changed = json.loads(json.dumps(rows))
            repeated = next(row for row in changed if row.get("run_id") == "d02_silence__repeat")
            repeated["pcm_sha256"] = "0" * 64
            reject(changed, "case metadata changed")
            changed = json.loads(json.dumps(rows))
            diff_positions = [index for index, row in enumerate(changed)
                              if row["type"] == "silence_difference"]
            changed[diff_positions[0]], changed[diff_positions[1]] = (
                changed[diff_positions[1]], changed[diff_positions[0]])
            reject(changed, "row chronology changed")

    def test_executable_go_requires_committed_receipts_and_ancestry(self) -> None:
        """A staged GO or uncommitted admission cannot authorize a model load."""
        with tempfile.TemporaryDirectory(dir=worker.ROOT) as temp:
            directory = Path(temp)
            admission, go = directory / "admission.json", directory / "go.json"
            admission.write_text('{"sources":{}}')
            go.write_text(json.dumps({
                "schema": "nonspeech_whisper_causal_executable_go_v2",
                "status": "GO",
                "admission_sha256": hashlib.sha256(admission.read_bytes()).hexdigest(),
                "sources": {},
            }))
            statuses = {"tracked": True, "staged": False, "ancestor": True}

            def fake_run(cmd, **_kwargs):
                if cmd[1] == "ls-files":
                    return subprocess.CompletedProcess(cmd, 0 if statuses["tracked"] else 1)
                if cmd[1] == "diff":
                    is_staged = "--cached" in cmd
                    return subprocess.CompletedProcess(
                        cmd, 1 if is_staged and statuses["staged"] else 0)
                if cmd[1] == "merge-base":
                    return subprocess.CompletedProcess(cmd, 0 if statuses["ancestor"] else 1)
                raise AssertionError(cmd)

            with patch.object(worker, "ADMISSION", admission), \
                 patch.object(worker, "GO", go), \
                 patch.object(worker, "preflight_receipt",
                              return_value={"sources": {}}), \
                 patch.object(worker.subprocess, "run", side_effect=fake_run), \
                 patch.object(worker.subprocess, "check_output",
                              side_effect=lambda *_a, **_k: "a" * 40):
                # The same commit is never an independent GO.
                with self.assertRaisesRegex(ValueError, "GO must postdate"):
                    worker.validate_go(admission, go)
                with patch.object(worker.subprocess, "check_output",
                                  side_effect=lambda cmd, **_kw: (
                                      "b" * 40 if str(go.relative_to(worker.ROOT))
                                      in cmd else "a" * 40)):
                    worker.validate_go(admission, go)
                    statuses["staged"] = True
                    with self.assertRaisesRegex(ValueError, "dirty receipt"):
                        worker.validate_go(admission, go)
                    statuses["staged"] = False
                    statuses["tracked"] = False
                    with self.assertRaisesRegex(ValueError, "untracked receipt"):
                        worker.validate_go(admission, go)
                    statuses["tracked"] = True
                    statuses["ancestor"] = False
                    with self.assertRaisesRegex(ValueError, "GO must postdate"):
                        worker.validate_go(admission, go)

    def test_fixed_328_names_and_no_hook_donors(self) -> None:
        rows = plan.schedule()
        self.assertEqual(len(rows), 328)
        self.assertEqual(len({worker.call_name(row) for row in rows}), 328)
        self.assertEqual([(row["case_id"], row["arm"])
                          for row in rows[:4]],
                         [("d02_silence", "no_hook"),
                          ("d02_silence", "read_only"),
                          ("d08_silence", "no_hook"),
                          ("d08_silence", "read_only")])

    def test_random_patch_exact_recipe_and_zero_target(self) -> None:
        native = np.zeros((1, 1500, 384), dtype=np.float32)
        donor = native.copy()
        donor[:, :10, :] = 0.125
        one, target, realized = worker.random_patch(native, donor, 123)
        two, target2, realized2 = worker.random_patch(native, donor, 123)
        self.assertEqual(worker.sha_bytes(one.tobytes()), worker.sha_bytes(two.tobytes()))
        self.assertEqual(target, target2)
        self.assertEqual(realized, realized2)
        self.assertLess(abs(realized-target), 1e-5)
        zero, zero_target, zero_realized = worker.random_patch(native, native, 123)
        self.assertEqual((zero_target, zero_realized), (0.0, 0.0))
        self.assertTrue(np.array_equal(zero, native))

    def test_fake_phase_specific_final_hook(self) -> None:
        import torch

        class Block(torch.nn.Module):
            def forward(self, x, mask):
                assert mask is None
                return x + 1

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Block()])
                self.layer_norm = torch.nn.Identity()

            def forward(self, x):
                return self.layer_norm(self.layers[0](x, None))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = types.SimpleNamespace(encoder=Encoder())
                self.generation_config = types.SimpleNamespace(lang_to_id={"<|en|>": 50259})

            def forward(self, x):
                _ = self.model.encoder(x)
                return types.SimpleNamespace(logits=torch.zeros((1, 1, 51865)))

            def detect_language(self, x):
                _ = self(x)
                return torch.tensor([50259])

        row = next(row for row in plan.schedule() if row["stage"] == "A"
                   and row["arm"] == "donor_transcribe")
        fake = Model()
        donor = {site: {phase: np.full((1, 1500, 384), 7, dtype=np.float32)
                        for phase in ("detect", "transcribe")}
                 for site in ("pre_block0", "final_encoder")}
        hooks = worker.PhaseHooks(fake, row, donor)
        hooks.install()
        try:
            x = torch.zeros((1, 1500, 384), dtype=torch.float32)
            fake.detect_language(x)
            fake(x)
            hooks.validated()
            self.assertEqual(hooks.hook["detect"]["final_encoder"]["edit_l2"], 0.0)
            self.assertEqual(hooks.hook["transcribe"]["final_encoder"]["edited_sha256"],
                             worker.sha_bytes(donor["final_encoder"]["transcribe"].tobytes()))
            self.assertEqual(hooks.hook["transcribe"]["pre_block0"]["edit_l2"], 0.0)
        finally:
            hooks.close()
        self.assertEqual(len(hooks.handles), 3)

    def test_full_output_token_parity(self) -> None:
        a = {"sequence_ids": [1, 2, 3], "actual_generation_steps": 1,
             "termination": "eos"}
        self.assertTrue(worker.token_parity(a, a))
        self.assertFalse(worker.token_parity(a, {**a, "sequence_ids": [1, 2, 4]}))
        self.assertIsNone(worker.token_parity(a, None))

    def test_early_active_displaced_and_seeded_full_random(self) -> None:
        cases = [row for row in plan.schedule() if row["stage"] == "B"
                 and row["case_id"] == "d02_beep_train_high"]
        native = np.zeros((1, 1500, 384), dtype=np.float32)
        donor = np.ones_like(native)
        donor_states = {site: {phase: donor for phase in ("detect", "transcribe")}
                        for site in ("pre_block0", "final_encoder")}
        for arm, start, end in (("active_both", 0, 100),
                                ("displaced_both", 100, 200)):
            row = next(row for row in cases if row["arm"] == arm)
            hook = worker.PhaseHooks(object(), row, donor_states)
            edited, support, seed = hook._edit(native, "pre_block0", "detect")
            self.assertEqual(support, [start, end])
            self.assertIsNone(seed)
            self.assertTrue(np.all(edited[:, start:end, :] == 1))
            self.assertTrue(np.all(edited[:, :start, :] == 0))
            self.assertTrue(np.all(edited[:, end:, :] == 0))
        row = next(row for row in cases if row["arm"] == "random_1_both")
        hook = worker.PhaseHooks(object(), row, donor_states)
        edited, support, seed = hook._edit(native, "pre_block0", "detect")
        self.assertEqual(support, [0, 1500])
        self.assertEqual(seed, row["seeds"]["detect"])
        self.assertEqual(worker.sha_bytes(edited.tobytes()),
                         worker.sha_bytes(worker.random_patch(native, donor, seed)[0].tobytes()))

    def test_complete_fake_328_call_census_and_failure_ledger(self) -> None:
        """Exercise orchestration with no checkpoint, model forward, or real WAV."""
        import torch

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
                self.eval()

        def fake_generate(row, _case, _processor, _model, _donor_states,
                          _output_dir, _budget, call_index):
            if getattr(fake_generate, "fail_at", None) == call_index:
                raise RuntimeError("fake mid-matrix failure")
            native = {phase: {site: np.zeros((1, 1, 1), dtype=np.float32)
                              for site in ("pre_block0", "final_encoder")}
                      for phase in ("detect", "transcribe")}
            output = {"sequence_ids": [50258, 50259, 50359, 50363, 291, 50257],
                      "actual_generation_steps": 2, "termination": "eos"}
            return ({"output": output, "hook": {}, "hook_counts": {},
                     "native": native, "sidecars": {}, "input_geometry": {},
                     "first_raw_meta": {}, "first_decoder_meta": {}},
                    {"raw_logits": (np.zeros(3, dtype=np.float32),)})

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            admission, go = base / "admission.json", base / "go.json"
            admission.write_text("{}")
            go.write_text("{}")
            cases = worker.old.trusted_cases()
            for fail_at in (None, 5):
                attempt = base / ("success" if fail_at is None else "failure")
                attempt.mkdir()
                fake_generate.fail_at = fail_at
                with patch.object(worker, "ATTEMPT", attempt), \
                     patch.object(worker, "validate_go", return_value={
                         "snapshot": {}, "schedule_sha256": "fake"}), \
                     patch.object(worker.old, "trusted_cases", return_value=cases), \
                     patch.object(worker.old, "_snapshot_load",
                                  return_value=(object(), FakeModel())), \
                     patch.object(worker, "generate_one", side_effect=fake_generate):
                    output = attempt / "worker_output"
                    if fail_at is None:
                        result = worker.run_model(output, admission, go)
                        self.assertEqual(result["forwards"], 328)
                        self.assertEqual(len((output / "call_progress.jsonl").read_text().splitlines()),
                                         656)
                        self.assertEqual(json.loads((output / "records.jsonl").read_text().splitlines()[-1])[
                            "type"], "complete")
                    else:
                        with self.assertRaisesRegex(RuntimeError, "fake mid-matrix"):
                            worker.run_model(output, admission, go)
                        failure = json.loads((output / "records.jsonl").read_text().splitlines()[-1])
                        self.assertEqual(failure["failed_attempted_call_name"],
                                         worker.call_name(plan.schedule()[4]))
                        self.assertEqual(len(failure["censored_call_names"]), 323)


if __name__ == "__main__":
    unittest.main()
