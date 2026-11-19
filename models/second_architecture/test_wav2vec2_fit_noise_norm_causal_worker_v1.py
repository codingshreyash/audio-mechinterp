"""No-checkpoint fake-model tests for the one-shot 88-forward execution path."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from models.second_architecture import wav2vec2_fit_noise_norm_causal_source_v1 as source
from models.second_architecture import wav2vec2_fit_noise_norm_causal_worker_v1 as worker
from models.second_architecture import wav2vec2_fit_noise_norm_causal_audit_v1 as audit
from models.second_architecture import wav2vec2_fit_noise_norm_causal_launcher_v1 as launcher


class FakeConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 512, kernel_size=10, stride=5, bias=False)
        self.layer_norm = nn.GroupNorm(512, 512, eps=1e-5)

    def forward(self, values):
        return self.layer_norm(self.conv(values[:, None, :]))


class FakeFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.ModuleList([FakeConv()])

    def forward(self, values):
        return self.conv_layers[0](values)


class FakeFeatureProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 768)

    def forward(self, values):
        output = self.linear(values)
        return output, output


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec2 = nn.Module()
        self.wav2vec2.feature_extractor = FakeFeatureExtractor()
        self.wav2vec2.feature_projection = FakeFeatureProjection()
        self.wav2vec2.encoder = nn.Module()
        self.wav2vec2.encoder.layers = nn.ModuleList([nn.Identity()])
        self.lm_head = nn.Linear(768, 32)

    def forward(self, input_values, **_kwargs):
        x = self.wav2vec2.feature_extractor(input_values).transpose(1, 2)
        final_frames = (input_values.shape[1] - 400) // 320 + 1
        x = x[:, :final_frames, :]
        x = self.wav2vec2.feature_projection(x)[0]
        x = self.wav2vec2.encoder.layers[0](x)
        return SimpleNamespace(logits=self.lm_head(x))


class FakeProcessor:
    def __call__(self, samples, **_kwargs):
        x = np.asarray(samples, dtype=np.float32)
        return {"input_values": np.asarray(
            ((x-x.mean()) / np.sqrt(x.var()+1e-7))[None, :], dtype=np.float32)}

    def batch_decode(self, ids):
        return [" ".join(str(int(x)) for x in row) for row in ids]


def synthetic_pcm(case_index: int):
    sample = np.arange(1000, dtype=np.int32)
    clean = np.rint(600 * np.sin(sample / 8 + case_index)).astype("<i2")
    word = clean.copy()
    word[250:350] += 120 + case_index
    outside = clean.copy()
    outside[500:550] += 90 + case_index
    return clean, word, outside


def prior_item(model, processor, pcm, arm):
    x = source.normalize_with_statistics(pcm, *source.input_statistics(pcm))
    with torch.inference_mode():
        ctc = worker.decode(model(torch.from_numpy(x[None, :])).logits, processor)
    return {"arm": arm, "transcript": ctc["transcript"],
            "argmax_ids_sha256": hashlib.sha256(json.dumps(
                ctc["argmax_ids"], separators=(",", ":")).encode()).hexdigest()}


class FakeExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(20261119)
        torch.set_num_threads(4)

    def test_88_forward_execution_schedule_with_fake_model(self):
        model = FakeModel().eval()
        processor = FakeProcessor()
        cache = {}
        rows = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            budget = worker.Budget(output)
            for index in range(4):
                clean, word, outside = synthetic_pcm(index)
                case = f"fake{index}_silver"
                for noise_arm, noisy in (("word_noise", word),
                                         ("outside_noise", outside)):
                    pair = {"case": case, "noise_arm": noise_arm,
                            "slot": [250, 350], "prior_stock": [
                                prior_item(model, processor, clean, "clean"),
                                prior_item(model, processor, noisy, noise_arm)]}
                    worker.stage_i_pair(model, processor, clean, noisy, pair,
                                        output, budget, cache, rows.append)
                pair = {"case": case, "noise_arm": "word_noise", "slot": [250, 350]}
                worker.stage_ii_pair(model, processor, clean, word, pair,
                                     output, budget, rows.append)
            self.assertEqual(budget.calls, 88)
            self.assertEqual(len(rows), 88)
            self.assertEqual(len({r["name"] for r in rows}), 88)
            self.assertEqual(len([r for r in rows if r["stage"] == "I"]), 56)
            self.assertEqual(len([r for r in rows if r["stage"] == "II"]), 32)
            for case_index in range(4):
                case = f"fake{case_index}_silver"
                self.assertEqual(sum(r["name"] == f"I_{case}_clean_stock" for r in rows), 1)
                self.assertEqual(sum(r["name"] == f"I_{case}_word_noise_stock" for r in rows), 1)
                self.assertEqual(sum(r["name"] == f"I_{case}_outside_noise_stock" for r in rows), 1)
            self.assertTrue(all(r["ctc"]["frame_count"] == 2 for r in rows))

    def test_missing_go_and_forward_ceiling_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "missing frozen admission"):
                worker.validate_go(root / "admission.json", root / "go.json")
            budget = worker.Budget(root)
            budget.calls = worker.MAX_FORWARDS
            with self.assertRaisesRegex(ValueError, "forward ceiling"):
                budget.check(before_forward=True)

    def test_frozen_execution_plan_and_audit_name_order(self):
        plan = worker.execution_plan()  # hashes old source and all frozen FIT PCM
        self.assertEqual(plan["total_forward_ceiling"], 88)
        self.assertEqual(plan["stage_i_stock_no_hook_count"], 12)
        self.assertEqual(plan["stage_i_stock_read_only_count"], 12)
        self.assertEqual(plan["scored_forward_count"], 48)
        self.assertEqual(len(audit.expected_forward_names(plan)), 88)

    def test_launcher_one_shot_fails_closed_on_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            admission, go = root / "admission.json", root / "go.json"
            admission.write_text("{}")
            go.write_text("{}")

            def fake_probe(attempt, name, **_kwargs):
                worker.write_once(attempt / f"{name}_host.json", {"collected_utc": "2026-11-19T00:00:00+00:00"})
                (attempt / f"{name}_power_log.txt.gz").write_bytes(b"fake")
                return {"collected_utc": "2026-11-19T00:00:00+00:00"}

            def fake_run(_argv, stdout, stderr):
                stdout.write_bytes(b"")
                stderr.write_bytes(b"")
                output = stdout.parent / "worker_output"
                output.mkdir()
                (output / "records.jsonl").write_text('{"type":"complete"}\n')
                return 0, False, 0.01

            with patch.object(launcher, "ATTEMPT", root / "attempt"), \
                 patch.object(worker, "validate_go", return_value={"source_binding": {}}), \
                 patch.object(launcher, "probe_host", side_effect=fake_probe), \
                 patch.object(launcher, "run_bounded", side_effect=fake_run):
                first = launcher.launch(admission, go)
                self.assertEqual(first["status"], "completed_pending_saved_audit")
                with self.assertRaisesRegex(ValueError, "already exists"):
                    launcher.launch(admission, go)

    def test_saved_array_auditor_rejects_changed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            evidence = attempt / "worker_output"
            evidence.mkdir()
            budget = worker.Budget(evidence)
            reference = worker.save_array(evidence, "logits/toy.npy",
                                          np.zeros((1, 2, 32), dtype=np.float32), budget)
            observed = audit.load_array(attempt, reference, (1, 2, 32))
            self.assertEqual(observed.shape, (1, 2, 32))
            (evidence / "logits/toy.npy").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "changed"):
                audit.load_array(attempt, reference, (1, 2, 32))

    def test_complete_saved_audit_on_fake_88_forward_attempt(self):
        from models.second_architecture import acoustic_fit_noise_pilot_v2 as old
        from transformers import Wav2Vec2Processor

        model = FakeModel().eval()
        processor = FakeProcessor()
        clips, pairs, expected = {}, [], {}
        for index in range(4):
            case = f"fake{index}_silver"
            clean, word, outside = synthetic_pcm(index)
            clips[case] = {"clean": clean, "word_noise": word,
                           "outside_noise": outside}
            expected[f"fake{index}"] = {"frames": 1000}
            for noise_arm, noisy in (("word_noise", word), ("outside_noise", outside)):
                pairs.append({"case": case, "engine": f"fake{index}",
                              "word": "silver", "noise_arm": noise_arm,
                              "slot": [250, 350], "prior_stock": [
                                  prior_item(model, processor, clean, "clean"),
                                  prior_item(model, processor, noisy, noise_arm)]})
        plan = {"pairs": pairs, "stage_ii_cases": [f"fake{i}_silver" for i in range(4)]}

        class FakeWeights:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def get_tensor(self, key):
                layer = model.wav2vec2.feature_extractor.conv_layers[0]
                if key.endswith("conv.weight"):
                    return layer.conv.weight.detach().clone()
                value = layer.layer_norm
                return value.weight.detach().clone() if key.endswith("weight") else value.bias.detach().clone()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "attempt"
            output = attempt / "worker_output"
            output.mkdir(parents=True)
            budget = worker.Budget(output)
            cache, records = {}, []
            admission, go = root / "admission.json", root / "go.json"
            admission.write_text("{}")
            go.write_text("{}")
            records.append({"type": "start", "planned_forwards": 88,
                            "admission_sha256": source.sha_file(admission),
                            "go_sha256": source.sha_file(go),
                            "plan_sha256": worker.bytes_sha(worker.canonical(plan).encode())})
            emit = lambda r: records.append({"type": "forward", **r})
            for pair in pairs:
                case, noise_arm = pair["case"], pair["noise_arm"]
                records.append({"type": "pair_start", "stage": "I", "pair": case,
                                "noise_arm": noise_arm, "forward_count_before": budget.calls})
                worker.stage_i_pair(model, processor, clips[case]["clean"],
                                    clips[case][noise_arm], pair, output, budget, cache, emit)
                records.append({"type": "pair_complete", "stage": "I", "pair": case,
                                "noise_arm": noise_arm, "forward_count_after": budget.calls})
            for pair in pairs:
                if pair["noise_arm"] != "word_noise":
                    continue
                case = pair["case"]
                records.append({"type": "pair_start", "stage": "II", "pair": case,
                                "noise_arm": "word_noise", "forward_count_before": budget.calls})
                worker.stage_ii_pair(model, processor, clips[case]["clean"],
                                     clips[case]["word_noise"], pair, output, budget, emit)
                records.append({"type": "pair_complete", "stage": "II", "pair": case,
                                "noise_arm": "word_noise", "forward_count_after": budget.calls})
            records.append({"type": "complete", "forward_count": budget.calls,
                            "output_bytes": worker.output_bytes(output),
                            "rss_bytes": worker.rss_bytes(),
                            "elapsed_seconds": budget.start and 1.0})
            (output / "records.jsonl").write_text(
                "".join(worker.canonical(r)+"\n" for r in records))
            launch_ref = {"schema": "fit_norm_causal_launch_v1", "status": "started",
                          "admission_sha256": source.sha_file(admission),
                          "go_sha256": source.sha_file(go)}
            result = {**launch_ref, "status": "completed_pending_saved_audit",
                      "worker_returncode": 0, "worker_timed_out": False,
                      "worker_elapsed_seconds": 1.0}
            worker.write_once(attempt / "launch_start.json", launch_ref)
            worker.write_once(attempt / "launch_result.json", result)
            for name in ("pre_host.json", "post_host.json", "pre_power_log.txt.gz",
                         "post_power_log.txt.gz", "worker_stdout.txt", "worker_stderr.txt"):
                (attempt / name).write_bytes(b"")
            with patch.object(worker, "validate_go", return_value={}), \
                 patch.object(worker, "execution_plan", return_value=plan), \
                 patch.object(old, "stimuli", return_value=clips), \
                 patch.object(old, "EXPECTED", expected), \
                 patch.object(Wav2Vec2Processor, "from_pretrained", return_value=processor), \
                 patch("safetensors.safe_open", return_value=FakeWeights()), \
                 patch.object(audit, "check_host", return_value=None):
                result = audit.audit(attempt, admission, go)
                self.assertEqual(result["status"], "saved_result_audit_pass")
                self.assertEqual(result["forwards"], 88)
                victim = output / "logits" / records[2]["logits"]["path"].split("/")[-1]
                victim.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "changed"):
                    audit.audit(attempt, admission, go)


if __name__ == "__main__":
    unittest.main()
