"""CPU-only fake-model and receipt tests for proposed Stage-B replay plumbing."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

import voxtral_old_fit_handoff_replay_stage_b_v1 as target
import voxtral_old_fit_handoff_native_stage_a_v1 as native
from test_voxtral_old_fit_handoff_native_stage_a_v1 import FakeModel


class FakeHead(nn.Module):
    def forward(self, hidden):
        logits = torch.zeros((*hidden.shape[:-1], target.LOGIT_WIDTH),
                             dtype=hidden.dtype, device=hidden.device)
        logits[..., 0] = hidden[..., 0]
        return logits


def fake_model_and_batch():
    model = FakeModel()
    model.lm_head = FakeHead()
    batch = {
        "input_ids": torch.cat((torch.full((1, 375), 99, dtype=torch.long),
                                 torch.zeros((1, 21), dtype=torch.long)), dim=1),
        "input_features": torch.linspace(-1, 1, 1500 * 1280).reshape(1, 1500, 1280),
    }
    return model, batch


def fake_details() -> dict:
    sources = {
        "encoder31": torch.zeros(target.SOURCE_SHAPES["encoder31"], dtype=torch.bfloat16),
        "projector": torch.zeros(target.SOURCE_SHAPES["projector"], dtype=torch.bfloat16),
    }
    lm_input = torch.zeros(target.LM_INPUT_SHAPE, dtype=torch.bfloat16)
    logits = torch.zeros((target.LOGIT_WIDTH,), dtype=torch.bfloat16)
    positions = list(range(375))
    observations = {}
    for site in target.SOURCE_REPLAY_STAGES:
        records = []
        for cycle in range(3):
            alphas = target.ALPHAS if cycle != 1 else tuple(reversed(target.ALPHAS))
            for alpha in alphas:
                edited_source = sources[site].clone()
                if site == "encoder31":
                    edited_source[0, 96, 0] = torch.tensor(alpha, dtype=torch.bfloat16)
                else:
                    edited_source[24, 0] = torch.tensor(alpha, dtype=torch.bfloat16)
                def trace(source):
                    return {"saved": {"source_full_post": {site: source},
                                      "lm_input_full": lm_input,
                                      "first_answer_logits": logits,
                                      "audio_placeholder_positions": positions}}
                records.append({"cycle": cycle, "alpha": alpha,
                                "zero": trace(sources[site]), "edited": trace(edited_source)})
        observations[site] = records
    return {"observations": observations}


class StageBReplayTests(unittest.TestCase):
    def test_protocol_and_replay_plan_are_fixed(self):
        target.require_protocol()
        self.assertEqual(len(target.planned_source_vjps("encoder31")), 36)
        self.assertEqual(len(target.planned_source_vjps("projector")), 32)
        self.assertEqual(target.planned_source_vjps("encoder31")[:4],
                         tuple(("projector_word", i) for i in range(4)))
        self.assertEqual(target.planned_source_vjps("projector")[:4],
                         tuple(("lm_input_word", i) for i in range(4)))
        self.assertEqual(target.SOURCE_REPLAY_FORWARDS, 24)
        self.assertEqual(target.STAGE_B_CAP_SECONDS,
                         {"bf16_direct_lm": 600, "fp32_replay": 1800})
        with self.assertRaises(ValueError):
            target.planned_source_vjps("lm29")

    def test_replay_caps_fail_at_fixed_boundaries(self):
        target.check_replay_limits("bf16_direct_lm", elapsed_seconds=599,
                                   previous_model_work_seconds=2000,
                                   mps_driver_bytes=100, output_bytes=100)
        with self.assertRaises(TimeoutError):
            target.check_replay_limits("bf16_direct_lm", elapsed_seconds=600,
                                       previous_model_work_seconds=0,
                                       mps_driver_bytes=0, output_bytes=0)
        with self.assertRaises(TimeoutError):
            target.check_replay_limits("fp32_replay", elapsed_seconds=500,
                                       previous_model_work_seconds=4000,
                                       mps_driver_bytes=0, output_bytes=0)
        with self.assertRaises(MemoryError):
            target.check_replay_limits("fp32_replay", elapsed_seconds=1,
                                       previous_model_work_seconds=0,
                                       mps_driver_bytes=target.MAX_MPS_DRIVER_BYTES,
                                       output_bytes=0)
        with self.assertRaises(OSError):
            target.check_replay_limits("fp32_replay", elapsed_seconds=1,
                                       previous_model_work_seconds=0,
                                       mps_driver_bytes=0,
                                       output_bytes=target.STAGE_B_OUTPUT_CAP_BYTES)

    def test_cycle_zero_full_tensor_binding_and_output_budget(self):
        details = fake_details()
        pairs = target.cycle_zero_pairs(details)
        self.assertEqual(len(pairs), 12)
        self.assertEqual([(p["site"], p["alpha"]) for p in pairs[:6]],
                         [("encoder31", alpha) for alpha in target.ALPHAS])
        estimate = target.estimate_replay_payload_bytes(details)
        self.assertGreater(estimate["estimated_tensor_payload_bytes"], 100_000_000)
        self.assertLess(estimate["estimated_tensor_payload_bytes"],
                        target.STAGE_B_OUTPUT_CAP_BYTES)
        bad = copy.deepcopy(details)
        bad["observations"]["encoder31"][0]["edited"]["saved"]["source_full_post"]["encoder31"][0, 20, 0] = 1
        with self.assertRaisesRegex(ValueError, "outside frozen word rows"):
            target.cycle_zero_pairs(bad)
        bad = fake_details()
        bad["observations"]["projector"][0]["edited"]["saved"]["lm_input_full"] = torch.zeros(
            target.LM_INPUT_SHAPE, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "LM input"):
            target.cycle_zero_pairs(bad)
        bad = fake_details()
        bad["observations"]["encoder31"][0]["alpha"] = 0.2
        with self.assertRaisesRegex(ValueError, "signed-alpha"):
            target.cycle_zero_pairs(bad)

    def test_native_attempt_requires_external_audit_and_exact_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp) / "native"
            attempt.mkdir()
            blobs = attempt / "blobs"
            blobs.mkdir()
            probes = native.fixed_probes()
            details = fake_details()
            details["probes"] = probes
            probes_tmp = blobs / "probes.tmp"
            details_tmp = blobs / "details.tmp"
            torch.save(probes, probes_tmp)
            torch.save(details, details_tmp)
            for name in ("meta.json", "model_layout.json", "progress.jsonl"):
                (attempt / name).write_text("{}\n")
            probes_sha = target.sha_file(probes_tmp)
            details_sha = target.sha_file(details_tmp)
            probes_path = blobs / f"{probes_sha}.pt"
            details_path = blobs / f"{details_sha}.pt"
            probes_tmp.rename(probes_path)
            details_tmp.rename(details_path)
            result = {
                "status": "completed_old_fit_handoff_stage_a_native_only",
                "arm": "bf16", "protocol_sha256": target.PROTOCOL_SHA256,
                "parity_passed": True,
                "probes_bytes": probes_path.stat().st_size,
                "details_bytes": details_path.stat().st_size,
                "probes_path": f"blobs/{probes_sha}.pt",
                "details_path": f"blobs/{details_sha}.pt",
            }
            for key, name in (("meta_sha256", "meta.json"),
                              ("model_layout_sha256", "model_layout.json"),
                              ("progress_sha256", "progress.jsonl")):
                result[key] = target.sha_file(attempt / name)
            result["probes_sha256"] = probes_sha
            result["details_sha256"] = details_sha
            (attempt / "result.json").write_text(json.dumps(result))
            audit = {"status": "audited_completed_old_fit_handoff_stage_a_native_only",
                     "arm": "bf16", "protocol_sha256": target.PROTOCOL_SHA256,
                     "result_sha256": target.sha_file(attempt / "result.json"),
                     "details_sha256": result["details_sha256"]}
            audit_path = Path(tmp) / "independent_audit.json"
            audit_path.write_text(json.dumps(audit))
            pinned = target.sha_file(audit_path)
            loaded_result, loaded_details = target.load_audited_native_attempt(
                attempt, audit_path, expected_audit_sha256=pinned)
            self.assertEqual(loaded_result["arm"], "bf16")
            self.assertEqual(len(target.cycle_zero_pairs(loaded_details)), 12)
            with self.assertRaisesRegex(ValueError, "audit file"):
                target.load_audited_native_attempt(
                    attempt, audit_path, expected_audit_sha256="0" * 64)
            result["details_path"] = "../details.pt"
            (attempt / "result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "content-addressed path"):
                target.load_audited_native_attempt(
                    attempt, audit_path, expected_audit_sha256=pinned)
            result["details_path"] = f"blobs/{details_sha}.pt"
            (attempt / "result.json").write_text(json.dumps(result))
            (attempt / "meta.json").write_text('{"changed":true}\n')
            with self.assertRaisesRegex(ValueError, "meta.json"):
                target.load_audited_native_attempt(
                    attempt, audit_path, expected_audit_sha256=pinned)

    def test_direct_lm_and_full_model_override_have_exact_fake_parity(self):
        model, batch = fake_model_and_batch()
        lm_input = torch.arange(396 * 3072, dtype=torch.float32).reshape(target.LM_INPUT_SHAPE) / 10000
        direct = target.direct_lm_forward(model, batch, lm_input, torch.float32, device="cpu")
        full = target.full_model_lm_override(model, batch, lm_input, torch.float32, device="cpu")
        target.assert_direct_parity(full, direct)
        self.assertEqual(direct["input_sha256"], full["input_sha256"])
        self.assertEqual(direct["first_answer_logits"].shape, (target.LOGIT_WIDTH,))
        altered = copy.deepcopy(full)
        altered["lm_terminal"]["lm14"][0] += 1
        with self.assertRaisesRegex(ValueError, "LM14|lm14"):
            target.assert_direct_parity(altered, direct)
        def wrong_dtype(_module, args, kwargs):
            changed = dict(kwargs)
            changed["inputs_embeds"] = kwargs["inputs_embeds"].to(torch.bfloat16)
            return args, changed
        handle = model.model.language_model.register_forward_pre_hook(wrong_dtype, with_kwargs=True)
        try:
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                target.full_model_lm_override(model, batch, lm_input, torch.float32, device="cpu")
        finally:
            handle.remove()

    def test_full_source_override_connects_expected_fake_gradients(self):
        model, batch = fake_model_and_batch()
        for site in ("encoder31", "projector"):
            source = (batch["input_features"] if site == "encoder31" else
                      model.model.multi_modal_projector(batch["input_features"])).to(torch.bfloat16)
            replay = target.source_replay_forward(
                model, batch, site=site, full_source=source, probes=native.fixed_probes(),
                grad=True, device="cpu")
            self.assertEqual(replay["saved"]["first_answer_logits"].shape, (target.LOGIT_WIDTH,))
            stage = "projector_word" if site == "encoder31" else "lm_input_word"
            scalar = replay["graph_targets"][stage][0]
            gradient, = torch.autograd.grad(scalar, replay["graph_source"])
            self.assertTrue(bool((gradient != 0).any()))
            self.assertEqual(tuple(gradient.shape), target.SOURCE_SHAPES[site])
            self.assertEqual(replay["saved"]["audio_placeholder_positions"], list(range(375)))
            del replay, gradient

    def test_source_hook_rejects_wrong_dtype_and_duplicate_invocation(self):
        model, batch = fake_model_and_batch()
        source = batch["input_features"].to(torch.bfloat16)
        wrong = dict(batch)
        wrong["input_features"] = source
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            target.source_replay_forward(model, wrong, site="encoder31",
                                         full_source=source, probes=native.fixed_probes(),
                                         device="cpu")

        class DuplicateSourceModel(FakeModel):
            def forward(self, *, input_ids, input_features, **kwargs):
                self.model.audio_tower.layers[31](input_features)
                return super().forward(input_ids=input_ids, input_features=input_features, **kwargs)

        duplicate = DuplicateSourceModel()
        duplicate.lm_head = FakeHead()
        with self.assertRaisesRegex(ValueError, "hook count"):
            target.source_replay_forward(duplicate, batch, site="encoder31",
                                         full_source=source, probes=native.fixed_probes(),
                                         device="cpu")

    def test_exact_source_replay_vjp_plan_runs_on_fake_graph(self):
        model, batch = fake_model_and_batch()
        probes = native.fixed_probes()
        for site in ("encoder31", "projector"):
            source = (batch["input_features"] if site == "encoder31" else
                      model.model.multi_modal_projector(batch["input_features"])).to(torch.bfloat16)
            replay = target.source_replay_forward(
                model, batch, site=site, full_source=source, probes=probes,
                grad=True, device="cpu")
            gradients = target.replay_source_vjps(replay, probes, site)
            self.assertEqual(sum(value.shape[0] for value in gradients.values()),
                             target.SOURCE_REPLAY_VJPS[site])
            self.assertEqual(set(gradients),
                             {stage for stage, _ in target.planned_source_vjps(site)})
            self.assertTrue(all(torch.isfinite(value).all() and value.dtype == torch.float32
                                for value in gradients.values()))
            self.assertTrue(bool((gradients["projector_word" if site == "encoder31"
                                            else "lm_input_word"] != 0).any()))

    def test_pair_verifier_rejects_unselected_mutation(self):
        model, batch = fake_model_and_batch()
        source = batch["input_features"].to(torch.bfloat16)
        changed = source.clone()
        changed[0, 96, 0] += torch.tensor(0.25, dtype=torch.bfloat16)
        probes = native.fixed_probes()
        zero = target.source_replay_forward(model, batch, site="encoder31",
                                            full_source=source, probes=probes, device="cpu")
        edited = target.source_replay_forward(model, batch, site="encoder31",
                                              full_source=changed, probes=probes, device="cpu")
        measured = target.verify_source_replay_pair(zero, edited, source, changed, "encoder31")
        self.assertGreater(measured["source_edit_l2"], 0)
        bad = changed.clone()
        bad[0, 20, 0] += torch.tensor(0.25, dtype=torch.bfloat16)
        bad_replay = target.source_replay_forward(model, batch, site="encoder31",
                                                  full_source=bad, probes=probes, device="cpu")
        with self.assertRaisesRegex(ValueError, "beyond word rows"):
            target.verify_source_replay_pair(zero, bad_replay, source, bad, "encoder31")


if __name__ == "__main__":
    unittest.main()
