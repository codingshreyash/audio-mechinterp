"""CPU-only source tests for the draft old-FIT native handoff primitives."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

import voxtral_old_fit_handoff_native_stage_a_v1 as target
import voxtral_old_fit_handoff_stage_a_runner_v1 as runner


ROWS = {"encoder31": list(range(96, 108)), "projector": [24, 25, 26]}


class FakeProjector(nn.Module):
    def forward(self, x):
        rows = x[0, ::4]
        return torch.cat((rows, rows, rows[:, :512]), dim=-1)


class FakeLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Identity() for _ in range(30))
        self.norm = nn.Identity()

    def forward(self, *, inputs_embeds, **_kwargs):
        state = inputs_embeds
        for layer in self.layers:
            state = layer(state)
        return self.norm(state)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.config = SimpleNamespace(audio_token_id=99)
        self.model.audio_tower = nn.Module()
        self.model.audio_tower.layers = nn.ModuleList(nn.Identity() for _ in range(32))
        self.model.multi_modal_projector = FakeProjector()
        self.model.language_model = FakeLanguage()

    def forward(self, *, input_ids, input_features, **_kwargs):
        audio = input_features
        for layer in self.model.audio_tower.layers:
            audio = layer(audio)
        projected = self.model.multi_modal_projector(audio)
        embedded = torch.zeros((1, 396, 3072), dtype=projected.dtype, device=projected.device)
        embedded = embedded.masked_scatter((input_ids == 99).unsqueeze(-1).expand_as(embedded), projected)
        state = self.model.language_model(inputs_embeds=embedded)
        logits = torch.zeros((1, 1, 131072), dtype=state.dtype, device=state.device)
        logits[0, 0, 0] = state[0, -1, 0]
        return SimpleNamespace(logits=logits)


class NativeStageATests(unittest.TestCase):
    def test_probe_build_is_deterministic_and_orthonormal(self):
        a = target.fixed_probes()
        b = target.fixed_probes()
        self.assertEqual(set(a), {"lm", "projector_word", "lm_input_word"})
        for name, matrix in a.items():
            self.assertTrue(torch.equal(matrix, b[name]))
            self.assertEqual(matrix.dtype, torch.float64)
            self.assertTrue(torch.allclose(matrix.T @ matrix,
                                           torch.eye(matrix.shape[1], dtype=torch.float64),
                                           atol=1e-12, rtol=0))
        self.assertEqual(a["lm"].shape, (3072, 16))
        self.assertEqual(a["projector_word"].shape, (9216, 4))

    def test_vjp_plan_and_archive_bounds(self):
        plan = target.planned_vjps()
        self.assertEqual(len(plan), 44)
        self.assertEqual(plan[:4], tuple(("projector_word", i) for i in range(4)))
        self.assertEqual(plan[-4:], tuple(("lm29_raw", i) for i in range(4)))
        self.assertEqual(sum(stage == "lm29" for stage, _ in plan), 16)
        for arm in ("bf16", "fp32"):
            estimate = target.estimate_native_archive_bytes(arm)
            self.assertLess(estimate["estimated_tensor_payload_bytes"],
                            estimate["draft_hard_archive_cap_bytes"])
            self.assertGreater(estimate["estimated_tensor_payload_bytes"], 600_000_000)
            self.assertEqual(estimate["source_effective_edit_bytes"], 3_538_944)
            self.assertEqual(estimate["probe_float64_bytes"], 1_966_080)
            self.assertEqual(estimate["estimated_tensor_payload_bytes"],
                             sum(value for key, value in estimate.items()
                                 if key.endswith("_bytes") and key not in (
                                     "estimated_tensor_payload_bytes", "draft_hard_archive_cap_bytes")))
        self.assertGreater(target.estimate_native_archive_bytes("fp32")["estimated_tensor_payload_bytes"],
                           target.estimate_native_archive_bytes("bf16")["estimated_tensor_payload_bytes"])
        self.assertLess(target.estimate_native_archive_bytes("fp32")["estimated_tensor_payload_bytes"],
                        .9 * target.ARCHIVE_CAP_BYTES["fp32"])

    def test_cpu_fake_hooks_patch_handoff_and_grad_path(self):
        model = FakeModel()
        batch = {"input_ids": torch.cat((torch.full((1, 375), 99), torch.zeros((1, 21), dtype=torch.long)), dim=1),
                 "input_features": torch.linspace(-1, 1, 1500 * 1280).reshape(1, 1500, 1280)}
        clean = target.trace_forward(model, batch, torch.float32, ROWS, device="cpu", grad=True,
                                     archive_full_source=True, capture_final_full=True)
        self.assertEqual(set(clean["graph_targets"]), set(target.STAGE_ORDER))
        self.assertEqual(clean["saved"]["lm_input_full"].shape, (1, 396, 3072))
        self.assertEqual(clean["saved"]["first_answer_logits"].shape, (131072,))
        self.assertTrue(torch.equal(clean["saved"]["final_full"][0, -1],
                                     clean["saved"]["lm_terminal"]["lm29"]))
        self.assertEqual(clean["saved"]["source_full_pre"]["encoder31"].shape, (1, 1500, 1280))
        self.assertEqual(clean["saved"]["source_full_pre_hashes"]["encoder31"],
                         clean["saved"]["source_full_post_hashes"]["encoder31"])
        self.assertEqual(clean["saved"]["audio_placeholder_positions"], list(range(375)))
        self.assertEqual(clean["saved"]["hook_events"], clean["events"])
        self.assertEqual(clean["saved"]["hook_events"]["lm_input"],
                         {"calls": 1, "shape": [1, 396, 3072], "dtype": "torch.float32"})
        self.assertEqual(clean["saved"]["hook_events"]["final_norm"]["calls"], 1)
        self.assertTrue(torch.equal(clean["saved"]["projector_full"],
                                    clean["saved"]["lm_input_full"][0, :375]))
        gradient = torch.autograd.grad(clean["graph_targets"]["lm29"][0],
                                       (clean["graph_sources"]["encoder31"],
                                        clean["graph_sources"]["projector"]))
        self.assertEqual(gradient[0].shape, (1, 1500, 1280))
        self.assertEqual(gradient[1].shape, (375, 3072))
        del clean, gradient
        direction = torch.ones(1280, dtype=torch.float32) * 0.125
        edited = target.trace_forward(model, batch, torch.float32, ROWS, device="cpu",
                                      patch_site="encoder31", direction=direction,
                                      alpha=0.1, archive_full_source=True)
        self.assertGreater(float(edited["saved"]["source_effective_edit"].norm()), 0)
        self.assertFalse(torch.equal(edited["saved"]["source_full_pre"]["encoder31"],
                                     edited["saved"]["source_full_post"]["encoder31"]))
        self.assertNotEqual(edited["saved"]["source_full_pre_hashes"]["encoder31"],
                            edited["saved"]["source_full_post_hashes"]["encoder31"])
        self.assertEqual(edited["saved"]["source_edit_metrics"]["kind"], "alpha")
        zero = target.trace_forward(model, batch, torch.float32, ROWS, device="cpu",
                                    patch_site="encoder31", direction=direction, alpha=0.0)
        self.assertEqual(float(zero["saved"]["source_effective_edit"].abs().sum()), 0.0)

    def test_contract_and_reject_wrong_shapes(self):
        gradients = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        edit = torch.ones((3, 4), dtype=torch.float32)
        self.assertTrue(torch.equal(target.contract_vjps(gradients, edit),
                                    gradients.sum((1, 2))))
        with self.assertRaises(ValueError):
            target.contract_vjps(gradients, torch.ones((4, 3)))

    def test_lm0_control_scale_and_compact_hook(self):
        unit = target.fixed_lm0_control_unit()
        self.assertAlmostEqual(float(torch.linalg.vector_norm(unit)), 1, delta=1e-5)
        clean_state = torch.zeros((3072,), dtype=torch.bfloat16)
        choice = target.choose_lm0_control_scale(clean_state, unit, 1, 0.125)
        self.assertTrue(choice["valid"])
        self.assertLessEqual(choice["relative_match_error"], .05)
        self.assertEqual(target.choose_lm0_control_scale(clean_state, unit, -1, 0)["reason"],
                         "zero_or_nonfinite_target")
        model = FakeModel()
        batch = {"input_ids": torch.cat((torch.full((1, 375), 99), torch.zeros((1, 21), dtype=torch.long)), dim=1),
                 "input_features": torch.zeros((1, 1500, 1280))}
        zero = target.compact_lm0_control_forward(model, batch, torch.float32, device="cpu",
                                                   scalar=0, unit=unit)
        edited = target.compact_lm0_control_forward(model, batch, torch.float32, device="cpu",
                                                     scalar=.125, unit=unit)
        self.assertEqual(float(zero["effective_edit_lm0"].abs().sum()), 0.0)
        self.assertTrue(torch.equal(zero["prepatch_lm0"], edited["prepatch_lm0"]))
        self.assertFalse(torch.equal(zero["lm_terminal"]["lm29"], edited["lm_terminal"]["lm29"]))
        self.assertEqual(set(edited["lm_terminal"]), {f"lm{i}" for i in target.LM_LAYERS})
        self.assertEqual(edited["hook_events"]["lm29"]["calls"], 1)

    def test_full_boundary_descriptors_and_each_cycle_odd_even(self):
        saved = {"projector_full": torch.zeros((375, 3072)),
                 "lm_input_full": torch.zeros((1, 396, 3072)),
                 "lm_terminal": {"lm29": torch.zeros(3072)},
                 "audio_placeholder_positions": list(range(375))}
        saved["projector_full"][0, 0] = 1
        saved["lm_input_full"][0, 0, 0] = 2
        self.assertEqual(target.full_stage_vector(saved, "projector_word", ROWS).numel(), 375 * 3072)
        self.assertEqual(target.full_stage_vector(saved, "lm_input_word", ROWS).numel(), 396 * 3072)
        self.assertEqual(float(target.stage_vector(saved, "projector_word", ROWS).abs().sum()), 0)
        self.assertEqual(float(target.full_stage_vector(saved, "projector_word", ROWS).abs().sum()), 1)
        self.assertEqual(float(target.full_stage_vector(saved, "lm_input_word", ROWS).abs().sum()), 2)
        rows = []
        for cycle in range(3):
            for magnitude in (.00625, .025, .1):
                rows.append({"cycle": cycle, "alpha": magnitude,
                             "measured": [10 * cycle + magnitude]})
                rows.append({"cycle": cycle, "alpha": -magnitude,
                             "measured": [-2 * cycle - magnitude]})
        paired = target.cycle_odd_even(rows)
        self.assertEqual(len(paired["0.1"]["cycles"]), 3)
        self.assertAlmostEqual(paired["0.1"]["cycles"]["2"]["measured_odd"][0], 12.1)
        self.assertAlmostEqual(paired["0.1"]["cycles"]["2"]["measured_even"][0], 8)

    def test_content_addressed_sidecar_and_hard_write_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "details.pt"
            with source.open("xb") as raw:
                limited = runner.LimitedWriter(raw, 4096)
                torch.save({"small": torch.arange(8)}, limited)
            digest, relative = runner.freeze_content_addressed(source)
            self.assertEqual(relative, f"blobs/{digest}.pt")
            self.assertFalse(source.exists())
            self.assertEqual(runner.fixture.sha(root / relative), digest)
            self.assertEqual(runner.directory_bytes(root), (root / relative).stat().st_size)
            with (root / "too_large.pt").open("xb") as raw:
                limited = runner.LimitedWriter(raw, 512)
                with self.assertRaises((OSError, RuntimeError)):
                    torch.save({"large": torch.zeros(512)}, limited)
                self.assertTrue(limited.hit_cap)


if __name__ == "__main__":
    unittest.main()
