"""No-weight fake-hook and evidence-gate tests for the padding worker."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from models.second_architecture.nonspeech_probe_v1 import padding_channel_plan_v4 as plan
from models.second_architecture.nonspeech_probe_v1 import padding_channel_worker_v1 as worker
from models.second_architecture.nonspeech_probe_v1 import padding_channel_core_v1 as core


class FakeHandle:
    def __init__(self, callbacks: list, callback):
        self.callbacks, self.callback = callbacks, callback

    def remove(self):
        self.callbacks.remove(self.callback)


class FakeModule:
    def __init__(self):
        self.before, self.after = [], []

    def register_forward_pre_hook(self, callback):
        self.before.append(callback)
        return FakeHandle(self.before, callback)

    def register_forward_hook(self, callback):
        self.after.append(callback)
        return FakeHandle(self.after, callback)

    def __call__(self, value):
        args = (value,)
        for callback in self.before:
            replacement = callback(self, args)
            if replacement is not None:
                args = replacement
        result = args[0]
        for callback in self.after:
            replacement = callback(self, args, result)
            if replacement is not None:
                result = replacement
        return result


class FakeModel:
    def __init__(self):
        encoder = SimpleNamespace(conv1=FakeModule(), layers=[FakeModule()])
        self.model = SimpleNamespace(encoder=encoder)
        self.generation_config = SimpleNamespace(lang_to_id={"en": 50259})
        self.after = []
        self.seen = []

    def register_forward_hook(self, callback):
        self.after.append(callback)
        return FakeHandle(self.after, callback)

    def run_encoder(self):
        mel = self.model.encoder.conv1(torch.zeros((1, 80, 3000), dtype=torch.float32))
        early = self.model.encoder.layers[0](torch.zeros((1, 1500, 384), dtype=torch.float32))
        self.seen.append((mel.detach().numpy().copy(), early.detach().numpy().copy()))

    def detect_language(self):
        self.run_encoder()
        logits = torch.zeros((1, 1, 51865), dtype=torch.float32)
        output = SimpleNamespace(logits=logits)
        for callback in self.after:
            callback(self, (), output)
        return torch.tensor([50259])


def frozen_row(site: str, arm: str, case_id: str = "d02_beep_train_low") -> dict:
    return next(row for row in plan.schedule() if row["site"] == site
                and row["case_id"] == case_id and row["arm"] == arm)


class WorkerFakeTests(unittest.TestCase):
    def states(self, case_id: str) -> dict:
        donor = plan.donor_for(case_id)
        return {name: {site: {phase: np.full(core.expected_shape(site), fill,
                                               np.float32) for phase in plan.PHASES}
                       for site in plan.SITES}
                for name, fill in ((case_id, 0.0), (donor, 1.0))}

    def test_two_phase_two_site_capture_and_cleanup(self):
        model = FakeModel()
        row = frozen_row("log_mel", "read_only")
        hooks = worker.PhaseHooks(model, row, {})
        hooks.install()
        model.detect_language()
        model.run_encoder()
        hooks.validated()
        self.assertEqual(hooks.counts, {
            phase: {site: 1 for site in plan.SITES} for phase in plan.PHASES})
        self.assertTrue(all(item["support"] == [0, 0] for phase in plan.PHASES
                            for item in hooks.hook[phase].values()))
        hooks.close()
        self.assertEqual(model.model.encoder.conv1.before, [])
        self.assertEqual(model.model.encoder.layers[0].before, [])
        self.assertEqual(model.after, [])

    def test_log_mel_padding_and_phase_split(self):
        for arm in ("RD", "padding_detect_only", "padding_transcribe_only"):
            model = FakeModel()
            row = frozen_row("log_mel", arm)
            states = self.states(row["case_id"])
            donor_before = states[row["donor_id"]]["log_mel"]["detect"].copy()
            hooks = worker.PhaseHooks(model, row, states)
            hooks.install()
            model.detect_language()
            model.run_encoder()
            hooks.validated()
            for phase, (mel, early) in zip(plan.PHASES, model.seen, strict=True):
                should_change = arm == "RD" or arm == "padding_detect_only" and phase == "detect" \
                    or arm == "padding_transcribe_only" and phase == "transcribe"
                self.assertTrue(np.all(mel[:, :, :200] == 0))
                self.assertTrue(np.all(mel[:, :, 200:] ==
                                       (1 if should_change else 0)))
                self.assertTrue(np.all(early == 0))
            self.assertTrue(np.array_equal(
                states[row["donor_id"]]["log_mel"]["detect"], donor_before))
            hooks.close()

    def test_preblock0_axis_and_duplicate_hook_fail(self):
        model = FakeModel()
        row = frozen_row("pre_block0", "DR", "d08_beep_train_high")
        hooks = worker.PhaseHooks(model, row, self.states(row["case_id"]))
        hooks.install()
        model.detect_language()
        model.run_encoder()
        hooks.validated()
        for mel, early in model.seen:
            self.assertTrue(np.all(mel == 0))
            self.assertTrue(np.all(early[:, :400, :] == 1))
            self.assertTrue(np.all(early[:, 400:, :] == 0))
        with self.assertRaisesRegex(ValueError, "repeated"):
            model.run_encoder()
        hooks.close()

    def test_score_mask_and_decoder_shape(self):
        a = np.array([0.0, -np.inf, 0.2], np.float32)
        b = np.array([0.0, -np.inf, 0.200001], np.float32)
        self.assertLess(core.processed_score_maxabs(a, b), 1e-5)
        with self.assertRaisesRegex(ValueError, "suppression mask"):
            core.processed_score_maxabs(a, np.array([0.0, -2.0, 0.2], np.float32))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            budget = worker.Budget(root)
            good = worker.save_array(root, "processed/good.npy", a, budget,
                                     processed=True)
            self.assertEqual(good["shape"], [3])
            with self.assertRaisesRegex(ValueError, "nonfinite raw"):
                worker.save_array(root, "raw/bad.npy", a, budget)
            decoder = worker.save_array(root, "decoder/first.npy",
                                        np.zeros((1, 4, 384), np.float32), budget)
            self.assertEqual(decoder["shape"], [1, 4, 384])

    def test_first_step_array_rejection(self):
        raw = np.zeros(plan.VOCAB_SIZE, np.float32)
        processed = raw.copy()
        processed[3] = -np.inf
        decoder = np.zeros((1, 4, 384), np.float32)
        worker.validate_first_step_arrays(raw, processed, decoder)
        for bad in (np.nan, np.inf):
            changed = processed.copy()
            changed[3] = bad
            with self.assertRaisesRegex(ValueError, "processed score"):
                worker.validate_first_step_arrays(raw, changed, decoder)
        changed_raw = raw.copy()
        changed_raw[0] = -np.inf
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            worker.validate_first_step_arrays(changed_raw, processed, decoder)
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            worker.validate_first_step_arrays(raw, processed, decoder[:, :1, :])
        changed_decoder = decoder.copy()
        changed_decoder[0, 2, 5] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            worker.validate_first_step_arrays(raw, processed, changed_decoder)

    def test_identical_text_cannot_hide_score_mask_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            budget = worker.Budget(root)
            raw = np.zeros(plan.VOCAB_SIZE, np.float32)
            first = raw.copy()
            first[5] = -np.inf
            second = raw.copy()
            common = {"output": {"sequence_ids": [1, 2, 3, 4, 5],
                                 "actual_generation_steps": 1,
                                 "termination": "eos", "prefix_ids": [1, 2, 3, 4],
                                 "transcript": "same text"}}
            raw_ref = worker.save_array(root, "raw/a.npy", raw, budget)
            decoder_ref = worker.save_array(root, "decoder/a.npy",
                                            np.zeros((1, 4, 384), np.float32), budget)
            a = {**common, "sidecars": {
                "first_raw": raw_ref, "first_decoder": decoder_ref,
                "first_processed": worker.save_array(root, "processed/a.npy",
                                                     first, budget, processed=True)}}
            b = {**common, "sidecars": {
                "first_raw": raw_ref, "first_decoder": decoder_ref,
                "first_processed": worker.save_array(root, "processed/b.npy",
                                                     second, budget, processed=True)}}
            with self.assertRaisesRegex(ValueError, "suppression mask mismatch"):
                worker.first_parity(a, b, root)

    def test_budget_reserve_rejects_forward(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            budget = worker.Budget(root)
            from unittest.mock import patch
            with patch.object(worker, "MAX_EVIDENCE_BYTES",
                              worker.EVIDENCE_RESERVE_BYTES - 1):
                with self.assertRaisesRegex(ValueError, "reserve"):
                    budget.check(before_forward=True)


if __name__ == "__main__":
    unittest.main()
