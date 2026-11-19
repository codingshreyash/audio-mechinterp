"""No-weight fake complete generation path and input-feature evidence tests."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from models.second_architecture.nonspeech_probe_v1 import padding_channel_plan_v5 as plan
from models.second_architecture.nonspeech_probe_v1 import padding_channel_worker_v2 as worker
from models.second_architecture.nonspeech_probe_v1 import padding_channel_core_v1 as core
from models.second_architecture.nonspeech_probe_v1.test_padding_channel_worker_v1 import FakeModel


class FakeTokenizer:
    eos_token_id = pad_token_id = 50257

    def decode(self, ids, skip_special_tokens=False):
        return " word" if 100 in ids else " other" if 101 in ids else ""


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def __init__(self, active=200):
        self.active = active

    def __call__(self, audio, **_kwargs):
        value = 1.0 if float(np.max(np.abs(audio))) > 0 else 0.0
        features = torch.full((1, 80, 3000), value, dtype=torch.float32)
        mask = torch.tensor([[1] * self.active + [0] * (3000-self.active)],
                            dtype=torch.long)
        return SimpleNamespace(input_features=features, attention_mask=mask)


class FullFakeModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.generation_config = SimpleNamespace(
            lang_to_id={"en": 50259}, is_multilingual=True, language=None,
            decoder_start_token_id=50258, task_to_id={"transcribe": 50359},
            no_timestamps_token_id=50363)

    def run_encoder(self, input_features):
        mel = self.model.encoder.conv1(input_features)
        level = float(mel.mean())
        early_input = torch.full((1, 1500, 384), level, dtype=torch.float32)
        early = self.model.encoder.layers[0](early_input)
        self.latest = (mel, early)

    def detect_language(self, input_features=None):
        self.run_encoder(input_features)
        logits = torch.zeros((1, 1, 51865), dtype=torch.float32)
        output = SimpleNamespace(logits=logits)
        for callback in self.after:
            callback(self, (), output)
        return torch.tensor([50259])

    def generate(self, input_features=None, **_kwargs):
        self.detect_language(input_features=input_features)
        self.run_encoder(input_features)
        token = 100 if float(self.latest[1].mean()) > 0.5 else 101
        prefix = [50258, 50259, 50359, 50363]
        ids = torch.tensor([prefix + [token, 50257]], dtype=torch.long)
        raw0 = torch.full((1, 51865), -10.0, dtype=torch.float32)
        raw0[0, token] = 10.0
        raw1 = torch.full((1, 51865), -10.0, dtype=torch.float32)
        raw1[0, 50257] = 10.0
        score0, score1 = raw0.clone(), raw1.clone()
        score0[0, 42] = -torch.inf
        hidden = torch.zeros((1, 4, 384), dtype=torch.float32)
        return SimpleNamespace(sequences=ids, scores=(score0, score1),
                               logits=(raw0, raw1),
                               decoder_hidden_states=((hidden,), (hidden,)))


def row(case: str, site: str, arm: str) -> dict:
    return next(item for item in plan.schedule() if item["case_id"] == case
                and item["site"] == site and item["arm"] == arm)


class WorkerV2FakeTests(unittest.TestCase):
    def test_complete_nohook_capture_dd_and_feature_dedup(self):
        donor_id, case_id = "d02_silence", "d02_beep_train_low"
        cases = {donor_id: {"case_id": donor_id,
                            "pcm": np.zeros(32000, np.int16)},
                 case_id: {"case_id": case_id,
                           "pcm": np.ones(32000, np.int16)}}
        model, processor = FullFakeModel(), FakeProcessor()
        states, feature_refs, feature_values = {}, {}, {}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            budget = worker.Budget(output)

            def call(plan_row, index):
                return worker.generate_one(plan_row, cases[plan_row["case_id"]],
                    processor, model, states, output, budget, index,
                    feature_refs, feature_values)

            donor = call(row(donor_id, "none", "no_hook"), 1)
            self.assertEqual(donor["hook"], {})
            donor_read = call(row(donor_id, "both", "read_only_donor_capture"), 2)
            states[donor_id] = {site: {phase: donor_read["native"][phase][site]
                                       for phase in plan.PHASES}
                                for site in plan.SITES}
            self.assertTrue(worker.first_parity(donor, donor_read, output)["within_1e-5"])
            rr = call(row(case_id, "log_mel", "RR"), 3)
            self.assertEqual(rr["hook"], {})
            read = call(row(case_id, "log_mel", "read_only"), 4)
            states[case_id] = {site: {phase: read["native"][phase][site]
                                     for phase in plan.PHASES}
                               for site in plan.SITES}
            self.assertTrue(worker.first_parity(rr, read, output)["tokens_match"])
            dd = call(row(case_id, "log_mel", "DD"), 5)
            self.assertTrue(worker.first_parity(dd, donor, output)["tokens_match"])
            self.assertTrue(worker.full_raw_parity(dd, donor, output)["within_1e-5"])
            self.assertEqual(len(feature_refs), 2)
            self.assertEqual(len(list((output / "input").glob("*.npy"))), 2)
            for observed in (rr, read, dd):
                self.assertEqual(observed["input_feature_ref"], feature_refs[case_id])
                self.assertEqual(observed["input_feature_sha256_before"],
                                 observed["input_feature_sha256_after"])
            repeat = call(row(case_id, "pre_block0", "read_only"), 6)
            self.assertTrue(all(np.array_equal(repeat["native"][phase][site],
                                               states[case_id][site][phase])
                                for phase in plan.PHASES for site in plan.SITES))

    def test_changed_mask_and_feature_fails_before_generation(self):
        case_id = "d02_beep_train_low"
        case = {"case_id": case_id, "pcm": np.ones(32000, np.int16)}
        model = FullFakeModel()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            budget = worker.Budget(output)
            with self.assertRaisesRegex(ValueError, "mask"):
                worker.generate_one(row(case_id, "log_mel", "RR"), case,
                    FakeProcessor(active=201), model, {}, output, budget, 1, {}, {})
            refs, values = {}, {}
            worker.generate_one(row(case_id, "log_mel", "RR"), case,
                FakeProcessor(), model, {}, output, budget, 2, refs, values)
            class ChangedProcessor(FakeProcessor):
                def __call__(self, audio, **kwargs):
                    result = super().__call__(audio, **kwargs)
                    result.input_features[0, 0, 0] += 1.0
                    return result
            with self.assertRaisesRegex(ValueError, "features changed"):
                worker.generate_one(row(case_id, "log_mel", "RR"), case,
                    ChangedProcessor(), model, {}, output, budget, 3, refs, values)

    def test_source_pin_drift_fails_without_model(self):
        with patch.dict(plan.SOURCE_SHA256,
                        {"feature_extraction_whisper.py": "0" * 64}):
            with self.assertRaisesRegex(ValueError, "source changed"):
                plan.verify_installed_source_bytes()


if __name__ == "__main__":
    unittest.main()
