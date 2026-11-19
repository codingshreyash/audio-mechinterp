"""Fake-only pinned loader tests; no real Voxtral weights or processor."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.second_architecture import voxtral_track_b_v6_pinned_loader_source_v1 as loader
from models.second_architecture import voxtral_track_b_v6_pre_pilot_launcher_source_v1 as admission_launcher
from models.second_architecture import voxtral_track_b_v6_pilot_worker_source_v1 as worker
from models.second_architecture.test_voxtral_track_b_v6_pre_pilot_contract_v1 import Fixture, write_json


class FakeMPS:
    def __init__(self):
        self.fraction = None

    def recommended_max_memory(self):
        return 40 * 1024**3

    def set_per_process_memory_fraction(self, fraction):
        self.fraction = fraction

    def driver_allocated_memory(self):
        return 1024

    def synchronize(self):
        return None


class FakeBackendMPS:
    @staticmethod
    def is_available():
        return True


class FakeTorch:
    float32 = 'float32'
    mps = FakeMPS()
    backends = type('Backends', (), {'mps': FakeBackendMPS()})()


class FakeTokenizer:
    eos_token_id = 2


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.audio_token_id = 24


class FakeAutoProcessor:
    calls = 0

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        cls.calls += 1
        return FakeProcessor()


class FakeParameter:
    dtype = 'float32'
    device = type('Device', (), {'type': 'mps'})()


class FakeModel:
    training = True
    generation_config = type('Config', (), {'eos_token_id': 2, 'pad_token_id': 0})()

    def eval(self):
        self.training = False
        return self

    def requires_grad_(self, value):
        self.grad_setting = value
        return self

    def parameters(self):
        return [FakeParameter()]


class FakeAutoModel:
    calls = 0
    kwargs = None

    @classmethod
    def from_pretrained(cls, *_args, **kwargs):
        cls.calls += 1
        cls.kwargs = kwargs
        return FakeModel()


class BadFakeAutoModel(FakeAutoModel):
    @classmethod
    def from_pretrained(cls, *_args, **kwargs):
        model = super().from_pretrained(*_args, **kwargs)
        bad = FakeParameter()
        bad.dtype = 'bfloat16'
        model.parameters = lambda: [bad]
        return model


class MutatingAutoModel(FakeAutoModel):
    @classmethod
    def from_pretrained(cls, snapshot, **kwargs):
        (Path(snapshot) / 'config.json').write_text('mutated during fake factory')
        return super().from_pretrained(snapshot, **kwargs)


def fake_snapshot_manifest(snapshot: Path) -> dict:
    return {'snapshot_tree': {'sha256': 'a' * 64},
            'config': {'sha256': worker.sha_file(snapshot / 'config.json')}}


def prepared_fixture(directory: Path) -> Fixture:
    f = Fixture(directory)
    snapshot = directory / 'fake_snapshot'
    snapshot.mkdir()
    (snapshot / 'config.json').write_text('pinned fake config')
    f.frozen['snapshot'] = fake_snapshot_manifest(snapshot)
    f.frozen['processor'] = {'processor_type': 'FakeProcessor',
                             'tokenizer_type': 'FakeTokenizer'}
    f.bindings['preflight_sha256'] = write_json(f.files.preflight, f.frozen)
    f.repin_admission_and_review()
    return f


class PinnedLoaderSourceTests(unittest.TestCase):
    def setUp(self):
        FakeAutoProcessor.calls = 0
        FakeAutoModel.calls = 0
        FakeAutoModel.kwargs = None
        source_patch = patch.object(loader, '_installed_source_map',
                                    return_value={'modeling_voxtral': 'b' * 64})
        source_patch.start()
        self.addCleanup(source_patch.stop)

    def test_fake_fp32_model_and_processor_after_bound_preflight(self):
        with tempfile.TemporaryDirectory() as temp:
            f = prepared_fixture(Path(temp))
            snapshot = Path(temp) / 'fake_snapshot'
            with patch.object(admission_launcher, '_actual_qa', return_value=f.qa), patch.object(
                    admission_launcher, '_fresh_no_weight_preflight', return_value=f.frozen), patch.object(
                    loader, '_pinned_snapshot_path', return_value=snapshot), patch.object(
                    loader, '_snapshot_manifest', side_effect=lambda: fake_snapshot_manifest(snapshot)):
                model, processor, evidence = loader._load_after_verified(
                    files=f.files, review_path=f.review_path, pins=f.pins,
                    torch_api=FakeTorch, snapshot=snapshot,
                    processor_factory=FakeAutoProcessor, model_factory=FakeAutoModel)
            self.assertIsInstance(model, FakeModel)
            self.assertIsInstance(processor, FakeProcessor)
            self.assertFalse(model.training)
            self.assertFalse(model.grad_setting)
            self.assertEqual((FakeAutoProcessor.calls, FakeAutoModel.calls), (1, 1))
            self.assertEqual(FakeAutoModel.kwargs['dtype'], FakeTorch.float32)
            self.assertEqual(FakeAutoModel.kwargs['device_map'], 'mps')
            self.assertEqual(evidence['allocator_cap']['configured_allocator_cap_bytes'], 30 * 1024**3)

    def test_bad_qa_fails_before_processor_or_model_factory(self):
        with tempfile.TemporaryDirectory() as temp:
            f = prepared_fixture(Path(temp))
            snapshot = Path(temp) / 'fake_snapshot'
            wrong_qa = {**f.qa, 'qa_go_receipt_sha256': '0' * 64}
            with patch.object(admission_launcher, '_actual_qa', return_value=wrong_qa), patch.object(
                    admission_launcher, '_fresh_no_weight_preflight', return_value=f.frozen), patch.object(
                    loader, '_pinned_snapshot_path', return_value=snapshot), patch.object(
                    loader, '_snapshot_manifest', side_effect=lambda: fake_snapshot_manifest(snapshot)):
                with self.assertRaisesRegex(ValueError, 'Actual human QA'):
                    loader._load_after_verified(
                        files=f.files, review_path=f.review_path, pins=f.pins,
                        torch_api=FakeTorch, snapshot=snapshot,
                        processor_factory=FakeAutoProcessor, model_factory=FakeAutoModel)
            self.assertEqual((FakeAutoProcessor.calls, FakeAutoModel.calls), (0, 0))

    def test_bad_model_dtype_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            f = prepared_fixture(Path(temp))
            snapshot = Path(temp) / 'fake_snapshot'
            with patch.object(admission_launcher, '_actual_qa', return_value=f.qa), patch.object(
                    admission_launcher, '_fresh_no_weight_preflight', return_value=f.frozen), patch.object(
                    loader, '_pinned_snapshot_path', return_value=snapshot), patch.object(
                    loader, '_snapshot_manifest', side_effect=lambda: fake_snapshot_manifest(snapshot)):
                with self.assertRaisesRegex(ValueError, 'FP32/eval/MPS'):
                    loader._load_after_verified(
                        files=f.files, review_path=f.review_path, pins=f.pins,
                        torch_api=FakeTorch, snapshot=snapshot,
                        processor_factory=FakeAutoProcessor, model_factory=BadFakeAutoModel)

    def test_alternate_snapshot_rejected_before_factories(self):
        with tempfile.TemporaryDirectory() as temp:
            f = prepared_fixture(Path(temp))
            pinned = Path(temp) / 'fake_snapshot'
            alternate = Path(temp) / 'alternate'
            alternate.mkdir()
            with patch.object(admission_launcher, '_actual_qa', return_value=f.qa), patch.object(
                    admission_launcher, '_fresh_no_weight_preflight', return_value=f.frozen), patch.object(
                    loader, '_pinned_snapshot_path', return_value=pinned):
                with self.assertRaisesRegex(ValueError, 'Supplied snapshot path differs'):
                    loader._load_after_verified(files=f.files, review_path=f.review_path,
                        pins=f.pins, torch_api=FakeTorch, snapshot=alternate,
                        processor_factory=FakeAutoProcessor, model_factory=FakeAutoModel)
            self.assertEqual((FakeAutoProcessor.calls, FakeAutoModel.calls), (0, 0))

    def test_pre_factory_snapshot_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            f = prepared_fixture(Path(temp))
            snapshot = Path(temp) / 'fake_snapshot'
            (snapshot / 'config.json').write_text('changed after fake preflight')
            with patch.object(admission_launcher, '_actual_qa', return_value=f.qa), patch.object(
                    admission_launcher, '_fresh_no_weight_preflight', return_value=f.frozen), patch.object(
                    loader, '_pinned_snapshot_path', return_value=snapshot), patch.object(
                    loader, '_snapshot_manifest', side_effect=lambda: fake_snapshot_manifest(snapshot)):
                with self.assertRaisesRegex(ValueError, 'changed before factory'):
                    loader._load_after_verified(files=f.files, review_path=f.review_path,
                        pins=f.pins, torch_api=FakeTorch, snapshot=snapshot,
                        processor_factory=FakeAutoProcessor, model_factory=FakeAutoModel)
            self.assertEqual((FakeAutoProcessor.calls, FakeAutoModel.calls), (0, 0))

    def test_in_factory_snapshot_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            f = prepared_fixture(Path(temp))
            snapshot = Path(temp) / 'fake_snapshot'
            with patch.object(admission_launcher, '_actual_qa', return_value=f.qa), patch.object(
                    admission_launcher, '_fresh_no_weight_preflight', return_value=f.frozen), patch.object(
                    loader, '_pinned_snapshot_path', return_value=snapshot), patch.object(
                    loader, '_snapshot_manifest', side_effect=lambda: fake_snapshot_manifest(snapshot)):
                with self.assertRaisesRegex(ValueError, 'changed during factory'):
                    loader._load_after_verified(files=f.files, review_path=f.review_path,
                        pins=f.pins, torch_api=FakeTorch, snapshot=snapshot,
                        processor_factory=FakeAutoProcessor, model_factory=MutatingAutoModel)

    def test_public_entry_stays_closed(self):
        with self.assertRaisesRegex(ValueError, 'HOLD'):
            loader.load_v6_pilot_model()


if __name__ == '__main__':
    unittest.main()
