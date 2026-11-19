"""Fake no-model V6 batch adapter tests. No checkpoint or corpus inference."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from models.second_architecture import voxtral_track_b_v6_bound_batch_adapter_source_v1 as adapter
from models.second_architecture import voxtral_track_b_v6_pilot_worker_source_v1 as worker
from models.second_architecture import voxtral_track_b_v6_pre_pilot_launcher_source_v1 as admission_launcher
from models.second_architecture import voxtral_track_b_v6_pinned_loader_source_v1 as pinned_loader
from models.second_architecture.test_voxtral_track_b_v6_pre_pilot_contract_v1 import Fixture, write_json


class FakeTokenizer:
    eos_token_id = 2


class FakeProcessor:
    calls = 0
    mutate_ids = False

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.audio_token_id = 24

    def apply_chat_template(self, conversation, **kwargs):
        assert kwargs == {'tokenize': True, 'return_dict': True, 'return_tensors': 'pt'}
        type(self).calls += 1
        prompt = conversation[0]['content'][1]['text']
        is_old = prompt == adapter.FIT_PROMPT
        ids = torch.ones((1, 396), dtype=torch.int64)
        ids[0, :375] = 24
        ids[0, -1] = 5 if is_old else 6
        if type(self).mutate_ids:
            ids[0, -2] = 9
        features = torch.zeros((1, 128, 3000), dtype=torch.float32)
        if not is_old:
            features += 1
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                'input_features': features}


class FakeProcessorFactory:
    calls = 0

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        cls.calls += 1
        return FakeProcessor()


def fake_batches():
    processor = FakeProcessor()
    old = processor.apply_chat_template([{'content': [{'type': 'audio'},
                                  {'type': 'text', 'text': adapter.FIT_PROMPT}]}],
                                  tokenize=True, return_dict=True, return_tensors='pt')
    zero = processor.apply_chat_template([{'content': [{'type': 'audio'},
                                  {'type': 'text', 'text': adapter.TRACK_PROMPT}]}],
                                  tokenize=True, return_dict=True, return_tensors='pt')
    return old, zero


def fake_case(directory: Path):
    f = Fixture(directory)
    snapshot = directory / 'fake_snapshot'
    snapshot.mkdir()
    old_wav = directory / 'old_silver.wav'
    old_pcm, old_wav_bytes = adapter._wav_from_zero_pcm(16000)
    old_wav.write_bytes(old_wav_bytes)
    f.old_wav_sha256 = worker.sha_file(old_wav)
    old, zero = fake_batches()
    geometry = f.frozen['v6_synthetic_zero_geometry']
    pcm, zero_wav = adapter._wav_from_zero_pcm(16000)
    geometry.update({
        'first_componentwise_satisfying_frames': 16000,
        'selected_max_frames': 16000,
        'pcm_sha256': adapter._sha_bytes(pcm),
        'wav_sha256': adapter._sha_bytes(zero_wav),
        'input_ids_sha256': adapter._raw_tensor_sha(zero['input_ids']),
        'attention_mask_sha256': adapter._raw_tensor_sha(zero['attention_mask']),
        'input_features_sha256': adapter._raw_tensor_sha(zero['input_features']),
        'input_features_shape': [1, 128, 3000],
        'audio_placeholder_count': 375, 'projector_rows_implied': 375,
        'attention_mask_length': 396,
    })
    f.frozen['processor'] = {'processor_type': 'FakeProcessor',
                             'tokenizer_type': 'FakeTokenizer',
                             'prompt': adapter.TRACK_PROMPT}
    f.bindings['old_fit_batch_sha256'] = worker._batch_sha(old)
    f.bindings['synthetic_batch_sha256'] = worker._batch_sha(zero)
    f.bindings['synthetic_processor_geometry'] = {
        'input_features_shape': [1, 128, 3000],
        'audio_placeholder_count': 375, 'attention_mask_length': 396}
    f.bindings['synthetic_geometry_sha256'] = worker.sha_bytes(json.dumps(
        geometry, sort_keys=True, separators=(',', ':')).encode())
    f.bindings['preflight_sha256'] = write_json(f.files.preflight, f.frozen)
    f.repin_admission_and_review()
    old_hashes = {key: adapter._raw_tensor_sha(value) for key, value in old.items()}
    return f, snapshot, old_wav, old_hashes


class BoundBatchAdapterTests(unittest.TestCase):
    def setUp(self):
        FakeProcessor.calls = 0
        FakeProcessor.mutate_ids = False
        FakeProcessorFactory.calls = 0

    def _build(self, f, snapshot, old_wav, old_hashes, *, qa=None):
        with patch.object(adapter, 'OLD_WAV_SHA256', f.old_wav_sha256), patch.object(
                adapter, 'OLD_FIT_RAW_HASHES', old_hashes), patch.object(
                adapter, '_resolve_old_fit_silver', return_value=old_wav), patch.object(
                admission_launcher, '_actual_qa', return_value=qa or f.qa), patch.object(
                admission_launcher, '_fresh_no_weight_preflight', return_value=f.frozen), patch.object(
                pinned_loader, '_pinned_snapshot_path', return_value=snapshot), patch.object(
                pinned_loader, '_snapshot_manifest', return_value=f.frozen['snapshot']), patch.object(
                pinned_loader, '_installed_source_map', return_value=f.frozen['installed_source_sha256']):
            return adapter.build_bound_pilot_batches(
                files=f.files, review_path=f.review_path, pins=f.pins,
                processor_factory=FakeProcessorFactory)

    def test_fake_old_fit_and_zero_batches_match_bound_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            f, snapshot, old_wav, old_hashes = fake_case(Path(temp))
            result = self._build(f, snapshot, old_wav, old_hashes)
            self.assertEqual(result.old_fit_sha256, f.bindings['old_fit_batch_sha256'])
            self.assertEqual(result.synthetic_sha256, f.bindings['synthetic_batch_sha256'])
            self.assertEqual(result.synthetic_pcm_sha256,
                             f.frozen['v6_synthetic_zero_geometry']['pcm_sha256'])
            self.assertEqual(FakeProcessorFactory.calls, 1)

    def test_qa_failure_precedes_processor_factory(self):
        with tempfile.TemporaryDirectory() as temp:
            f, snapshot, old_wav, old_hashes = fake_case(Path(temp))
            bad_qa = {**f.qa, 'qa_go_receipt_sha256': '0' * 64}
            with self.assertRaisesRegex(ValueError, 'Actual human QA'):
                self._build(f, snapshot, old_wav, old_hashes, qa=bad_qa)
            self.assertEqual(FakeProcessorFactory.calls, 0)

    def test_changed_old_wav_and_synthetic_geometry_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            f, snapshot, old_wav, old_hashes = fake_case(Path(temp))
            old_wav.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Old FIT silver WAV bytes differ'):
                self._build(f, snapshot, old_wav, old_hashes)
        with tempfile.TemporaryDirectory() as temp:
            f, snapshot, old_wav, old_hashes = fake_case(Path(temp))
            f.frozen['v6_synthetic_zero_geometry']['pcm_sha256'] = '0' * 64
            f.bindings['synthetic_geometry_sha256'] = worker.sha_bytes(json.dumps(
                f.frozen['v6_synthetic_zero_geometry'], sort_keys=True,
                separators=(',', ':')).encode())
            f.bindings['preflight_sha256'] = write_json(f.files.preflight, f.frozen)
            f.repin_admission_and_review()
            with self.assertRaisesRegex(ValueError, 'Synthetic zero PCM or WAV bytes differ'):
                self._build(f, snapshot, old_wav, old_hashes)

    def test_changed_prompt_ids_or_admission_batch_sha_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            f, snapshot, old_wav, old_hashes = fake_case(Path(temp))
            FakeProcessor.mutate_ids = True
            with self.assertRaisesRegex(ValueError, 'input_ids raw bytes differ'):
                self._build(f, snapshot, old_wav, old_hashes)
            FakeProcessor.mutate_ids = False
        with tempfile.TemporaryDirectory() as temp:
            f, snapshot, old_wav, old_hashes = fake_case(Path(temp))
            f.bindings['old_fit_batch_sha256'] = 'f' * 64
            f.repin_admission_and_review()
            with self.assertRaisesRegex(ValueError, 'Pilot batch SHA differs'):
                self._build(f, snapshot, old_wav, old_hashes)


if __name__ == '__main__':
    unittest.main()
