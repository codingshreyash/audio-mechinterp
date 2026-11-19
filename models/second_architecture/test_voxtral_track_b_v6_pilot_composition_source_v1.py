"""Fake-only V6 composition checks; no real model or candidate corpus."""

from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.second_architecture import voxtral_track_b_v6_pilot_composition_source_v1 as composition
from models.second_architecture import voxtral_track_b_v6_pilot_worker_source_v1 as worker
from models.second_architecture import voxtral_track_b_v6_pre_pilot_launcher_source_v1 as admission_launcher
from models.second_architecture import voxtral_track_b_v6_pinned_loader_source_v1 as pinned_loader
from models.second_architecture.test_voxtral_track_b_v6_pre_pilot_contract_v1 import Fixture, fresh_host, write_json
from models.second_architecture.test_voxtral_track_b_v6_pilot_worker_source_v1 import FakePilotModel, fake_host
from models.second_architecture.test_voxtral_track_b_answer_link_core_v1 import fixture
from models.second_architecture.test_voxtral_track_b_v6_mps_runtime_adapter_source_v1 import FakeMPS, fake_hang


def fake_case(directory: Path, batch) -> Fixture:
    f = Fixture(directory)
    f.bindings['old_fit_batch_sha256'] = worker._batch_sha(batch)
    f.bindings['synthetic_batch_sha256'] = worker._batch_sha(batch)
    geometry = f.frozen['v6_synthetic_zero_geometry']
    geometry['input_features_shape'] = list(batch['input_features'].shape)
    geometry['audio_placeholder_count'] = int((batch['input_ids'] == 24).sum())
    geometry['projector_rows_implied'] = geometry['audio_placeholder_count']
    geometry['attention_mask_length'] = int(batch['attention_mask'].shape[-1])
    f.bindings['synthetic_processor_geometry'] = {
        'input_features_shape': geometry['input_features_shape'],
        'audio_placeholder_count': geometry['audio_placeholder_count'],
        'attention_mask_length': geometry['attention_mask_length']}
    f.bindings['longest_alias_ids_with_eos'] = list(worker.LONGEST_ALIAS)
    f.bindings['synthetic_geometry_sha256'] = worker.sha_bytes(json.dumps(
        geometry, sort_keys=True, separators=(',', ':')).encode())
    f.bindings['preflight_sha256'] = write_json(f.files.preflight, f.frozen)
    f.repin_admission_and_review()
    return f


class PilotCompositionSourceTests(unittest.TestCase):
    def test_fake_full_worker_schedule_and_post_host_sidecar(self):
        _model, batch, _source, _donor = fixture()
        with tempfile.TemporaryDirectory() as temp, patch.object(
                worker, 'LONGEST_ALIAS', (7, 9, 8, 2)):
            root = Path(temp)
            f = fake_case(root, batch)
            loader_calls = []
            def fake_loader():
                loader_calls.append(1)
                return FakePilotModel()
            with patch.object(admission_launcher, '_actual_qa', return_value=f.qa), patch.object(
                    admission_launcher, '_fresh_no_weight_preflight', return_value=f.frozen):
                result = composition._execute_fake_testable_composition(
                    files=f.files, review_path=f.review_path, pins=f.pins,
                    output_dir=root / 'pilot', batch_provider=lambda: (batch, batch, 0),
                    load_model=fake_loader, host_probe=fake_host,
                    post_host=fresh_host, memory_probe=lambda: 1024,
                    free_bytes=lambda _: 100 * 1024**3, sync=lambda: None)
            self.assertEqual(loader_calls, [1])
            self.assertEqual(result['status'], 'schedule_complete_resource_unverified')
            saved = json.loads((root / 'pilot.composition.json').read_text())
            self.assertEqual(saved['campaign_attempt_id'], f.bindings['campaign_attempt_id'])
            self.assertEqual(saved['status'], 'worker_complete_post_host_checked_not_pilot_go')
            measurement = json.loads((root / 'pilot/pilot_measurement.json').read_text())
            self.assertEqual(measurement['attempted_call_count'], 88)

    def test_fake_bad_qa_or_batch_blocks_model_loader(self):
        _model, batch, _source, _donor = fixture()
        with tempfile.TemporaryDirectory() as temp, patch.object(
                worker, 'LONGEST_ALIAS', (7, 9, 8, 2)):
            root = Path(temp)
            f = fake_case(root, batch)
            calls = []
            wrong_qa = {**f.qa, 'qa_go_receipt_sha256': '0' * 64}
            with patch.object(admission_launcher, '_actual_qa', return_value=wrong_qa), patch.object(
                    admission_launcher, '_fresh_no_weight_preflight', return_value=f.frozen):
                with self.assertRaisesRegex(ValueError, 'Actual human QA'):
                    composition._execute_fake_testable_composition(
                        files=f.files, review_path=f.review_path, pins=f.pins,
                        output_dir=root / 'bad_qa', batch_provider=lambda: (batch, batch, 0),
                        load_model=lambda: calls.append(1), host_probe=fake_host,
                        post_host=fresh_host, memory_probe=lambda: 1024,
                        free_bytes=lambda _: 100 * 1024**3, sync=lambda: None)
            with patch.object(admission_launcher, '_actual_qa', return_value=f.qa), patch.object(
                    admission_launcher, '_fresh_no_weight_preflight', return_value=f.frozen):
                with self.assertRaisesRegex(ValueError, 'Bound pilot batches differ'):
                    composition._execute_fake_testable_composition(
                        files=f.files, review_path=f.review_path, pins=f.pins,
                        output_dir=root / 'bad_batch', batch_provider=lambda: (batch, batch, -1),
                        load_model=lambda: calls.append(1), host_probe=fake_host,
                        post_host=fresh_host, memory_probe=lambda: 1024,
                        free_bytes=lambda _: 100 * 1024**3, sync=lambda: None)
            self.assertEqual(calls, [])

    def test_replayed_pre_host_timestamp_cannot_be_post_host_go(self):
        _model, batch, _source, _donor = fixture()
        with tempfile.TemporaryDirectory() as temp, patch.object(
                worker, 'LONGEST_ALIAS', (7, 9, 8, 2)):
            root = Path(temp)
            f = fake_case(root, batch)
            with patch.object(admission_launcher, '_actual_qa', return_value=f.qa), patch.object(
                    admission_launcher, '_fresh_no_weight_preflight', return_value=f.frozen):
                with self.assertRaisesRegex(ValueError, 'strictly later'):
                    composition._execute_fake_testable_composition(
                        files=f.files, review_path=f.review_path, pins=f.pins,
                        output_dir=root / 'pilot', batch_provider=lambda: (batch, batch, 0),
                        load_model=FakePilotModel, host_probe=fake_host,
                        post_host=lambda: f.frozen['host'], memory_probe=lambda: 1024,
                        free_bytes=lambda _: 100 * 1024**3, sync=lambda: None)
            self.assertTrue((root / 'pilot/pilot_measurement.json').is_file())
            self.assertFalse((root / 'pilot.composition.json').exists())

    def test_top_level_child_operation_pickleable_but_holds_before_loader(self):
        with tempfile.TemporaryDirectory() as temp:
            f = Fixture(Path(temp))
            operation = composition.PilotOperation(f.files, f.review_path, f.pins,
                                                   Path(temp) / 'pilot')
            self.assertIsInstance(pickle.loads(pickle.dumps(operation)), composition.PilotOperation)
            with patch.object(admission_launcher, 'verify_pre_pilot', return_value={}), patch.object(
                    pinned_loader, '_load_pinned_fp32_voxtral') as model_load:
                with self.assertRaisesRegex(ValueError, 'batch adapter absent'):
                    operation()
            model_load.assert_not_called()

    def test_private_parent_timeout_claim_no_retry_and_public_hold(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            claim = root / 'campaign.json'
            result = composition._guarded_fake_attempt(
                operation=fake_hang, campaign_claim_path=claim,
                output_dir=root / 'pilot', fake_mps=FakeMPS(), deadline_seconds=0.2)
            self.assertEqual(result['status'], 'timeout_no_retry')
            with self.assertRaises(FileExistsError):
                composition._guarded_fake_attempt(
                    operation=fake_hang, campaign_claim_path=claim,
                    output_dir=root / 'alternate', fake_mps=FakeMPS(), deadline_seconds=0.2)
        with self.assertRaisesRegex(ValueError, 'HOLD'):
            composition.launch_v6_pilot()


if __name__ == '__main__':
    unittest.main()
