"""Fake-only V6 pilot schedule/admission tests. No checkpoint or corpus audio."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from models.second_architecture import voxtral_track_b_v6_pilot_worker_source_v1 as pilot
from models.second_architecture.test_voxtral_track_b_answer_link_core_v1 import FakeVoxtral, fixture


class FakePilotModel(FakeVoxtral):
    def generate(self, *, input_ids, attention_mask, input_features,
                 do_sample, use_cache, max_new_tokens, eos_token_id, pad_token_id,
                 min_new_tokens=None):
        self.forward(input_ids=input_ids, attention_mask=attention_mask,
                     input_features=input_features, use_cache=True, logits_to_keep=1)
        if min_new_tokens == 32:
            suffix = [7] * 31 + [2]
        elif min_new_tokens is None:
            suffix = [7, 2]
        else:
            raise ValueError('Unexpected forced-generation bound')
        return torch.cat([input_ids,
                          torch.tensor([suffix], dtype=input_ids.dtype, device=input_ids.device)], dim=1)


class FailingPilotModel(FakePilotModel):
    def generate(self, **kwargs):
        self.forward(input_ids=kwargs['input_ids'],
                     attention_mask=kwargs['attention_mask'],
                     input_features=kwargs['input_features'],
                     use_cache=True, logits_to_keep=1)
        raise RuntimeError('fake interrupted model call')


def fake_host(post: bool):
    value = {'ac': True, 'lid_open': True, 'ten_minute_clean': True,
             'raw_power_log': b'fake bounded pmset event log'}
    if post:
        value['during_run_clean'] = True
    return value


class PilotSourceTests(unittest.TestCase):
    def test_externally_pinned_admission_and_worker_hashes(self):
        def write_json(path, value):
            path.write_text(json.dumps(value, sort_keys=True))
            return pilot.sha_file(path)

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            qa = directory / 'qa.json'
            qa_sha = write_json(qa, {
                'schema': 'word_disjoint_human_qa_audit_v1',
                'status': 'human_qa_32_of_32_pass', 'human_quality_gate_passed': True,
                'planned_clips': 32, 'passed_clips': 32})
            qa_review = directory / 'qa_review.md'
            qa_review.write_text(f'Decision: GO\nExact QA SHA: {qa_sha}\n')
            geometry = {
                'longest_alias': {'token_ids_with_eos': list(pilot.LONGEST_ALIAS)},
                'input_features_shape': [1, 128, 3000],
                'audio_placeholder_count': 375,
                'attention_mask_length': 405,
                'projector_rows_implied': 375,
            }
            preflight = directory / 'preflight.json'
            preflight_sha = write_json(preflight, {'v6_synthetic_zero_geometry': geometry})
            runner = directory / 'runner.py'
            runner.write_text('# fake frozen runner source\n')
            auditor = directory / 'auditor.py'
            auditor.write_text('# fake frozen auditor source\n')
            worker_sha = pilot.sha_file(Path(pilot.__file__))
            registry = directory / 'registry'
            registry.mkdir()
            bindings = {
                'protocol_sha256': pilot.PROTOCOL_SHAS,
                'qa_sha256': qa_sha,
                'qa_review_sha256': pilot.sha_file(qa_review),
                'preflight_sha256': preflight_sha,
                'study_runner_sha256': pilot.sha_file(runner),
                'pilot_auditor_sha256': pilot.sha_file(auditor),
                'pilot_worker_sha256': worker_sha,
                'synthetic_geometry_sha256': pilot.sha_bytes(json.dumps(
                    geometry, sort_keys=True, separators=(',', ':')).encode()),
                'synthetic_processor_geometry': {
                    'input_features_shape': [1, 128, 3000],
                    'audio_placeholder_count': 375, 'attention_mask_length': 405},
                'checkpoint_tree_sha256': 'a' * 64,
                'transformers_source_sha256': 'b' * 64,
                'synthetic_batch_sha256': 'c' * 64,
                'old_fit_batch_sha256': 'd' * 64,
                'campaign_attempt_id': 'frozen_fake_attempt',
                'campaign_claim_path': str(registry / 'attempt.json'),
            }
            admission = directory / 'admission.json'
            admission_sha = write_json(admission, {
                'schema': 'voxtral_track_b_v6_external_pilot_admission_v1',
                'status': 'pilot_go', 'bindings': bindings})
            files = pilot.AdmissionFiles(admission, qa, qa_review, preflight, runner, auditor)
            self.assertEqual(pilot.verify_admission(files, admission_sha, worker_sha)['bindings'], bindings)
            with self.assertRaisesRegex(ValueError, 'Worker bytes differ'):
                pilot.verify_admission(files, admission_sha, '0' * 64)
            with self.assertRaisesRegex(ValueError, 'admission bytes differ'):
                pilot.verify_admission(files, '0' * 64, worker_sha)
            self.assertFalse((registry / 'attempt.json').exists())

    def test_public_entry_refuses_before_model_loader(self):
        calls = []
        with self.assertRaisesRegex(ValueError, 'Public pilot worker disabled'):
            pilot.run_pilot(files=pilot.AdmissionFiles(*(Path('/missing') for _ in range(6))),
                            load_model=lambda: calls.append('loaded'),
                            old_fit_batch={}, synthetic_batch={}, pad=0,
                            output_dir=Path('/tmp/not-created-pilot-test'),
                            host_probe=fake_host, memory_probe=lambda: 0,
                            free_bytes=lambda _: 100 * 1024**3, sync=lambda: None)
        self.assertEqual(calls, [])

    def test_fake_exact_schedule_hash_only_receipt_and_no_retry(self):
        _model, batch, _source, _donor = fixture()
        loader_calls = []
        bindings = {'old_fit_batch_sha256': pilot._batch_sha(batch),
                    'synthetic_batch_sha256': pilot._batch_sha(batch),
                    'synthetic_processor_geometry': {
                        'input_features_shape': list(batch['input_features'].shape),
                        'audio_placeholder_count': 3, 'attention_mask_length': 4},
                    'longest_alias_ids_with_eos': [7, 9, 8, 2],
                    'qa_sha256': 'a' * 64,
                    'preflight_sha256': 'b' * 64,
                    'study_runner_sha256': 'c' * 64,
                    'pilot_worker_sha256': 'd' * 64}
        with tempfile.TemporaryDirectory() as temp, patch.object(pilot, 'LONGEST_ALIAS', (7, 9, 8, 2)):
            dest = Path(temp) / 'pilot'
            registry = Path(temp) / 'registry'
            registry.mkdir()
            bindings.update(campaign_attempt_id='fake_campaign_1',
                            campaign_claim_path=str(registry / 'attempt.json'))
            def loader():
                loader_calls.append(1)
                return FakePilotModel()
            kwargs = dict(load_model=loader, old_fit_batch=batch, synthetic_batch=batch,
                          pad=0, output_dir=dest, bindings=bindings, host_probe=fake_host,
                          memory_probe=lambda: 1024, free_bytes=lambda _: 100 * 1024**3,
                          sync=lambda: None)
            result = pilot._execute_after_admission(**kwargs)
            self.assertEqual(loader_calls, [1])
            self.assertEqual(result['status'], 'schedule_complete_resource_unverified')
            receipt = json.loads((dest / 'pilot_measurement.json').read_text())
            self.assertIs(receipt['measured_mps_high_water_present'], False)
            self.assertIs(receipt['hard_per_call_watchdog_present'], False)
            self.assertEqual(receipt['attempted_call_count'], 88)
            self.assertEqual(receipt['attempted_calls'],
                             {'old_fit_clean': 1, 'old_fit_zero': 1,
                              'forced_clean': 1, 'forced_zero': 1,
                              'score_nograd': 72, 'score_grad': 12})
            self.assertEqual(receipt['missing_calls'], {k: 0 for k in receipt['planned_calls']})
            self.assertEqual(receipt['completed_calls'], receipt['attempted_calls'])
            self.assertEqual(receipt['synthetic_suffix_length'], 32)
            self.assertEqual(receipt['calls_sha256'], pilot.sha_file(dest / 'calls.jsonl'))
            rows = [json.loads(line) for line in (dest / 'calls.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 176)
            starts = [r for r in rows if r['phase'] == 'started']
            finishes = [r for r in rows if r['phase'] == 'completed']
            self.assertEqual(len(starts), 88)
            self.assertEqual(len(finishes), 88)
            self.assertEqual([r['class'] for r in finishes[:4]],
                             ['old_fit_clean', 'old_fit_zero', 'forced_clean', 'forced_zero'])
            self.assertEqual([r['suffix_length'] for r in finishes[:4]], [2, 2, 32, 32])
            self.assertTrue(all('suffix_ids' not in row and 'decoded' not in row and
                                'score_value' not in row for row in rows))
            self.assertTrue(all(row['hook_calls'] == 1 for row in finishes))
            self.assertEqual(len([r for r in finishes if r['class'] == 'score_grad']), 12)
            self.assertTrue((dest / 'host_pre.raw').is_file() and (dest / 'host_post.raw').is_file())
            claim = json.loads((registry / 'attempt.json').read_text())
            self.assertEqual(claim['campaign_attempt_id'], 'fake_campaign_1')
            with self.assertRaises(FileExistsError):
                pilot._execute_after_admission(**kwargs)
            with self.assertRaises(FileExistsError):
                pilot._execute_after_admission(**{**kwargs, 'output_dir': Path(temp) / 'alternate'})
            self.assertFalse((Path(temp) / 'alternate').exists())

    def test_host_gate_fails_before_loader_and_retains_no_retry_receipt(self):
        _model, batch, _source, _donor = fixture()
        called = []
        bindings = {'old_fit_batch_sha256': pilot._batch_sha(batch),
                    'synthetic_batch_sha256': pilot._batch_sha(batch),
                    'synthetic_processor_geometry': {
                        'input_features_shape': list(batch['input_features'].shape),
                        'audio_placeholder_count': 3, 'attention_mask_length': 4},
                    'longest_alias_ids_with_eos': [7, 9, 8, 2]}
        with tempfile.TemporaryDirectory() as temp, patch.object(pilot, 'LONGEST_ALIAS', (7, 9, 8, 2)):
            dest = Path(temp) / 'pilot'
            registry = Path(temp) / 'registry'
            registry.mkdir()
            bindings.update(campaign_attempt_id='fake_campaign_2',
                            campaign_claim_path=str(registry / 'attempt.json'))
            with self.assertRaisesRegex(ValueError, 'Host AC'):
                pilot._execute_after_admission(
                    load_model=lambda: called.append(1), old_fit_batch=batch,
                    synthetic_batch=batch, pad=0, output_dir=dest, bindings=bindings,
                    host_probe=lambda _: {'ac': False, 'lid_open': True,
                                          'ten_minute_clean': True, 'raw_power_log': b'x'},
                    memory_probe=lambda: 0, free_bytes=lambda _: 100 * 1024**3,
                    sync=lambda: None)
            self.assertEqual(called, [])
            self.assertEqual(json.loads((dest / 'pilot_failure.json').read_text())['status'],
                             'failed_no_retry')

    def test_failed_model_call_is_persisted_as_attempt(self):
        _model, batch, _source, _donor = fixture()
        bindings = {'old_fit_batch_sha256': pilot._batch_sha(batch),
                    'synthetic_batch_sha256': pilot._batch_sha(batch),
                    'synthetic_processor_geometry': {
                        'input_features_shape': list(batch['input_features'].shape),
                        'audio_placeholder_count': 3, 'attention_mask_length': 4},
                    'longest_alias_ids_with_eos': [7, 9, 8, 2]}
        with tempfile.TemporaryDirectory() as temp, patch.object(pilot, 'LONGEST_ALIAS', (7, 9, 8, 2)):
            dest = Path(temp) / 'pilot'
            registry = Path(temp) / 'registry'
            registry.mkdir()
            bindings.update(campaign_attempt_id='fake_campaign_3',
                            campaign_claim_path=str(registry / 'attempt.json'))
            with self.assertRaisesRegex(RuntimeError, 'fake interrupted'):
                pilot._execute_after_admission(
                    load_model=FailingPilotModel, old_fit_batch=batch,
                    synthetic_batch=batch, pad=0, output_dir=dest, bindings=bindings,
                    host_probe=fake_host, memory_probe=lambda: 1024,
                    free_bytes=lambda _: 100 * 1024**3, sync=lambda: None)
            failure = json.loads((dest / 'pilot_failure.json').read_text())
            self.assertEqual(failure['attempted_calls']['old_fit_clean'], 1)
            self.assertEqual(failure['completed_calls']['old_fit_clean'], 0)
            rows = [json.loads(line) for line in (dest / 'calls.jsonl').read_text().splitlines()]
            self.assertEqual([r['phase'] for r in rows], ['started', 'failed'])


if __name__ == '__main__':
    unittest.main()
