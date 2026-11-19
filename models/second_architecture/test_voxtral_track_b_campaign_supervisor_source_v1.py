"""Fake CPU campaign tests; no candidate weights or audio."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import voxtral_track_b_campaign_supervisor_source_v1 as sup  # noqa: E402
import voxtral_track_b_answer_link_core_v1 as core  # noqa: E402
import voxtral_fp32_track_b_study_runner_v1 as study  # noqa: E402


def plan():
    out = []
    for pair_no, pair in enumerate(study.preflight.PAIR_ORDER):
        for engine in study.preflight.ENGINES:
            for recipient, donor in (pair, tuple(reversed(pair))):
                out.append({'index': len(out),
                    'context_id': f'pair_{pair_no:02d}|{engine}|{recipient}->{donor}',
                    'pair_id': f'pair_{pair_no:02d}', 'engine': engine,
                    'recipient': recipient, 'donor': donor,
                    'recipient_aliases': [[7, 2]], 'donor_aliases': [[8, 2]]})
    return out


def clean(p, guard):
    guard('before|generation|clean')
    source = torch.zeros((2, core.PROJECTOR_WIDTH), dtype=torch.float32)
    if p['recipient'] == study.preflight.PAIR_ORDER[int(p['pair_id'][-2:])][1]:
        source[0, 0] = 1
    guard('after|generation|clean')
    return ({'hook': {'captured_during_generation': True,
                      'source_sha256': core.tensor_sha(source)},
             'suffix_ids': [7, 2]}, source)


def complete(p, source, donor_source, recipient_clean, donor_clean, guard):
    for label in study.expected_context_call_labels(p)[1:]:
        guard('before|' + label)
        guard('after|' + label)
    tensors = {}
    grad_scores = {}
    for word in ('recipient', 'donor'):
        value = torch.zeros((1, core.PROJECTOR_WIDTH), dtype=torch.float32)
        value[0, 0] = 1 if word == 'recipient' else -1
        tensors[word + '_0'] = value
        grad_scores[word] = {'entries': [{
            'selected_gradient_sha256': core.tensor_sha(value),
            'selected_gradient_shape': list(value.shape),
            'selected_gradient_l2': float(torch.linalg.vector_norm(value))}]}
    evidence = {'context_id': p['context_id'], 'source_sha256': core.tensor_sha(source),
                'donor_source_sha256': core.tensor_sha(donor_source),
                'grad_scores': grad_scores,
                'decision': {'local_score_pass': True, 'predicted_switch_count': 1,
                             'raw_donor_switches': {'plus_0_1': True,
                                                   'minus_0_1': False,
                                                   'plus_1_0': False},
                             'random_switch': False}}
    return evidence, tensors


class CampaignTests(unittest.TestCase):
    def test_sixteen_contexts_complete_and_no_retry(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            kwargs = dict(plan=plan(), attempt_id='attempt_one',
                          claim_path=base / 'stable.claim', clean=clean, complete=complete,
                          host_probe=lambda: {'ac_power': True, 'lid_open': True,
                                              'clean_pmset_10min': True, 'no_transition': True},
                          mps_bytes=lambda: 0, free_bytes=lambda _p: 100 * 1024**3)
            result = sup._execute_fake_schedule(output_dir=base / 'first', **kwargs)
            self.assertEqual(result['status'], 'schedule_complete_resource_unverified')
            self.assertEqual((result['completed_contexts'], result['failed_contexts'],
                              result['missing_contexts']), (16, 0, 0))
            self.assertEqual(result['call_events'], 2 * 16 * (7 + 12 + 2))
            self.assertFalse(result['hard_per_call_watchdog_present'])
            self.assertEqual(result['frozen_gate']['local_score_pass_contexts'], 16)
            self.assertEqual(result['frozen_gate']['predicted_switch_contexts'], 16)
            self.assertTrue(result['frozen_gate']['all_clauses'])
            record = json.loads((base / 'first/context_00.json').read_text())
            self.assertEqual(len(record['gradient_sidecars']), 2)
            for meta in record['gradient_sidecars'].values():
                tensor = torch.load(base / 'first' / meta['path'], weights_only=True)
                self.assertEqual(core.tensor_sha(tensor), meta['tensor_sha256'])
            with self.assertRaises(FileExistsError):
                sup._execute_fake_schedule(output_dir=base / 'second', **kwargs)

    def test_failed_call_keeps_all_sixteen_rows_and_no_retry(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            def fail_first(p, guard):
                guard('before|generation|clean')
                raise RuntimeError('fake crash')
            result = sup._execute_fake_schedule(
                plan=plan(), output_dir=base / 'failed', attempt_id='one',
                claim_path=base / 'claim', clean=fail_first, complete=complete,
                host_probe=lambda: {'ac_power': True, 'lid_open': True,
                                    'clean_pmset_10min': True, 'no_transition': True},
                mps_bytes=lambda: 0, free_bytes=lambda _p: 100 * 1024**3)
            self.assertEqual(result['status'], 'failed_no_retry')
            self.assertEqual((result['failed_contexts'], result['missing_contexts']), (1, 15))
            self.assertEqual(len(result['contexts']), 16)
            self.assertFalse(result['frozen_gate']['all_clauses'])
            self.assertTrue(all((base / 'failed' / f'context_{i:02d}.json').is_file()
                                for i in range(16)))
            events = [json.loads(x)['event'] for x in (base / 'failed/calls.jsonl').read_text().splitlines()]
            self.assertEqual(events, ['started', 'failed'])

    def test_disk_drop_after_record_keeps_original_and_final_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            output = base / 'after_record'
            def free_bytes(_path):
                return 0 if (output / 'context_00.json').is_file() else 100 * 1024**3
            result = sup._execute_fake_schedule(
                plan=plan(), output_dir=output, attempt_id='disk_drop',
                claim_path=base / 'stable.claim', clean=clean, complete=complete,
                host_probe=lambda: {'ac_power': True, 'lid_open': True,
                                    'clean_pmset_10min': True, 'no_transition': True},
                mps_bytes=lambda: 0, free_bytes=free_bytes)
            self.assertEqual(result['status'], 'failed_no_retry')
            self.assertEqual((result['completed_contexts'], result['failed_contexts'],
                              result['partial_contexts'], result['missing_contexts']),
                             (0, 0, 1, 15))
            original = output / 'context_00.json'
            self.assertIn('gradient_sidecars', json.loads(original.read_text()))
            self.assertEqual(result['contexts'][0]['record_sha256'], sup.sha(original))
            self.assertEqual(json.loads((output / 'campaign_result.json').read_text()), result)
            self.assertTrue(all((output / f'context_{i:02d}.json').is_file()
                                for i in range(16)))

    def test_public_loader_never_called_without_or_with_bound_fake_go(self):
        invoked = []
        with self.assertRaises(TypeError):
            sup.run_campaign(load_model=lambda: invoked.append(True))
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            def write(name, value):
                path = base / name
                path.write_text(json.dumps(value, sort_keys=True))
                return path, sup.sha(path)
            qa, qa_sha = write('qa.json', {'fake': True})
            pre, pre_sha = write('pre.json', {'fake': True})
            calls = base / 'pilot_calls.jsonl'
            calls.write_text('{"event":"fake"}\n')
            calls_sha = sup.sha(calls)
            measurement, measurement_sha = write('pilot_measurement.json', {
                'schema': 'voxtral_track_b_v6_pilot_measurement_v1',
                'status': 'complete_feasible', 'calls_sha256': calls_sha,
                'hard_per_call_watchdog_present': True,
                'measured_mps_high_water_present': True,
                'bindings': {'qa_sha256': qa_sha, 'preflight_sha256': pre_sha}})
            study_sha, supervisor_sha = sup.sha(Path(study.__file__)), sup.sha(Path(sup.__file__))
            bindings = {'pilot_measurement_sha256': measurement_sha,
                        'pilot_calls_sha256': calls_sha, 'qa_sha256': qa_sha,
                        'preflight_sha256': pre_sha, 'study_source_sha256': study_sha,
                        'supervisor_source_sha256': supervisor_sha,
                        'campaign_attempt_id': 'fake', 'campaign_claim_path': str(base / 'claim')}
            go, go_sha = write('go.json', {'schema': sup.GO_SCHEMA, 'status': 'GO',
                                         'independent_reviewer': 'fake-test',
                                         'external_anchor_sha256': 'a' * 64,
                                         'bindings': bindings})
            gate = dict(go_path=go, go_sha=go_sha, measurement_path=measurement,
                        measurement_sha=measurement_sha, calls_path=calls, calls_sha=calls_sha,
                        qa_path=qa, qa_sha=qa_sha, preflight_path=pre, preflight_sha=pre_sha,
                        study_source_sha=study_sha, supervisor_sha=supervisor_sha)
            with self.assertRaisesRegex(RuntimeError, 'hard watchdog'):
                sup.run_campaign(load_model=lambda: invoked.append(True), **gate)
            measurement.write_text(measurement.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'pilot measurement hash mismatch'):
                sup.run_campaign(load_model=lambda: invoked.append(True), **gate)
        self.assertEqual(invoked, [])


if __name__ == '__main__':
    unittest.main()
