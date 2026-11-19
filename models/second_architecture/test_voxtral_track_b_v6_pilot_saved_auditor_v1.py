"""Synthetic saved-data tamper checks. No pilot model or candidate audio."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from models.second_architecture import voxtral_track_b_v6_pilot_saved_auditor_v1 as audit


def write_json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n')
    return audit.sha(path)


def make_fixture(root: Path) -> tuple[Path, dict]:
    directory = root / 'pilot'
    directory.mkdir()
    registry = root / 'registry'
    registry.mkdir()
    qa = root / 'qa.json'
    qa_sha = write_json(qa, {
        'schema': 'word_disjoint_human_qa_audit_v1',
        'status': 'human_qa_32_of_32_pass', 'human_quality_gate_passed': True,
        'planned_clips': 32, 'passed_clips': 32})
    review = root / 'qa_review.md'
    review.write_text(f'Decision: GO\nExact SHA {qa_sha}\n')
    geometry = {'longest_alias': {'token_ids_with_eos': [7, 8, 2]}}
    preflight = root / 'preflight.json'
    write_json(preflight, {'v6_synthetic_zero_geometry': geometry})
    study = audit.HERE / 'voxtral_fp32_track_b_study_runner_v1.py'
    worker = audit.HERE / 'voxtral_track_b_v6_pilot_worker_source_v1.py'
    bindings = {
        'protocol_sha256': audit.PROTOCOL_SHAS,
        'qa_sha256': audit.sha(qa), 'qa_review_sha256': audit.sha(review),
        'preflight_sha256': audit.sha(preflight),
        'study_runner_sha256': audit.sha(study), 'pilot_worker_sha256': audit.sha(worker),
        'pilot_auditor_sha256': audit.sha(Path(audit.__file__)),
        'synthetic_geometry_sha256': hashlib.sha256(json.dumps(
            geometry, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'synthetic_batch_sha256': 'a' * 64, 'old_fit_batch_sha256': 'b' * 64,
        'campaign_attempt_id': 'fake_1',
        'campaign_claim_path': str(registry / 'claim.json'),
    }
    write_json(Path(bindings['campaign_claim_path']), {
        'schema': 'voxtral_track_b_v6_campaign_claim_v1',
        'campaign_attempt_id': 'fake_1', 'output_dir': str(directory.resolve()),
        'bindings_sha256': hashlib.sha256(json.dumps(
            bindings, sort_keys=True, separators=(',', ':')).encode()).hexdigest()})
    admission = root / 'admission.json'
    write_json(admission, {'schema': 'voxtral_track_b_v6_external_pilot_admission_v1',
                           'status': 'pilot_go', 'bindings': bindings})
    host = b'fake bounded host log; auditor must still HOLD for external parsing'
    (directory / 'host_pre.raw').write_bytes(host)
    (directory / 'host_post.raw').write_bytes(host)
    rows = []
    for ordinal, label in enumerate(audit.ORDER, 1):
        rows.append({'ordinal': ordinal, 'class': label, 'phase': 'started'})
        done = {'ordinal': ordinal, 'class': label, 'phase': 'completed',
                'elapsed_seconds': .01, 'observed_postcall_mps_driver_bytes': 1024,
                'hook_calls': 1, 'projector_pre_sha256': 'c' * 64}
        if label.startswith('score_'):
            done.update(score_finite=True, score_sha256='d' * 64)
            if label == 'score_grad':
                done.update(gradient_finite=True, gradient_sha256='e' * 64)
        else:
            done.update(suffix_sha256='f' * 64,
                        suffix_length=32 if label.startswith('forced_') else 2,
                        projector_post_sha256='c' * 64)
        rows.append(done)
    calls = directory / 'calls.jsonl'
    calls.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    tg = ts = tb = .01
    p = 1.0
    w = 7 * tg + 72 * ts + 12 * tb
    receipt = {
        'schema': audit.SCHEMA, 'status': 'schedule_complete_resource_unverified',
        'bindings': bindings,
        'planned_calls': audit.COUNTS, 'attempted_calls': audit.COUNTS,
        'completed_calls': audit.COUNTS,
        'missing_calls': {k: 0 for k in audit.COUNTS},
        'attempted_call_count': 88, 'calls_sha256': audit.sha(calls),
        'host_pre_sha256': audit.sha(directory / 'host_pre.raw'),
        'host_post_sha256': audit.sha(directory / 'host_post.raw'),
        'host_pre': {'raw_power_log_sha256': audit.sha(directory / 'host_pre.raw'),
                     'raw_power_log_bytes': len(host), 'ac': True,
                     'lid_open': True, 'ten_minute_clean': True},
        'host_post': {'raw_power_log_sha256': audit.sha(directory / 'host_post.raw'),
                      'raw_power_log_bytes': len(host), 'ac': True,
                      'lid_open': True, 'ten_minute_clean': True,
                      'during_run_clean': True},
        'clock_gap_seconds': 0.0, 'max_observed_postcall_mps_driver_bytes': 1024,
        'measured_mps_high_water_present': False,
        'hard_per_call_watchdog_present': False,
        'synthetic_suffix_length': 32,
        'synthetic_input_sha256': bindings['synthetic_batch_sha256'],
        'old_fit_input_sha256': bindings['old_fit_batch_sha256'],
        'longest_alias_ids_with_eos': [7, 8, 2],
        'timing': {'t_g': tg, 't_s': ts, 't_b': tb, 'P': p,
                   'per_context_1p2_W': 1.2 * w,
                   'campaign_1p2_P_plus_16W': 1.2 * (p + 16 * w)},
        'model_work_seconds': p,
    }
    write_json(directory / 'pilot_measurement.json', receipt)
    args = dict(admission=admission, qa=qa, qa_review=review, preflight=preflight,
                study_runner=study, worker=worker, free_disk_bytes=100 * 1024**3,
                expected_measurement_sha256=audit.sha(directory / 'pilot_measurement.json'),
                expected_admission_sha256=audit.sha(admission))
    return directory, args


class SavedAuditTests(unittest.TestCase):
    def test_valid_saved_schedule_remains_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, args = make_fixture(Path(tmp))
            got = audit.audit_saved(directory, **args)
            self.assertEqual(got['status'], 'HOLD_no_post_pilot_go')
            self.assertFalse(got['go_issued'])
            self.assertTrue(got['timing_feasible'])
            self.assertEqual(got['ledger']['counts'], audit.COUNTS)
            self.assertIn('true_within_call_mps_high_water_absent', got['hold_reasons'])
            self.assertIn('hard_per_call_watchdog_absent', got['hold_reasons'])

    def test_rejects_ledger_tamper_and_rehashed_wrong_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, args = make_fixture(Path(tmp))
            calls = directory / 'calls.jsonl'
            calls.write_text(calls.read_text().replace('"phase": "completed"',
                                                      '"phase": "failed"', 1))
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                audit.audit_saved(directory, **args)
            receipt_path = directory / 'pilot_measurement.json'
            receipt = audit.read_json(receipt_path)
            receipt['calls_sha256'] = audit.sha(calls)
            write_json(receipt_path, receipt)
            args['expected_measurement_sha256'] = audit.sha(receipt_path)
            with self.assertRaisesRegex(ValueError, 'completion'):
                audit.audit_saved(directory, **args)

    def test_rejects_false_resource_attestation_and_bad_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, args = make_fixture(Path(tmp))
            receipt_path = directory / 'pilot_measurement.json'
            receipt = audit.read_json(receipt_path)
            receipt['measured_mps_high_water_present'] = True
            write_json(receipt_path, receipt)
            args['expected_measurement_sha256'] = audit.sha(receipt_path)
            with self.assertRaisesRegex(ValueError, 'cannot attest'):
                audit.audit_saved(directory, **args)
            receipt['measured_mps_high_water_present'] = False
            receipt['timing']['per_context_1p2_W'] += 1
            write_json(receipt_path, receipt)
            args['expected_measurement_sha256'] = audit.sha(receipt_path)
            with self.assertRaisesRegex(ValueError, 'recomputation'):
                audit.audit_saved(directory, **args)

    def test_rejects_host_and_claim_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, args = make_fixture(Path(tmp))
            (directory / 'host_post.raw').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'Host post'):
                audit.audit_saved(directory, **args)
        with tempfile.TemporaryDirectory() as tmp:
            directory, args = make_fixture(Path(tmp))
            claim = Path(audit.read_json(args['admission'])['bindings']['campaign_claim_path'])
            row = audit.read_json(claim)
            row['output_dir'] = '/other'
            write_json(claim, row)
            with self.assertRaisesRegex(ValueError, 'campaign claim'):
                audit.audit_saved(directory, **args)

    def test_rejects_failure_artifact_and_unpinned_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, args = make_fixture(Path(tmp))
            args['expected_measurement_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'Externally pinned'):
                audit.audit_saved(directory, **args)
            args['expected_measurement_sha256'] = audit.sha(directory / 'pilot_measurement.json')
            write_json(directory / 'pilot_failure.json', {'status': 'failed_no_retry'})
            with self.assertRaisesRegex(ValueError, 'failure artifact'):
                audit.audit_saved(directory, **args)


if __name__ == '__main__':
    unittest.main()
