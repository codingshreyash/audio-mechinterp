"""Fake-artifact admission tests. No candidate weights, audio or model inference."""

from __future__ import annotations

import base64
import gzip
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from models.second_architecture import voxtral_track_b_v6_pilot_worker_source_v1 as worker
from models.second_architecture import voxtral_track_b_v6_pre_pilot_admission_contract_v1 as contract
from models.second_architecture import voxtral_track_b_v6_pre_pilot_launcher_source_v1 as launcher


def write_json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, sort_keys=True))
    return worker.sha_file(path)


def fresh_host(*, old: bool = False) -> dict:
    raw = b'fake raw host log, not a real admission'
    packed = gzip.compress(raw, mtime=0)
    when = datetime.now(timezone.utc) - timedelta(minutes=5 if old else 0)
    return {'eligible_at_collection': True, 'collected_utc': when.isoformat(),
            'power_log_gzip_base64': base64.b64encode(packed).decode(),
            'power_log_gzip_sha256': worker.sha_bytes(packed),
            'power_log_raw_bytes': len(raw),
            'output_sha256': {'power_log': worker.sha_bytes(raw)}}


class Fixture:
    def __init__(self, directory: Path):
        self.directory = directory
        geometry = {'longest_alias': {'token_ids_with_eos': list(worker.LONGEST_ALIAS)},
                    'input_features_shape': [1, 128, 3000],
                    'audio_placeholder_count': 375, 'projector_rows_implied': 375,
                    'attention_mask_length': 405}
        self.frozen = {'schema': 'voxtral_fp32_track_b_no_weight_preflight_v1',
                       'candidate_weights_loaded': False, 'candidate_forward_count': 0,
                       'human_qa': {'passed': True},
                       'v6_synthetic_zero_geometry': geometry,
                       'snapshot': {'snapshot_tree': {'sha256': 'a' * 64}},
                       'installed_source_sha256': {'modeling_voxtral': 'b' * 64}}
        for key in launcher.STABLE_PREFLIGHT_FIELDS:
            self.frozen.setdefault(key, {'fake': key})
        self.frozen['host'] = fresh_host()
        qa = directory / 'qa.json'
        qa_sha = write_json(qa, {'schema': 'word_disjoint_human_qa_audit_v1',
                                  'status': 'human_qa_32_of_32_pass',
                                  'human_quality_gate_passed': True,
                                  'planned_clips': 32, 'passed_clips': 32})
        review_qa = directory / 'qa_review.md'
        review_qa.write_text(f'Decision: GO\nExact QA SHA: {qa_sha}\n')
        preflight = directory / 'preflight.json'
        preflight_sha = write_json(preflight, self.frozen)
        runner = directory / 'runner.py'
        runner.write_text('# fake frozen runner\n')
        auditor = directory / 'auditor.py'
        auditor.write_text('# fake frozen auditor\n')
        self.files = worker.AdmissionFiles(directory / 'admission.json', qa,
                                           review_qa, preflight, runner, auditor)
        self.qa = {'qa_go_receipt_sha256': 'e' * 64,
                   'qa_review_sha256': worker.sha_file(review_qa),
                   'human_qa_audit_sha256': qa_sha}
        registry = directory / 'registry'
        registry.mkdir()
        self.bindings = {
            'protocol_sha256': worker.PROTOCOL_SHAS,
            'qa_sha256': qa_sha, 'qa_go_receipt_sha256': self.qa['qa_go_receipt_sha256'],
            'human_qa_audit_sha256': qa_sha,
            'qa_review_sha256': self.qa['qa_review_sha256'],
            'preflight_sha256': preflight_sha,
            'study_runner_sha256': worker.sha_file(runner),
            'pilot_auditor_sha256': worker.sha_file(auditor),
            'pilot_worker_sha256': worker.sha_file(Path(worker.__file__)),
            'synthetic_geometry_sha256': worker.sha_bytes(json.dumps(
                geometry, sort_keys=True, separators=(',', ':')).encode()),
            'synthetic_processor_geometry': {
                'input_features_shape': [1, 128, 3000],
                'audio_placeholder_count': 375, 'attention_mask_length': 405},
            'checkpoint_tree_sha256': self.frozen['snapshot']['snapshot_tree']['sha256'],
            'transformers_source_sha256': worker.sha_bytes(json.dumps(
                self.frozen['installed_source_sha256'], sort_keys=True,
                separators=(',', ':')).encode()),
            'synthetic_batch_sha256': 'c' * 64,
            'old_fit_batch_sha256': 'd' * 64,
            'campaign_attempt_id': 'source_only_fake_attempt',
            'campaign_claim_path': str(registry / 'attempt.json'),
        }
        self.admission_sha = write_json(self.files.admission, {
            'schema': 'voxtral_track_b_v6_external_pilot_admission_v1',
            'status': 'pilot_go', 'bindings': self.bindings})
        self.review_path = directory / 'independent_review.json'
        self.review = {
            'schema': 'voxtral_track_b_v6_pre_pilot_independent_review_v1',
            'decision': 'GO_for_pilot_only',
            'admission_sha256': self.admission_sha,
            'worker_sha256': self.bindings['pilot_worker_sha256'],
            'campaign_attempt_id': self.bindings['campaign_attempt_id'],
            'campaign_claim_path': self.bindings['campaign_claim_path'],
        }
        review_sha = write_json(self.review_path, self.review)
        self.pins = contract.ExternalPins(self.bindings['pilot_worker_sha256'],
                                          self.admission_sha, review_sha)

    def repin_admission_and_review(self):
        self.admission_sha = write_json(self.files.admission, {
            'schema': 'voxtral_track_b_v6_external_pilot_admission_v1',
            'status': 'pilot_go', 'bindings': self.bindings})
        self.review['admission_sha256'] = self.admission_sha
        review_sha = write_json(self.review_path, self.review)
        self.pins = contract.ExternalPins(self.bindings['pilot_worker_sha256'],
                                          self.admission_sha, review_sha)


class PrePilotContractTests(unittest.TestCase):
    def test_fresh_host_rejects_compressed_expansion_over_bound(self):
        raw = b'x' * 65
        packed = gzip.compress(raw, mtime=0)
        host = fresh_host()
        host.update(power_log_gzip_base64=base64.b64encode(packed).decode(),
                    power_log_gzip_sha256=worker.sha_bytes(packed),
                    power_log_raw_bytes=len(raw),
                    output_sha256={'power_log': worker.sha_bytes(raw)})
        with patch.object(launcher, 'MAX_HOST_RAW_BYTES', 64):
            with self.assertRaisesRegex(ValueError, 'exceeds 128 MiB'):
                launcher._verify_fresh_host(host)

    def test_external_chain_qa_processor_and_fresh_host(self):
        with tempfile.TemporaryDirectory() as temp:
            f = Fixture(Path(temp))
            with patch.object(launcher, '_actual_qa', return_value=f.qa), patch.object(
                    launcher, '_fresh_no_weight_preflight', return_value=f.frozen):
                result = launcher.verify_pre_pilot(files=f.files,
                    review_path=f.review_path, pins=f.pins)
            self.assertEqual(result['campaign_attempt_id'], f.bindings['campaign_attempt_id'])
            self.assertFalse(Path(f.bindings['campaign_claim_path']).exists())

    def test_forged_missing_or_changed_bindings_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            f = Fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, 'review.*bytes differ'):
                contract.verify_contract(f.files, f.review_path,
                    contract.ExternalPins(f.pins.worker_sha256, f.pins.admission_sha256, '0' * 64))
            f.bindings.pop('campaign_attempt_id')
            write_json(f.files.admission, {'schema': 'voxtral_track_b_v6_external_pilot_admission_v1',
                                          'status': 'pilot_go', 'bindings': f.bindings})
            with self.assertRaisesRegex(ValueError, 'admission bytes differ'):
                contract.verify_contract(f.files, f.review_path, f.pins)
            self.assertFalse(Path(f.bindings['campaign_claim_path']).exists())

    def test_changed_actual_qa_fingerprint_or_stale_host_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            f = Fixture(Path(temp))
            wrong_qa = {**f.qa, 'qa_go_receipt_sha256': 'f' * 64}
            with patch.object(launcher, '_actual_qa', return_value=wrong_qa), patch.object(
                    launcher, '_fresh_no_weight_preflight', return_value=f.frozen):
                with self.assertRaisesRegex(ValueError, 'Actual human QA'):
                    launcher.verify_pre_pilot(files=f.files, review_path=f.review_path, pins=f.pins)
            wrong_fingerprint = {**f.frozen, 'processor': {'changed': True}}
            with patch.object(launcher, '_actual_qa', return_value=f.qa), patch.object(
                    launcher, '_fresh_no_weight_preflight', return_value=wrong_fingerprint):
                with self.assertRaisesRegex(ValueError, 'processor fingerprint changed'):
                    launcher.verify_pre_pilot(files=f.files, review_path=f.review_path, pins=f.pins)
            stale = {**f.frozen, 'host': fresh_host(old=True)}
            with patch.object(launcher, '_actual_qa', return_value=f.qa), patch.object(
                    launcher, '_fresh_no_weight_preflight', return_value=stale):
                with self.assertRaisesRegex(ValueError, 'Fresh host timestamp'):
                    launcher.verify_pre_pilot(files=f.files, review_path=f.review_path, pins=f.pins)
            malformed = {**f.frozen, 'host': {**fresh_host(), 'output_sha256': 'forged'}}
            with patch.object(launcher, '_actual_qa', return_value=f.qa), patch.object(
                    launcher, '_fresh_no_weight_preflight', return_value=malformed):
                with self.assertRaisesRegex(ValueError, 'Fresh host timestamp or raw log'):
                    launcher.verify_pre_pilot(files=f.files, review_path=f.review_path, pins=f.pins)

    def test_repinned_forged_checkpoint_or_source_digest_still_fails(self):
        for key in ('checkpoint_tree_sha256', 'transformers_source_sha256'):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                f = Fixture(Path(temp))
                f.bindings[key] = 'f' * 64
                f.repin_admission_and_review()
                self.assertEqual(contract.verify_contract(
                    f.files, f.review_path, f.pins)['bindings'][key], 'f' * 64)
                with patch.object(launcher, '_actual_qa', return_value=f.qa), patch.object(
                        launcher, '_fresh_no_weight_preflight', return_value=f.frozen):
                    with self.assertRaisesRegex(ValueError, 'Admission checkpoint or Transformers'):
                        launcher.verify_pre_pilot(files=f.files,
                                                  review_path=f.review_path, pins=f.pins)

    def test_public_launcher_never_calls_loader_even_with_fake_pins(self):
        with tempfile.TemporaryDirectory() as temp:
            f = Fixture(Path(temp))
            calls = []
            with patch.object(launcher, 'FROZEN_PINS', f.pins), patch.object(
                    launcher, '_actual_qa', return_value=f.qa), patch.object(
                    launcher, '_fresh_no_weight_preflight', return_value=f.frozen):
                with self.assertRaisesRegex(ValueError, 'watchdog and MPS peak sampler absent'):
                    launcher.launch_pilot(files=f.files, review_path=f.review_path,
                                          load_model=lambda: calls.append(1))
            self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
