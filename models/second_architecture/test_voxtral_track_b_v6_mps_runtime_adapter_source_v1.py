"""Fake-process tests for held MPS runtime adapter; no model or GPU."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from models.second_architecture import voxtral_track_b_v6_mps_runtime_adapter_source_v1 as runtime


class FakeMPS:
    def __init__(self, recommended: int = 40 * 1024**3):
        self.recommended = recommended
        self.fraction = None

    def recommended_max_memory(self):
        return self.recommended

    def set_per_process_memory_fraction(self, value):
        self.fraction = value

    def driver_allocated_memory(self):
        return 1024

    def synchronize(self):
        return None


class QueueSetupFailure:
    def Queue(self, **_kwargs):
        raise OSError('fake queue setup failed')


class FakeQueue:
    def close(self):
        return None


class UnkillableProcess:
    pid = 98765
    exitcode = None

    def start(self):
        return None

    def join(self, _timeout):
        return None

    def is_alive(self):
        return True

    def terminate(self):
        return None

    def kill(self):
        return None


class UnkillableContext:
    def Queue(self, **_kwargs):
        return FakeQueue()

    def Process(self, **_kwargs):
        return UnkillableProcess()


def fake_complete():
    return {'status': 'fake_complete'}


def fake_oom():
    raise MemoryError('fake allocator OOM')


def fake_hang():
    time.sleep(10)
    return {'status': 'fake_complete'}


class MPSRuntimeSourceTests(unittest.TestCase):
    def test_allocator_fraction_and_recommended_limit(self):
        mps = FakeMPS()
        cap = runtime.configure_mps_allocator(mps)
        self.assertEqual(cap['configured_allocator_cap_bytes'], 30 * 1024**3)
        self.assertAlmostEqual(mps.fraction, 0.75)
        with self.assertRaisesRegex(MemoryError, 'recommended memory below'):
            runtime.configure_mps_allocator(FakeMPS(20 * 1024**3))

    def test_sampler_tolerates_only_incomplete_final_ledger_row(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = Path(temp) / 'calls.jsonl'
            calls.write_text('{"ordinal":1,"phase":"started"}\n{"ordinal":1,"phase":"com')
            sampler = runtime.SampledMPSHighWater(FakeMPS(), calls)
            sampler.sample_once()
            self.assertEqual(sampler.by_ordinal, {1: 1024})
            calls.write_text('{"ordinal":1,"phase":"started"}\n'
                             '{"ordinal":1,"phase":"completed"}\n')
            sampler.sample_once()
            self.assertIsNone(sampler._active)

    def test_fake_process_success_and_no_retry_at_alternate_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            claim = root / 'campaign.json'
            result = runtime._guarded_child_attempt(
                campaign_claim_path=claim, output_dir=root / 'pilot_a',
                operation=fake_complete, mps=FakeMPS(), deadline_seconds=10)
            self.assertEqual(result['status'], 'child_completed')
            self.assertEqual(result['cap']['configured_allocator_cap_bytes'], 30 * 1024**3)
            self.assertFalse(result['samples']['exact_transient_peak_measured'])
            self.assertGreaterEqual(result['samples']['sample_count'], 2)
            self.assertTrue(runtime._claim_path(claim).is_file())
            self.assertTrue(runtime._claim_path(claim).with_name(
                runtime._claim_path(claim).name + '.result.json').is_file())
            with self.assertRaises(FileExistsError):
                runtime._guarded_child_attempt(
                    campaign_claim_path=claim, output_dir=root / 'pilot_b',
                    operation=fake_complete, mps=FakeMPS(), deadline_seconds=10)
            self.assertFalse((root / 'pilot_b').exists())

    def test_fake_oom_retains_claim_and_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            claim = root / 'campaign.json'
            result = runtime._guarded_child_attempt(
                campaign_claim_path=claim, output_dir=root / 'pilot',
                operation=fake_oom, mps=FakeMPS(), deadline_seconds=10)
            self.assertEqual((result['status'], result['error_type']),
                             ('child_failed', 'MemoryError'))
            self.assertTrue(runtime._claim_path(claim).is_file())

    def test_fake_hang_is_killed_and_claim_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            claim = root / 'campaign.json'
            result = runtime._guarded_child_attempt(
                campaign_claim_path=claim, output_dir=root / 'pilot',
                operation=fake_hang, mps=FakeMPS(), deadline_seconds=0.2)
            self.assertEqual(result['status'], 'timeout_no_retry')
            self.assertTrue(runtime._claim_path(claim).is_file())
            saved = json.loads(runtime._claim_path(claim).with_name(
                runtime._claim_path(claim).name + '.result.json').read_text())
            self.assertEqual(saved['status'], 'timeout_no_retry')

    def test_queue_setup_failure_retains_claim_and_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            claim = root / 'campaign.json'
            with self.assertRaisesRegex(OSError, 'fake queue setup failed'):
                runtime._guarded_child_attempt(
                    campaign_claim_path=claim, output_dir=root / 'pilot',
                    operation=fake_complete, mps=FakeMPS(),
                    deadline_seconds=10, context=QueueSetupFailure())
            marker = runtime._claim_path(claim)
            self.assertTrue(marker.is_file())
            saved = json.loads(marker.with_name(marker.name + '.result.json').read_text())
            self.assertEqual((saved['status'], saved['error_type']), ('child_failed', 'OSError'))

    def test_unconfirmed_termination_is_explicit_hold(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            claim = root / 'campaign.json'
            result = runtime._guarded_child_attempt(
                campaign_claim_path=claim, output_dir=root / 'pilot',
                operation=fake_complete, mps=FakeMPS(),
                deadline_seconds=0.01, context=UnkillableContext())
            self.assertEqual(result['status'], 'termination_unconfirmed')
            self.assertEqual(result['child_pid'], 98765)
            self.assertTrue(result['child_alive'])

    def test_public_entry_rejects_before_supplied_loader(self):
        called = []
        with self.assertRaisesRegex(ValueError, 'HOLD'):
            runtime.launch_v6_pilot(load_model=lambda: called.append(1))
        self.assertEqual(called, [])


if __name__ == '__main__':
    unittest.main()
