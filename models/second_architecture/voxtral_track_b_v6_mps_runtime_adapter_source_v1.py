"""Source-only Mac MPS runtime adapter for the V6 pilot; public launch is HOLD.

The parent owns a one-attempt claim and hard process deadline. The child sets
the MPS allocator cap before its operation and samples driver allocation while
the operation runs. Samples are lower bounds on the true transient peak;
the configured allocator ceiling is the separate enforceable upper bound.
No real model loader or public worker delegation is exposed by this source.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from models.second_architecture import voxtral_track_b_v6_pilot_worker_source_v1 as worker


CAP_BYTES = 30 * 1024**3
DEADLINE_SECONDS = 240.0
SAMPLE_INTERVAL_SECONDS = 0.02


def configure_mps_allocator(mps: Any) -> dict[str, Any]:
    """Set the 30-GiB PyTorch MPS allocator ceiling before a model load."""
    recommended = int(mps.recommended_max_memory())
    if recommended < CAP_BYTES:
        raise MemoryError('MPS recommended memory below required 30-GiB pilot cap')
    fraction = CAP_BYTES / recommended
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('MPS memory fraction invalid')
    mps.set_per_process_memory_fraction(fraction)
    return {'recommended_max_memory_bytes': recommended,
            'configured_allocator_cap_bytes': CAP_BYTES,
            'configured_fraction': fraction,
            'allocator_cap_kind': 'PyTorch MPS allocation limit; not exact device peak'}


@dataclass
class SampledMPSHighWater:
    mps: Any
    calls_path: Path
    interval_seconds: float = SAMPLE_INTERVAL_SECONDS

    def __post_init__(self) -> None:
        if not 0 < self.interval_seconds <= 0.1:
            raise ValueError('MPS sampler interval invalid')
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.maximum = 0
        self.samples = 0
        self.by_ordinal: dict[int, int] = {}
        self._active: int | None = None
        self.error: str | None = None

    def _read_active_call(self) -> None:
        if not self.calls_path.is_file():
            return
        # The V6 pilot ledger is bounded at 176 rows, so rereading is bounded.
        raw = self.calls_path.read_text()
        if raw and not raw.endswith('\n'):
            raw = raw[:raw.rfind('\n') + 1]  # append in flight; only final row may be partial
        rows = raw.splitlines()
        if len(rows) > 200:
            raise ValueError('MPS sampler call ledger exceeds V6 bound')
        active: int | None = None
        for line in rows:
            row = json.loads(line)
            if row.get('phase') == 'started':
                active = int(row['ordinal'])
            elif row.get('phase') in ('completed', 'failed'):
                active = None
        self._active = active

    def sample_once(self) -> None:
        self._read_active_call()
        observed = int(self.mps.driver_allocated_memory())
        if observed < 0:
            raise ValueError('MPS driver allocation negative')
        self.maximum = max(self.maximum, observed)
        self.samples += 1
        if self._active is not None:
            self.by_ordinal[self._active] = max(self.by_ordinal.get(self._active, 0), observed)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample_once()
            except BaseException as exc:
                self.error = f'{type(exc).__name__}: {exc}'
                self._stop.set()
                return
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self.sample_once()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def finish(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            if self._thread.is_alive():
                raise TimeoutError('MPS sampler thread did not stop')
        self.mps.synchronize()
        self.sample_once()
        if self.error is not None or self.samples < 2:
            raise ValueError(f'MPS sampler failed: {self.error}')
        return {'sampled_driver_highwater_lower_bound_bytes': self.maximum,
                'sample_count': self.samples, 'sample_interval_seconds': self.interval_seconds,
                'sampled_lower_bound_by_call_ordinal': self.by_ordinal,
                'exact_transient_peak_measured': False}


def _child_entry(operation: Callable[[], dict[str, Any]], mps: Any | None,
                 calls_path: Path, results: Any) -> None:
    sampler: SampledMPSHighWater | None = None
    try:
        if mps is None:
            import torch
            mps = torch.mps  # import in spawned child; do not pickle torch.mps
        cap = configure_mps_allocator(mps)
        sampler = SampledMPSHighWater(mps, calls_path)
        sampler.start()
        outcome = operation()
        if not isinstance(outcome, dict) or outcome.get('status') not in (
                'schedule_complete_resource_unverified', 'fake_complete'):
            raise ValueError('Child operation status invalid')
        samples = sampler.finish()
        if samples['sampled_driver_highwater_lower_bound_bytes'] > CAP_BYTES:
            raise MemoryError('Observed MPS allocation exceeds configured cap')
        results.put({'status': 'child_completed', 'cap': cap, 'samples': samples,
                     'operation_status': outcome['status']})
    except BaseException as exc:
        if sampler is not None:
            try:
                sampler.finish()
            except BaseException:
                pass
        results.put({'status': 'child_failed', 'error_type': type(exc).__name__,
                     'error': str(exc)})


def _claim_path(campaign_claim_path: Path) -> Path:
    return campaign_claim_path.with_name(campaign_claim_path.name + '.watchdog')


def _guarded_child_attempt(*, campaign_claim_path: Path, output_dir: Path,
                           operation: Callable[[], dict[str, Any]], mps: Any,
                           deadline_seconds: float = DEADLINE_SECONDS,
                           context: Any | None = None) -> dict[str, Any]:
    """Private fake-testable boundary; never called by public launch."""
    if not 0 < deadline_seconds <= DEADLINE_SECONDS:
        raise ValueError('V6 deadline invalid')
    if (not campaign_claim_path.is_absolute() or not campaign_claim_path.parent.is_dir() or
            campaign_claim_path.is_symlink() or output_dir.exists() or
            campaign_claim_path.resolve().is_relative_to(output_dir.resolve())):
        raise ValueError('V6 attempt paths invalid or already used')
    claim = _claim_path(campaign_claim_path)
    result_path = claim.with_name(claim.name + '.result.json')
    start = time.monotonic()
    worker._exclusive_json(claim, {'schema': 'voxtral_track_b_v6_parent_deadline_claim_v1',
                                   'campaign_claim_path': str(campaign_claim_path),
                                   'output_dir': str(output_dir.resolve()),
                                   'deadline_seconds': deadline_seconds})
    ctx = context or mp.get_context('spawn')
    results = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_child_entry,
                          args=(operation, mps, output_dir / 'calls.jsonl', results))
    try:
        process.start()
        remaining = max(0.0, deadline_seconds - (time.monotonic() - start))
        process.join(remaining)
        if process.is_alive():
            process.terminate()
            process.join(2)
            if process.is_alive():
                process.kill()
                process.join(2)
            outcome = {'status': 'timeout_no_retry', 'error_type': 'TimeoutError'}
        else:
            try:
                outcome = results.get_nowait()
            except queue.Empty:
                outcome = {'status': 'child_failed', 'error_type': 'NoResult',
                           'exit_code': process.exitcode}
        outcome['elapsed_seconds'] = time.monotonic() - start
        outcome['child_exit_code'] = process.exitcode
        if outcome['elapsed_seconds'] > deadline_seconds and outcome['status'] == 'child_completed':
            outcome = {'status': 'timeout_no_retry', 'error_type': 'DeadlineExceeded',
                       'elapsed_seconds': outcome['elapsed_seconds'],
                       'child_exit_code': process.exitcode}
        worker._exclusive_json(result_path, outcome)
        return outcome
    except BaseException as exc:
        if process.is_alive():
            process.terminate()
            process.join(2)
            if process.is_alive():
                process.kill()
                process.join(2)
        if not result_path.exists():
            worker._exclusive_json(result_path, {'status': 'child_failed',
                                                 'error_type': type(exc).__name__,
                                                 'error': str(exc)})
        raise
    finally:
        results.close()


def launch_v6_pilot(*_args: Any, **_kwargs: Any) -> None:
    """Public entry: no real load until QA, pins, loader and audit are reviewed."""
    raise ValueError('V6 MPS runtime adapter HOLD: no reviewed live launcher')
