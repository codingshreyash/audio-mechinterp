"""Source-only Track B campaign supervisor. Public execution always HOLDs.

The private callback schedule exists for fake CPU tests and source review. A
separate frozen live launcher must provide a hard per-call watchdog and an MPS
high-water sampler before any candidate model may be loaded.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

import voxtral_fp32_track_b_study_runner_v1 as study
import voxtral_track_b_answer_link_core_v1 as core

SCHEMA = 'voxtral_track_b_campaign_supervisor_source_v1'
GO_SCHEMA = 'voxtral_track_b_v6_postpilot_independent_go_v1'


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _read_bound(path: Path, expected: str, label: str) -> dict:
    _need(path.is_file() and sha(path) == expected, f'{label} hash mismatch')
    value = json.loads(path.read_text())
    _need(isinstance(value, dict), f'{label} must be an object')
    return value


def verify_postpilot_go(*, go_path: Path, go_sha: str, measurement_path: Path,
                        measurement_sha: str, calls_path: Path, calls_sha: str,
                        qa_path: Path, qa_sha: str, preflight_path: Path,
                        preflight_sha: str, study_source_sha: str,
                        supervisor_sha: str) -> dict:
    """Verify an externally pinned independent GO, never self-issue one."""
    for value in (go_sha, measurement_sha, calls_sha, qa_sha, preflight_sha,
                  study_source_sha, supervisor_sha):
        _need(isinstance(value, str) and len(value) == 64 and
              all(c in '0123456789abcdef' for c in value), 'Invalid SHA-256 pin')
    _need(sha(Path(__file__)) == supervisor_sha and
          sha(Path(study.__file__)) == study_source_sha,
          'Supervisor/study source differs from external pin')
    go = _read_bound(go_path, go_sha, 'post-pilot GO')
    measurement = _read_bound(measurement_path, measurement_sha, 'pilot measurement')
    _read_bound(qa_path, qa_sha, '32/32 QA')
    _read_bound(preflight_path, preflight_sha, 'no-weight preflight')
    _need(calls_path.is_file() and sha(calls_path) == calls_sha,
          'Pilot call log hash mismatch')
    _need(go.get('schema') == GO_SCHEMA and go.get('status') == 'GO' and
          go.get('independent_reviewer') and go.get('external_anchor_sha256'),
          'Post-pilot independent GO absent')
    bindings = go.get('bindings')
    expected = {'pilot_measurement_sha256': measurement_sha,
                'pilot_calls_sha256': calls_sha, 'qa_sha256': qa_sha,
                'preflight_sha256': preflight_sha,
                'study_source_sha256': study_source_sha,
                'supervisor_source_sha256': supervisor_sha}
    _need(isinstance(bindings, dict) and
          all(bindings.get(k) == v for k, v in expected.items()),
          'Post-pilot GO does not bind exact study bytes')
    _need(measurement.get('schema') == 'voxtral_track_b_v6_pilot_measurement_v1' and
          measurement.get('status') == 'complete_feasible' and
          measurement.get('calls_sha256') == calls_sha and
          measurement.get('hard_per_call_watchdog_present') is True and
          measurement.get('measured_mps_high_water_present') is True,
          'Measured pilot is not independently admissible')
    _need(measurement.get('bindings', {}).get('qa_sha256') == qa_sha and
          measurement.get('bindings', {}).get('preflight_sha256') == preflight_sha,
          'Pilot QA/preflight binding differs')
    return {'go_sha256': go_sha, 'measurement_sha256': measurement_sha,
            'calls_sha256': calls_sha, 'qa_sha256': qa_sha,
            'preflight_sha256': preflight_sha,
            'attempt_id': bindings.get('campaign_attempt_id'),
            'claim_path': bindings.get('campaign_claim_path')}


def run_campaign(*, load_model: Callable[[], Any], **gate: Any) -> None:
    """Fail closed before loader, including when a synthetic GO is supplied."""
    verify_postpilot_go(**gate)
    raise RuntimeError('HOLD: live hard watchdog and MPS high-water sampler absent; loader not called')


def _write_json(path: Path, value: Any) -> None:
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _append_jsonl(path: Path, value: Any) -> None:
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False) + '\n').encode()
    with path.open('ab') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _save_sidecar(path: Path, tensor: torch.Tensor) -> dict:
    _need(tensor.dtype == torch.float32 and tensor.device.type == 'cpu' and
          tensor.ndim == 2 and tensor.shape[1] == core.PROJECTOR_WIDTH and
          bool(torch.isfinite(tensor).all()), 'Invalid selected-gradient tensor')
    with path.open('xb') as stream:
        torch.save(tensor.contiguous(), stream)
        stream.flush()
        os.fsync(stream.fileno())
    reloaded = torch.load(path, map_location='cpu', weights_only=True)
    _need(core.exact_equal(reloaded, tensor), 'Lossless gradient sidecar replay failed')
    return {'path': path.name, 'file_sha256': sha(path),
            'tensor_sha256': core.tensor_sha(tensor),
            'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'l2': float(torch.linalg.vector_norm(tensor))}


def _execute_fake_schedule(*, plan: list[dict], output_dir: Path,
                           attempt_id: str, claim_path: Path,
                           clean: Callable[[dict, Callable[[str], None]], tuple[dict, torch.Tensor]],
                           complete: Callable[[dict, torch.Tensor, torch.Tensor, dict, dict,
                                               Callable[[str], None]], tuple[dict, dict]],
                           host_probe: Callable[[], Mapping[str, bool]],
                           mps_bytes: Callable[[], int],
                           free_bytes: Callable[[Path], int],
                           now: Callable[[], float] = time.monotonic,
                           wall: Callable[[], float] = time.time) -> dict:
    """Fake-only 16-context schedule; not a live admission or model launcher."""
    expected_contexts = []
    for pair_no, pair in enumerate(study.preflight.PAIR_ORDER):
        for engine in study.preflight.ENGINES:
            for recipient, donor in (pair, tuple(reversed(pair))):
                expected_contexts.append((f'pair_{pair_no:02d}', engine,
                                          recipient, donor,
                                          f'pair_{pair_no:02d}|{engine}|{recipient}->{donor}'))
    _need(len(plan) == 16 and [p['index'] for p in plan] == list(range(16)) and
          [(p['pair_id'], p['engine'], p['recipient'], p['donor'], p['context_id'])
           for p in plan] == expected_contexts,
          'Full fixed 16-context plan required')
    _need(attempt_id and len(attempt_id) <= 128 and
          all(c.isalnum() or c in '_-' for c in attempt_id), 'Invalid attempt ID')
    _need(claim_path.is_absolute() and claim_path.parent.is_dir() and
          not claim_path.is_symlink() and not output_dir.exists() and
          not claim_path.resolve().is_relative_to(output_dir.resolve()),
          'Stable external claim or fresh output required')
    _write_json(claim_path, {'schema': SCHEMA + '_claim', 'attempt_id': attempt_id,
                             'output_dir': str(output_dir.resolve()),
                             'plan_sha256': _digest(plan)})
    output_dir.mkdir(parents=True, exist_ok=False)
    log = output_dir / 'calls.jsonl'
    with log.open('xb') as stream:
        stream.flush(); os.fsync(stream.fileno())
    state = {p['context_id']: {'status': 'missing', 'reason': 'not_started'} for p in plan}
    caches: dict[str, tuple[dict, torch.Tensor]] = {}
    work_seconds = {p['context_id']: 0.0 for p in plan}
    active_start: dict[str, float | None] = {p['context_id']: None for p in plan}
    def context_clock(cid: str) -> float:
        started = active_start[cid]
        return work_seconds[cid] + (now() - started if started is not None else 0.0)
    ledgers = {p['context_id']: study.CallLedger(context_id=p['context_id'],
               now=lambda c=p['context_id']: context_clock(c), memory_bytes=mps_bytes)
               for p in plan}
    start_mono, start_wall = now(), wall()
    counter = 0
    failure: str | None = None

    def check() -> None:
        host = host_probe()
        _need(all(host.get(k) is True for k in
                  ('ac_power', 'lid_open', 'clean_pmset_10min', 'no_transition')),
              'Host power/lid/transition gate failed')
        _need(0 <= mps_bytes() <= study.MPS_BYTES, 'MPS current allocation cap failed')
        total_work = sum(work_seconds.values()) + sum(
            now() - t for t in active_start.values() if t is not None)
        _need(total_work <= study.CAMPAIGN_SECONDS,
              'Campaign model-work cap failed between calls')
        _need(free_bytes(output_dir) >= study.FREE_DISK_BYTES,
              'Free-disk gate failed')
        total = sum(p.stat().st_size for p in output_dir.rglob('*') if p.is_file())
        _need(total <= study.OUTPUT_BYTES, 'New-output cap failed')

    def guard(context_id: str, event: str) -> None:
        nonlocal counter
        phase, sep, label = event.partition('|')
        _need(sep and phase in ('before', 'after') and label,
              'Invalid model-call label')
        check()
        if phase == 'before':
            _need(active_start[context_id] is None, 'Overlapping model call')
            active_start[context_id] = now()
            counter += 1
            _append_jsonl(log, {'ordinal': counter, 'context_id': context_id,
                                'event': 'started', 'label': label,
                                'monotonic': now(), 'utc_seconds': wall(),
                                'mps_driver_bytes': mps_bytes()})
        ledgers[context_id].guard(event)
        if phase == 'after':
            started = active_start[context_id]
            _need(started is not None, 'Call start absent')
            work_seconds[context_id] += now() - started
            active_start[context_id] = None
            _append_jsonl(log, {'ordinal': counter, 'context_id': context_id,
                                'event': 'completed', 'label': label,
                                'monotonic': now(), 'utc_seconds': wall(),
                                'mps_driver_bytes': mps_bytes()})

    def failed_call(context_id: str, exc: BaseException) -> None:
        active = ledgers[context_id].active
        if active is not None:
            started = active_start[context_id]
            if started is not None:
                work_seconds[context_id] += now() - started
                active_start[context_id] = None
            _append_jsonl(log, {'ordinal': counter, 'context_id': context_id,
                                'event': 'failed', 'label': active[0],
                                'error_type': type(exc).__name__,
                                'monotonic': now(), 'utc_seconds': wall()})

    try:
        check()
        # Capture every natural endpoint exactly once. Each opposite direction
        # later reuses this capture as its donor, with no eighth generation.
        for p in plan:
            cid = p['context_id']
            try:
                caches[cid] = clean(p, lambda event, c=cid: guard(c, event))
                answer, source = caches[cid]
                _need(answer['hook']['captured_during_generation'] is True and
                      source.shape[1] == core.PROJECTOR_WIDTH and
                      core.tensor_sha(source) == answer['hook']['source_sha256'],
                      'Clean source was not captured during generation')
                state[cid] = {'status': 'captured'}
            except BaseException as exc:
                failed_call(cid, exc)
                state[cid] = {'status': 'failed', 'reason': f'clean:{type(exc).__name__}:{exc}'}
                raise
        for p in plan:
            cid = p['context_id']
            opposite = next(q['context_id'] for q in plan if
                            q['pair_id'] == p['pair_id'] and q['engine'] == p['engine'] and
                            q['recipient'] == p['donor'] and q['donor'] == p['recipient'])
            recipient_clean, source = caches[cid]
            donor_clean, donor_source = caches[opposite]
            try:
                evidence, gradients = complete(
                    p, source, donor_source, recipient_clean, donor_clean,
                    lambda event, c=cid: guard(c, event))
                _need(evidence['context_id'] == cid and
                      evidence['source_sha256'] == core.tensor_sha(source) and
                      evidence['donor_source_sha256'] == core.tensor_sha(donor_source),
                      'Context evidence/source binding differs')
                expected = study.expected_context_call_labels(p)
                ledger = ledgers[cid].seal(expected_generation=7,
                    expected_no_grad=6 * (len(p['recipient_aliases']) + len(p['donor_aliases'])),
                    expected_grad=len(p['recipient_aliases']) + len(p['donor_aliases']),
                    expected_labels=expected)
                sidecars = {}
                for key, tensor in gradients.items():
                    sidecars[key] = _save_sidecar(output_dir / f'{p["index"]:02d}_{key}.pt', tensor)
                    word, ordinal = key.rsplit('_', 1)
                    row = evidence['grad_scores'][word]['entries'][int(ordinal)]
                    _need(row['selected_gradient_sha256'] == sidecars[key]['tensor_sha256'] and
                          row['selected_gradient_shape'] == sidecars[key]['shape'] and
                          abs(row['selected_gradient_l2'] - sidecars[key]['l2']) <= 1e-5,
                          'Gradient sidecar/receipt mismatch')
                _need(set(sidecars) == {f'{word}_{i}' for word in ('recipient', 'donor')
                                        for i in range(len(p[f'{word}_aliases']))},
                      'Gradient sidecar population incomplete')
                record = {'schema': SCHEMA + '_context', 'context': evidence,
                          'call_ledger': ledger, 'gradient_sidecars': sidecars}
                record_path = output_dir / f'context_{p["index"]:02d}.json'
                _write_json(record_path, record)
                decision = evidence['decision']
                raw = decision['raw_donor_switches']
                state[cid] = {'status': 'completed', 'record_path': record_path.name,
                              'record_sha256': sha(record_path),
                              'gate_receipt': {
                                  'local_score_pass': decision['local_score_pass'],
                                  'predicted_switch': decision['predicted_switch_count'] > 0,
                                  'donor_switch': any(raw[a] for a in
                                      ('plus_0_1', 'minus_0_1', 'plus_1_0')),
                                  'random_switch': decision['random_switch'],
                                  'pair_id': p['pair_id'], 'engine': p['engine']}}
                check()
            except BaseException as exc:
                failed_call(cid, exc)
                state[cid] = {'status': 'failed', 'reason': f'context:{type(exc).__name__}:{exc}'}
                raise
    except BaseException as exc:
        failure = f'{type(exc).__name__}:{exc}'
    elapsed_mono, elapsed_wall = now() - start_mono, wall() - start_wall
    gap = elapsed_wall - elapsed_mono
    if not math.isfinite(gap) or abs(gap) > 5:
        failure = failure or 'clock_gap_over_5_seconds'
    try:
        check()
    except BaseException as exc:
        failure = failure or f'postgate:{type(exc).__name__}:{exc}'
    rows = [{'index': p['index'], 'context_id': p['context_id'], **state[p['context_id']]}
            for p in plan]
    for row in rows:
        if row['status'] != 'completed':
            record_path = output_dir / f"context_{row['index']:02d}.json"
            _write_json(record_path, {'schema': SCHEMA + '_context', **row,
                                      'attempt_id': attempt_id,
                                      'calls_sha256_at_finalization': sha(log)})
            row['record_path'] = record_path.name
            row['record_sha256'] = sha(record_path)
    completed = sum(r['status'] == 'completed' for r in rows)
    passing = [r['gate_receipt'] for r in rows if r['status'] == 'completed' and
               r['gate_receipt']['local_score_pass'] is True]
    predicted = [r['gate_receipt'] for r in rows if r['status'] == 'completed' and
                 r['gate_receipt']['predicted_switch'] is True]
    all_receipts = [r['gate_receipt'] for r in rows if r['status'] == 'completed']
    donor_count = sum(r['donor_switch'] for r in all_receipts)
    random_count = sum(r['random_switch'] for r in all_receipts)
    categories = ({'food' if r['pair_id'] in ('pair_00', 'pair_01') else 'transport'
                   for r in passing})
    score_gate = (len(passing) >= 12 and categories == {'food', 'transport'} and
                  {r['engine'] for r in passing} == {'piper', 'samantha'})
    switch_gate = len(predicted) >= 4 and len({r['pair_id'] for r in predicted}) >= 2
    direction_gate = donor_count - random_count >= 4
    result = {'schema': SCHEMA + '_result', 'status':
              'schedule_complete_resource_unverified' if failure is None and completed == 16
              else 'failed_no_retry', 'attempt_id': attempt_id,
              'plan_sha256': _digest(plan), 'contexts': rows,
              'planned_contexts': 16, 'completed_contexts': completed,
              'failed_contexts': sum(r['status'] == 'failed' for r in rows),
              'missing_contexts': sum(r['status'] in ('missing', 'captured') for r in rows),
              'calls_sha256': sha(log), 'call_events': len(log.read_text().splitlines()),
              'model_work_seconds': sum(work_seconds.values()),
              'elapsed_monotonic_seconds': elapsed_mono,
              'clock_gap_seconds': gap, 'failure': failure,
              'frozen_gate': {'status': 'source_only_unverified',
                              'local_score_pass_contexts': len(passing),
                              'predicted_switch_contexts': len(predicted),
                              'donor_switch_contexts': donor_count,
                              'random_switch_contexts': random_count,
                              'score_gate': score_gate, 'switch_gate': switch_gate,
                              'direction_gate': direction_gate,
                              'all_clauses': completed == 16 and score_gate and
                                             switch_gate and direction_gate},
              'hard_per_call_watchdog_present': False,
              'measured_mps_high_water_present': False,
              'independent_saved_data_audit_required': True}
    _write_json(output_dir / 'campaign_result.json', result)
    return result
