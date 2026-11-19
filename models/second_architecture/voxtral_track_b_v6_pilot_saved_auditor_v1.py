"""Read-only, source-only V6 pilot saved-data auditor. Never issues GO.

The currently reviewed worker emits post-call memory samples, not a measured
within-call high-water mark, and has no hard per-call watchdog. A complete
saved ledger can therefore be internally valid while remaining admission HOLD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
SCHEMA = 'voxtral_track_b_v6_pilot_measurement_v1'
WORKER_SHA = '261ff1dc61b96268fd1ba5594c1a95e31dedc7eba5d7daf39bc567c85d0ae30f'
STUDY_SHA = '6bddaff0dad628f4181aefd701eaf533ab2b2ac1d7cc91af64314022ecab5c26'
PROTOCOL_SHAS = {
    'v3': 'f3716897cfff30b26a4bb66d882222fc2137b0911266d76ea0a0c4319854ce0b',
    'v4': '87ef41b3c854ed23fc4cfb3f67a23f2b6cd595958a1f941bc9726288ea0d923c',
    'v5': '70d24e5d4ba341e1aaa918608d57f1ed237fd56742b0d956816f4ce636ac3115',
    'v6': '6a341f316e4056c6bc81c6e191deea3f90dcddcd81050a39c734a257d4f02b9c',
    'v6_go': '28341f87e9eb7ba8a7827460f2872d01ea17adcab3f3a36b649d4542eacae0b1',
}
PROTOCOL_FILES = {
    'v3': 'VOXTRAL_FP32_TRACK_B_ANSWER_LINK_PROSPECTIVE_V3.md',
    'v4': 'VOXTRAL_FP32_TRACK_B_ANSWER_LINK_PROSPECTIVE_V4_ERRATUM.md',
    'v5': 'VOXTRAL_FP32_TRACK_B_ANSWER_LINK_PROSPECTIVE_V5_SEQUENTIAL_ALIAS_AMENDMENT.md',
    'v6': 'VOXTRAL_FP32_TRACK_B_ANSWER_LINK_PROSPECTIVE_V6_RESOURCE_PILOT_AMENDMENT.md',
    'v6_go': 'VOXTRAL_FP32_TRACK_B_ANSWER_LINK_PROSPECTIVE_V6_INDEPENDENT_REVIEW_GO.md',
}
COUNTS = {'old_fit_clean': 1, 'old_fit_zero': 1, 'forced_clean': 1,
          'forced_zero': 1, 'score_nograd': 72, 'score_grad': 12}
ORDER = (['old_fit_clean', 'old_fit_zero', 'forced_clean', 'forced_zero'] +
         ['score_nograd'] * 72 + ['score_grad'] * 12)
MAX_DRIVER_BYTES = 30 * 1024**3
MAX_OUTPUT_BYTES = 2 * 1024**3
MIN_FREE_BYTES = 12 * 1024**3


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), f'Missing or symlinked {path.name}')
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f'{path.name} must be JSON object')
    return value


def finite_positive(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(value) and value > 0, f'{label} must be finite positive')
    return float(value)


def near(actual: Any, expected: float, label: str) -> None:
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool) and
            math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8),
            f'{label} differs from saved ledger recomputation')


def audit_ledger(path: Path, receipt: Mapping[str, Any]) -> dict:
    require(path.is_file() and not path.is_symlink(), 'Missing or symlinked calls.jsonl')
    require(sha(path) == receipt.get('calls_sha256'), 'Call ledger SHA mismatch')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    require(len(rows) == 2 * len(ORDER), 'Pilot ledger is not 88 started/completed pairs')
    by_class: dict[str, list[float]] = {key: [] for key in COUNTS}
    peaks = []
    for idx, label in enumerate(ORDER, start=1):
        started, done = rows[2 * idx - 2:2 * idx]
        require(started == {'ordinal': idx, 'class': label, 'phase': 'started'},
                f'Call {idx} start/order differs')
        require(done.get('ordinal') == idx and done.get('class') == label and
                done.get('phase') == 'completed' and done.get('hook_calls') == 1,
                f'Call {idx} completion, class, or hook count differs')
        elapsed = finite_positive(done.get('elapsed_seconds'), f'call {idx} elapsed')
        peak = done.get('observed_postcall_mps_driver_bytes')
        require(isinstance(peak, int) and not isinstance(peak, bool) and
                0 <= peak <= MAX_DRIVER_BYTES, f'Call {idx} post-call driver sample invalid')
        by_class[label].append(elapsed)
        peaks.append(peak)
        pre = done.get('projector_pre_sha256')
        require(isinstance(pre, str) and len(pre) == 64, f'Call {idx} source hash missing')
        if label.startswith('score_'):
            require(done.get('score_finite') is True and
                    isinstance(done.get('score_sha256'), str) and
                    len(done['score_sha256']) == 64,
                    f'Call {idx} score evidence missing')
            if label == 'score_grad':
                require(done.get('gradient_finite') is True and
                        isinstance(done.get('gradient_sha256'), str) and
                        len(done['gradient_sha256']) == 64,
                        f'Call {idx} gradient evidence missing')
        else:
            require(isinstance(done.get('suffix_sha256'), str) and
                    len(done['suffix_sha256']) == 64 and
                    done.get('projector_post_sha256') == pre,
                    f'Call {idx} suffix or projector parity evidence missing')
            length = done.get('suffix_length')
            require(isinstance(length, int) and 1 <= length <= 32,
                    f'Call {idx} suffix length invalid')
            if label.startswith('forced_'):
                require(length == 32, f'Call {idx} forced generation not 32 tokens')
    done = {k: len(v) for k, v in by_class.items()}
    for key in ('planned_calls', 'attempted_calls', 'completed_calls'):
        require(receipt.get(key) == COUNTS, f'{key} differs from fixed schedule')
    require(receipt.get('missing_calls') == {key: 0 for key in COUNTS} and
            receipt.get('attempted_call_count') == 88 and done == COUNTS,
            'Pilot receipt counts/missing slots differ')
    require(receipt.get('max_observed_postcall_mps_driver_bytes') == max(peaks),
            'Saved post-call driver maximum differs')
    for a, b in ((1, 3), (5, 7)):
        require(rows[a]['suffix_sha256'] == rows[b]['suffix_sha256'] and
                rows[a]['projector_pre_sha256'] == rows[b]['projector_pre_sha256'],
                'Old-FIT or forced-32 clean/zero parity failed')
    # The worker permits <=1e-4 numeric differences, but saves only score
    # hashes. Distinct hashes cannot establish whether that tolerance passed.
    return {'counts': done, 't_g': max(by_class['forced_clean'] + by_class['forced_zero']),
            't_s': max(by_class['score_nograd']), 't_b': max(by_class['score_grad']),
            'postcall_driver_max_bytes': max(peaks), 'rows': len(rows),
            'sum_call_elapsed_seconds': sum(sum(v) for v in by_class.values())}


def audit_saved(directory: Path, *, admission: Path, qa: Path, qa_review: Path,
                preflight: Path, study_runner: Path, worker: Path,
                free_disk_bytes: int, expected_measurement_sha256: str,
                expected_admission_sha256: str) -> dict:
    """Verify immutable evidence and return HOLD decision; never write a GO."""
    require(directory.is_dir() and not directory.is_symlink(), 'Pilot output directory absent')
    require(not (directory / 'pilot_failure.json').exists(), 'Pilot has failure artifact')
    measurement_path = directory / 'pilot_measurement.json'
    require(sha(measurement_path) == expected_measurement_sha256 and
            sha(admission) == expected_admission_sha256,
            'Externally pinned measurement/admission SHA differs')
    receipt = read_json(measurement_path)
    require(receipt.get('schema') == SCHEMA and receipt.get('status') in
            ('schedule_complete_resource_unverified', 'complete_infeasible'),
            'Pilot measurement schema/status invalid')
    for key, name in PROTOCOL_FILES.items():
        require(sha(HERE / name) == PROTOCOL_SHAS[key], f'Frozen {key} protocol changed')
    require(sha(worker) == WORKER_SHA and sha(study_runner) == STUDY_SHA,
            'Pinned worker/study runner source changed')
    gate = read_json(admission)
    binding = receipt.get('bindings')
    require(gate.get('schema') == 'voxtral_track_b_v6_external_pilot_admission_v1' and
            gate.get('status') == 'pilot_go' and isinstance(binding, dict) and
            gate.get('bindings') == binding and binding.get('protocol_sha256') == PROTOCOL_SHAS,
            'Exact independent pilot admission bindings differ')
    claim_text = binding.get('campaign_claim_path')
    require(isinstance(claim_text, str) and Path(claim_text).is_absolute(),
            'Campaign claim path missing')
    claim = read_json(Path(claim_text))
    require(claim.get('schema') == 'voxtral_track_b_v6_campaign_claim_v1' and
            claim.get('campaign_attempt_id') == binding.get('campaign_attempt_id') and
            claim.get('output_dir') == str(directory.resolve()) and
            claim.get('bindings_sha256') == hashlib.sha256(json.dumps(
                binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            'Immutable campaign claim differs')
    for key, path in (('qa_sha256', qa), ('qa_review_sha256', qa_review),
                      ('preflight_sha256', preflight), ('study_runner_sha256', study_runner),
                      ('pilot_worker_sha256', worker),
                      ('pilot_auditor_sha256', Path(__file__))):
        require(sha(path) == binding.get(key), f'{key} file binding differs')
    qa_data = read_json(qa)
    require(qa_data.get('schema') == 'word_disjoint_human_qa_audit_v1' and
            qa_data.get('status') == 'human_qa_32_of_32_pass' and
            qa_data.get('human_quality_gate_passed') is True and
            qa_data.get('planned_clips') == qa_data.get('passed_clips') == 32 and
            'Decision: GO' in qa_review.read_text() and binding['qa_sha256'] in qa_review.read_text(),
            '32/32 human QA or independent review differs')
    pre_data = read_json(preflight)
    geometry = pre_data.get('v6_synthetic_zero_geometry')
    require(isinstance(geometry, dict) and
            hashlib.sha256(json.dumps(geometry, sort_keys=True,
                         separators=(',', ':')).encode()).hexdigest() ==
            binding.get('synthetic_geometry_sha256') and
            geometry.get('longest_alias', {}).get('token_ids_with_eos') ==
            receipt.get('longest_alias_ids_with_eos') and
            receipt.get('synthetic_input_sha256') == binding.get('synthetic_batch_sha256') and
            receipt.get('old_fit_input_sha256') == binding.get('old_fit_batch_sha256'),
            'Pinned synthetic geometry, alias, or input hash differs')
    require(receipt.get('synthetic_suffix_length') == 32,
            'Forced synthetic suffix length receipt differs')
    files = list(directory.rglob('*'))
    require(all(not p.is_symlink() for p in files), 'Symlink in immutable pilot output')
    output_bytes = sum(p.stat().st_size for p in files if p.is_file())
    require(output_bytes <= MAX_OUTPUT_BYTES and
            isinstance(free_disk_bytes, int) and free_disk_bytes >= MIN_FREE_BYTES,
            'Output size or current free-disk cap failed')
    require({p.name for p in files if p.is_file()} ==
            {'pilot_measurement.json', 'calls.jsonl', 'host_pre.raw', 'host_post.raw'} and
            all(p.parent == directory for p in files if p.is_file()) and
            len([p for p in files if p.is_file()]) == 4,
            'Unexpected or missing pilot output file')
    host = {}
    for label in ('pre', 'post'):
        raw = directory / f'host_{label}.raw'
        claim = receipt.get(f'host_{label}')
        require(isinstance(claim, dict) and sha(raw) == receipt.get(f'host_{label}_sha256') ==
                claim.get('raw_power_log_sha256') and
                raw.stat().st_size == claim.get('raw_power_log_bytes') and
                0 < raw.stat().st_size <= 128 * 1024**2 and
                claim.get('ac') is True and claim.get('lid_open') is True and
                claim.get('ten_minute_clean') is True,
                f'Host {label} raw/claim mismatch')
        host[label] = {'sha256': sha(raw), 'bytes': raw.stat().st_size}
    require(receipt['host_post'].get('during_run_clean') is True,
            'During-run host transition claim failed')
    gap = receipt.get('clock_gap_seconds')
    require(isinstance(gap, (int, float)) and not isinstance(gap, bool) and
            math.isfinite(gap) and 0 <= gap <= 5, 'UTC/monotonic gap failed')
    ledger = audit_ledger(directory / 'calls.jsonl', receipt)
    p = finite_positive(receipt.get('model_work_seconds'), 'pilot model work P')
    require(p <= 240 and p + 1e-5 >= ledger['sum_call_elapsed_seconds'],
            'Pilot 240-second cap or elapsed accounting failed')
    tg, ts, tb = ledger['t_g'], ledger['t_s'], ledger['t_b']
    w = 7 * tg + 72 * ts + 12 * tb
    timing = receipt.get('timing')
    require(isinstance(timing, dict), 'Timing projection missing')
    for key, expected in {'t_g': tg, 't_s': ts, 't_b': tb, 'P': p,
                          'per_context_1p2_W': 1.2 * w,
                          'campaign_1p2_P_plus_16W': 1.2 * (p + 16 * w)}.items():
        near(timing.get(key), expected, key)
    feasible = 1.2 * w <= 240 and 1.2 * (p + 16 * w) <= 3840
    require((receipt['status'] == 'schedule_complete_resource_unverified') == feasible,
            'Pilot feasibility label differs from recomputed timing')
    require(receipt.get('measured_mps_high_water_present') is False and
            receipt.get('hard_per_call_watchdog_present') is False,
            'Pinned worker cannot attest true high-water or hard watchdog')
    reasons = []
    if not feasible:
        reasons.append('measured_timing_projection_exceeds_cap')
    reasons.append('true_within_call_mps_high_water_absent')
    reasons.append('hard_per_call_watchdog_absent')
    # The worker saves raw pmset evidence and self-reported booleans. This
    # module intentionally does not turn those into an independent host GO.
    reasons.append('independent_host_log_interpretation_required')
    reasons.append('numeric_score_repeat_parity_not_recomputable_from_hashes')
    reasons.append('utc_monotonic_gap_raw_clock_pairs_absent')
    reasons.append('independent_human_post_pilot_review_required')
    return {'schema': 'voxtral_track_b_v6_saved_audit_v1',
            'status': 'HOLD_no_post_pilot_go', 'hold_reasons': reasons,
            'pilot_measurement_sha256': sha(measurement_path),
            'calls_sha256': sha(directory / 'calls.jsonl'),
            'admission_sha256': sha(admission), 'ledger': ledger,
            'host_raw': host, 'clock_gap_seconds': gap,
            'output_bytes': output_bytes, 'free_disk_bytes': free_disk_bytes,
            'timing_recomputed': {'t_g': tg, 't_s': ts, 't_b': tb, 'P': p,
                                  'per_context_1p2_W': 1.2 * w,
                                  'campaign_1p2_P_plus_16W': 1.2 * (p + 16 * w)},
            'timing_feasible': feasible, 'go_issued': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir', type=Path, required=True)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--qa', type=Path, required=True)
    parser.add_argument('--qa-review', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--expected-measurement-sha256', required=True)
    parser.add_argument('--expected-admission-sha256', required=True)
    args = parser.parse_args()
    import shutil
    result = audit_saved(args.pilot_dir, admission=args.admission, qa=args.qa,
                         qa_review=args.qa_review, preflight=args.preflight,
                         study_runner=HERE / 'voxtral_fp32_track_b_study_runner_v1.py',
                         worker=HERE / 'voxtral_track_b_v6_pilot_worker_source_v1.py',
                         free_disk_bytes=shutil.disk_usage(args.pilot_dir).free,
                         expected_measurement_sha256=args.expected_measurement_sha256,
                         expected_admission_sha256=args.expected_admission_sha256)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
