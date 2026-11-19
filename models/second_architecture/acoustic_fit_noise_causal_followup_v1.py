"""Source-only FIT causal follow-up for one observed Wav2Vec2 noise correction.

Public model execution stays HOLD until a separately reviewed worker/parent
launcher, no-weight preflight and ordered GO receipts exist. This module
provides frozen evidence, deterministic patch construction and a private
injected-model hook for fake tests. See the prospective V1 protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import acoustic_fit_noise_pilot_v2 as pilot
import acoustic_fit_noise_parent_launch_v2 as parent
import acoustic_fit_noise_parent_launch_v1 as parent_v1


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "ACOUSTIC_FIT_NOISE_CAUSAL_FOLLOWUP_PROSPECTIVE_V1.md"
PRIOR = HERE / "runs/acoustic_fit_noise_parent_v2/wav2vec2_attempt_002"
PRIOR_RESULT_SHA = "5862c0b0e326840727473a5ffee9356ffb5181ca8a4d1c9053f9a27992d9e453"
PRIOR_RECORDS_SHA = "c1badaf305dfaebe9beb9567b73f4e4bfdfab84b45ca8194f4c13c943f60038c"
PRIOR_RECEIPT = HERE / "ACOUSTIC_FIT_NOISE_PREFLIGHT_V2_RESTRICTED.json"
PRIOR_WORKER_GO = HERE / "ACOUSTIC_FIT_NOISE_WORKER_SOURCE_GO_V2.json"
PRIOR_PARENT_GO = HERE / "ACOUSTIC_FIT_NOISE_PARENT_SOURCE_GO_V2.json"
FAILED_V1 = HERE / "runs/acoustic_fit_noise_parent_v1/wav2vec2_attempt_001"
FAILED_V1_RESULT_SHA = "b69468b110a06a55ae8b4b96be367dbc27d608bd1ecfc2b5ecd334547a7d6ebd"
FAILED_V1_RECORDS_SHA = "f1752019284dec76d8cd14203b1d8bc63f5b2890a6f140dbb42d3901c732d5cc"
FAILED_V1_STDERR_SHA = "a23cc760bf3fd9e1b43ca7d9da914d2111bfcf8a6df2ae5256a59973100dd9b5"
FAILED_V1_RECEIPT = HERE / "ACOUSTIC_FIT_NOISE_PREFLIGHT_V1.json"
FAILED_V1_WORKER_GO = HERE / "ACOUSTIC_FIT_NOISE_WORKER_SOURCE_GO_V1.json"
FAILED_V1_PARENT_GO = HERE / "ACOUSTIC_FIT_NOISE_PARENT_SOURCE_GO_V1.json"
CASE = "kokoro_teal"
FRAMES = 91840
STATE_SHAPE = (1, 286, 768)
WORD = (105, 179)
ARMS = ("self_W", "W_donor", "C_donor", "full_donor",
        "R1_W", "R2_W", "R1_C", "R2_C")
PCM_HASHES = {
    "clean": "84e07ded47cb6410b0840e7d502f10810018319b5d55b50fe22aaed890848006",
    "word_noise": "f9a48dcfb965605774aec3ec375a9dacc33fe7b4ed95153430f116122ebedc63",
}
STATE_HASHES = {
    "clean": "b2ee40b5b3b660156152a47cf7e6c912c0dc64e0f726472b1ef4141b13c3aef7",
    "word_noise": "6c3157be1232e2c345a1c092c5ac0d1c45ac185534405dfc76a1f1fbf8877d65",
}
TRANSCRIPTS = {
    "clean": "THE WORD I CHOSE IS TIL WHAT WORD DID I CHOOSE",
    "word_noise": "THE WORD I CHOSE IS TEAL WHAT WORD DID I CHOOSE",
}
SEEDS = {
    "clean": {"R1_W": 2026111901, "R2_W": 2026111902,
              "R1_C": 2026111903, "R2_C": 2026111904},
    "word_noise": {"R1_W": 2026111911, "R2_W": 2026111912,
                   "R1_C": 2026111913, "R2_C": 2026111914},
}


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def array_sha(value: np.ndarray) -> str:
    return sha_bytes(np.ascontiguousarray(value).tobytes())


def _failed_v1_evidence() -> dict[str, str]:
    """Bind and re-audit the immutable first-forward capture failure."""
    expected = {
        "launch_result.json": FAILED_V1_RESULT_SHA,
        "worker_output/records.jsonl": FAILED_V1_RECORDS_SHA,
        "worker_stderr.txt": FAILED_V1_STDERR_SHA,
    }
    if any(pilot.sha_file(FAILED_V1 / name) != digest
           for name, digest in expected.items()):
        raise ValueError("failed V1 attempt bytes changed")
    audit = parent_v1.audit_saved_attempt(
        "wav2vec2", FAILED_V1_RECEIPT.resolve(), FAILED_V1_WORKER_GO.resolve(),
        FAILED_V1_PARENT_GO.resolve(), FAILED_V1.resolve())
    if (audit.get("status") != "bound_invalid_or_partial"
            or audit.get("result_sha256") != FAILED_V1_RESULT_SHA):
        raise ValueError("failed V1 saved audit no longer returns bound failure")
    result = json.loads((FAILED_V1 / "launch_result.json").read_text())
    rows = [json.loads(line) for line in
            (FAILED_V1 / "worker_output/records.jsonl").read_text().splitlines()]
    if (result.get("model") != "wav2vec2"
            or result.get("status") != "invalid_or_partial"
            or result.get("worker_complete") is not False
            or result.get("worker_returncode") != 1
            or result.get("errors") != ["worker_exit_1", "worker_matrix_incomplete"]
            or len(rows) != 2 or rows[0].get("type") != "start"
            or rows[1].get("type") != "failure"
            or rows[1].get("error") != "ValueError"
            or rows[1].get("detail") !=
            "capture shape mismatch at post_block0: torch.Size([286, 768])"):
        raise ValueError("failed V1 attempt status or first-forward failure differs")
    return {
        "parent_result_sha256": FAILED_V1_RESULT_SHA,
        "worker_records_sha256": FAILED_V1_RECORDS_SHA,
        "worker_stderr_sha256": FAILED_V1_STDERR_SHA,
        "preflight_sha256": pilot.sha_file(FAILED_V1_RECEIPT),
        "worker_go_sha256": pilot.sha_file(FAILED_V1_WORKER_GO),
        "parent_go_sha256": pilot.sha_file(FAILED_V1_PARENT_GO),
    }


def _selected_rows() -> dict[str, dict[str, Any]]:
    """Read-only identity and saved-audit gate; no candidate model factory."""
    if (pilot.sha_file(PRIOR / "launch_result.json") != PRIOR_RESULT_SHA
            or pilot.sha_file(PRIOR / "worker_output/records.jsonl") != PRIOR_RECORDS_SHA):
        raise ValueError("selected V2 acoustic attempt bytes changed")
    audit = parent.audit_saved_attempt(
        "wav2vec2", PRIOR_RECEIPT.resolve(), PRIOR_WORKER_GO.resolve(),
        PRIOR_PARENT_GO.resolve(), PRIOR.resolve())
    if audit.get("status") != "complete_evidence_pass":
        raise ValueError("selected V2 acoustic attempt is not a saved-audit pass")
    rows = [json.loads(line) for line in
            (PRIOR / "worker_output/records.jsonl").read_text().splitlines()]
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        arm = row.get("arm")
        if row.get("type") == "case" and row.get("case") == CASE and arm in PCM_HASHES:
            if arm in selected:
                raise ValueError("selected acoustic row duplicated")
            selected[arm] = row
    if set(selected) != set(PCM_HASHES):
        raise ValueError("selected acoustic pair incomplete")
    for arm, row in selected.items():
        metric = row["metrics"]["feature_projection"]
        ids = row["output"]["argmax_ids"]
        if (row.get("pcm_sha256") != PCM_HASHES[arm]
                or row["output"].get("transcript") != TRANSCRIPTS[arm]
                or metric.get("sha256") != STATE_HASHES[arm]
                or metric.get("shape") != list(STATE_SHAPE)
                or metric.get("nominal_window") != list(WORD)
                or not isinstance(ids, list) or len(ids) != STATE_SHAPE[1]):
            raise ValueError("selected acoustic output/state binding differs")
    return selected


def no_weight_preflight() -> dict[str, Any]:
    """Recompute all frozen source/attempt checks before any model load."""
    failed_v1 = _failed_v1_evidence()
    selected = _selected_rows()
    current = pilot.preflight()
    if pilot.canonical(json.loads(PRIOR_RECEIPT.read_text())) != pilot.canonical(current):
        raise ValueError("V2 no-weight preflight changed")
    clips = pilot.stimuli()[CASE]
    clean, noisy = (clips[arm] for arm in PCM_HASHES)
    if (len(clean) != len(noisy) or len(clean) != FRAMES
            or any(array_sha(clips[arm]) != PCM_HASHES[arm] for arm in PCM_HASHES)
            or not np.array_equal(clean[:33920], noisy[:33920])
            or not np.array_equal(clean[57120:], noisy[57120:])
            or np.array_equal(clean[33920:57120], noisy[33920:57120])):
        raise ValueError("selected PCM or word-only change differs")
    return {
        "schema": "acoustic_fit_noise_causal_followup_preflight_v2",
        "source_sha256": pilot.sha_file(Path(__file__)),
        "source_commit": pilot.source_commit(Path(__file__)),
        "protocol_sha256": pilot.sha_file(PROTOCOL),
        "protocol_commit": pilot.source_commit(PROTOCOL),
        "prior_result_sha256": PRIOR_RESULT_SHA,
        "prior_records_sha256": PRIOR_RECORDS_SHA,
        "prior_preflight_sha256": pilot.sha_file(PRIOR_RECEIPT),
        "prior_worker_go_sha256": pilot.sha_file(PRIOR_WORKER_GO),
        "prior_parent_go_sha256": pilot.sha_file(PRIOR_PARENT_GO),
        "failed_v1_evidence": failed_v1,
        "selected_pcm_sha256": PCM_HASHES,
        "selected_state_sha256": STATE_HASHES,
        "selected_transcripts": TRANSCRIPTS,
        "selected_frame_path_sha256": {
            arm: sha_bytes(pilot.canonical(selected[arm]["output"]["argmax_ids"]).encode())
            for arm in PCM_HASHES},
        "state_shape": list(STATE_SHAPE), "word_window": list(WORD),
        "arms": list(ARMS), "random_seeds": SEEDS,
        "model_revisions": current["model_revisions"],
        "capture_contract": current["wav2vec2_capture_contract"],
        "limits": {"cpu_threads": 4, "parent_seconds": 1800,
                   "rss_bytes": 4 * 1024**3, "output_bytes": 128 * 1024**2},
    }


def _region_indices(arm: str) -> np.ndarray:
    if arm in ("self_W", "W_donor", "R1_W", "R2_W"):
        return np.arange(*WORD)
    if arm in ("C_donor", "R1_C", "R2_C"):
        return np.r_[np.arange(0, WORD[0]), np.arange(WORD[1], STATE_SHAPE[1])]
    if arm == "full_donor":
        return np.arange(STATE_SHAPE[1])
    raise ValueError("unknown frozen patch arm")


def construct_patch(recipient: np.ndarray, donor: np.ndarray, *,
                    recipient_arm: str, arm: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Create one exact row edit, including equal-norm random controls."""
    if (recipient_arm not in PCM_HASHES or arm not in ARMS
            or not isinstance(recipient, np.ndarray) or not isinstance(donor, np.ndarray)
            or recipient.shape != STATE_SHAPE or donor.shape != STATE_SHAPE
            or recipient.dtype != donor.dtype or recipient.dtype != np.float32
            or not np.isfinite(recipient).all() or not np.isfinite(donor).all()):
        raise ValueError("patch identity, shape, dtype or finite check failed")
    indices = _region_indices(arm)
    target = donor[:, indices, :].astype(np.float64) - recipient[:, indices, :].astype(np.float64)
    target_norm = float(np.linalg.norm(target))
    patched = recipient.copy()
    seed = SEEDS[recipient_arm].get(arm)
    if arm == "self_W":
        pass
    elif arm in ("W_donor", "C_donor", "full_donor"):
        patched[:, indices, :] = donor[:, indices, :]
    else:
        if not math.isfinite(target_norm) or target_norm <= 0 or seed is None:
            raise ValueError("random arm has zero/nonfinite target radius or absent seed")
        draw = np.random.Generator(np.random.PCG64(seed)).standard_normal(target.shape)
        radius = float(np.linalg.norm(draw))
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("random arm draw invalid")
        edit = (draw * (target_norm / radius)).astype(np.float32)
        patched[:, indices, :] = recipient[:, indices, :] + edit
    realized = float(np.linalg.norm(
        patched.astype(np.float64) - recipient.astype(np.float64)))
    realized_edit = np.ascontiguousarray(patched - recipient)
    if (not np.isfinite(patched).all()
            or (arm == "self_W" and not np.array_equal(patched, recipient))
            or (arm == "full_donor" and not np.array_equal(patched, donor))):
        raise ValueError("patch parity or finite check failed")
    unchanged = np.setdiff1d(np.arange(STATE_SHAPE[1]), indices)
    if unchanged.size and not np.array_equal(patched[:, unchanged, :],
                                             recipient[:, unchanged, :]):
        raise ValueError("patch changed rows outside frozen region")
    if arm.startswith("R") and abs(realized - target_norm) > 1e-4 * target_norm:
        raise ValueError("random edit norm differs from natural row delta")
    return patched, {
        "recipient_arm": recipient_arm, "arm": arm, "seed": seed,
        "row_indices_sha256": array_sha(indices.astype("<i8")),
        "target_l2": target_norm, "realized_l2": realized,
        "recipient_state_sha256": array_sha(recipient),
        "donor_state_sha256": array_sha(donor),
        "patched_state_sha256": array_sha(patched),
        "realized_edit_sha256": array_sha(realized_edit),
        "realized_edit_dtype": str(realized_edit.dtype),
        "realized_edit_shape": list(realized_edit.shape),
    }


def patched_forward(model: Any, input_values: Any, *, recipient_state: np.ndarray,
                    patched_state: np.ndarray) -> tuple[np.ndarray, int]:
    """Private fake-testable hook; caller must bind model/input and parent gates."""
    import torch

    if (recipient_state.shape != STATE_SHAPE or patched_state.shape != STATE_SHAPE
            or recipient_state.dtype != patched_state.dtype
            or recipient_state.dtype != np.float32):
        raise ValueError("feature-projection state geometry differs")
    count = 0

    def hook(_module: Any, _args: tuple[Any, ...], result: Any) -> Any:
        nonlocal count
        count += 1
        if (count != 1 or not isinstance(result, tuple) or len(result) != 2
                or not all(torch.is_tensor(x) for x in result)
                or result[0].device.type != "cpu"
                or result[0].dtype != torch.float32
                or tuple(result[0].shape) != STATE_SHAPE
                or not np.array_equal(result[0].detach().contiguous().numpy(),
                                      recipient_state)):
            raise ValueError("feature-projection hook parity differs")
        return (torch.from_numpy(patched_state.copy()), result[1])

    handle = model.wav2vec2.feature_projection.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = model(input_values=input_values, attention_mask=None,
                           output_hidden_states=False, return_dict=True)
    finally:
        handle.remove()
    if count != 1:
        raise ValueError("feature-projection hook count differs")
    logits = output.logits.detach().cpu().contiguous().numpy().copy()
    if (logits.dtype != np.float32 or logits.shape != (1, 286, 32)
            or not np.isfinite(logits).all()):
        raise ValueError("patched logits geometry/finite check failed")
    return logits, count


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run"))
    args = parser.parse_args()
    if args.command == "preflight":
        print(pilot.canonical(no_weight_preflight()))
    else:
        raise ValueError("causal follow-up HOLD: reviewed worker/parent launcher absent")


if __name__ == "__main__":
    main()
