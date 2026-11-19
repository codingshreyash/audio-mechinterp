"""Held one-case Wav2Vec2 causal worker; no model load without later GO.

The public run path requires an independently committed worker GO receipt.
The parent also requires a separate later parent GO and owns the wall deadline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import acoustic_fit_noise_causal_followup_v1 as causal
import acoustic_fit_noise_pilot_v2 as pilot


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "ACOUSTIC_FIT_NOISE_CAUSAL_FOLLOWUP_PROSPECTIVE_V1.md"
PREFLIGHT = HERE / "ACOUSTIC_FIT_NOISE_CAUSAL_FOLLOWUP_PREFLIGHT_V2.json"
WORKER_GO = HERE / "ACOUSTIC_FIT_NOISE_CAUSAL_WORKER_GO_V1.json"
MAX_SECONDS = 1800
MAX_RSS_BYTES = 4 * 1024**3
MAX_OUTPUT_BYTES = 128 * 1024**2
BASELINE_ARMS = ("clean", "word_noise")


def _sha(path: Path) -> str:
    return pilot.sha_file(path)


def _source_commit(path: Path) -> str:
    return pilot.source_commit(path.resolve())


def verify_worker_go(receipt: Path = PREFLIGHT,
                     review: Path = WORKER_GO) -> dict[str, Any]:
    """Fail before model load unless source, preflight and later review bind."""
    current = causal.no_weight_preflight()
    if (not receipt.is_file() or receipt.is_symlink()
            or pilot.canonical(json.loads(receipt.read_text())) != pilot.canonical(current)):
        raise ValueError("causal preflight receipt differs")
    if not review.is_file() or review.is_symlink():
        raise ValueError("causal worker HOLD: independent GO receipt absent")
    approved = json.loads(review.read_text())
    if (approved.get("schema") != "acoustic_fit_noise_causal_worker_go_v1"
            or approved.get("status") != "GO"
            or approved.get("preflight_sha256") != _sha(receipt)
            or approved.get("protocol_sha256") != _sha(PROTOCOL)
            or approved.get("source_sha256") != _sha(Path(__file__))
            or approved.get("helper_sha256") != _sha(Path(causal.__file__))):
        raise ValueError("causal worker GO source/receipt binding differs")
    review_commit = _source_commit(review)
    for source in (Path(__file__), PROTOCOL, Path(causal.__file__), receipt):
        frozen_commit = _source_commit(source)
        if (review_commit == frozen_commit or subprocess.run(
                ("git", "merge-base", "--is-ancestor", frozen_commit,
                 review_commit), cwd=pilot.ROOT, check=False).returncode != 0):
            raise ValueError("causal worker review is not later than sources")
    return approved


def _output_bytes(output: Path) -> int:
    if any(p.is_symlink() for p in output.rglob("*")):
        raise ValueError("worker output contains symlink")
    return sum(p.stat().st_size for p in output.rglob("*") if p.is_file())


def _write_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(pilot.canonical(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _save_array(output: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    if not name or "/" in name or ".." in name:
        raise ValueError("sidecar name invalid")
    path = output / f"{name}.npy"
    with path.open("xb") as stream:
        np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": path.name, "file_sha256": _sha(path),
            "raw_sha256": causal.array_sha(value), "shape": list(value.shape),
            "dtype": str(value.dtype), "bytes": path.stat().st_size}


def _decode(processor: Any, logits: np.ndarray) -> dict[str, Any]:
    if (logits.shape != (1, 286, 32) or logits.dtype != np.float32
            or not np.isfinite(logits).all()):
        raise ValueError("CTC logits geometry/finite check failed")
    ids = logits.argmax(axis=-1)[0].astype(np.int64).tolist()
    decoded = processor.batch_decode([ids])[0]
    if not isinstance(decoded, str):
        raise ValueError("CTC decode did not return transcript")
    return {"argmax_ids": ids, "transcript": decoded}


def _load_pinned_model() -> tuple[Any, Any]:
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import wav2vec2_fit_baseline_stage_capture_v1 as source

    if torch.__version__ != "2.15.0":
        raise ValueError("causal worker Torch source changed")
    processor = Wav2Vec2Processor.from_pretrained(source.SNAPSHOT,
                                                  local_files_only=True)
    model, loading = Wav2Vec2ForCTC.from_pretrained(
        source.SNAPSHOT, local_files_only=True, output_loading_info=True)
    if (set(loading["missing_keys"]) != {"wav2vec2.masked_spec_embed"}
            or loading["unexpected_keys"] or loading["mismatched_keys"]
            or loading["error_msgs"]):
        raise ValueError("causal worker checkpoint load report changed")
    model = model.to("cpu", dtype=torch.float32).eval()
    if (model.training or model.config.mask_feature_prob != 0.0
            or not hasattr(model.wav2vec2, "masked_spec_embed")):
        raise ValueError("causal worker model eval/mask contract changed")
    if any(p.device.type != "cpu" or p.dtype != torch.float32
           for p in model.parameters()):
        raise ValueError("causal worker model not wholly CPU FP32")
    return model, processor


def execute_after_admission(output: Path, *, model: Any | None = None,
                            processor: Any | None = None) -> None:
    """One-shot complete experiment; public caller must verify both reviews."""
    import torch
    import wav2vec2_fit_baseline_stage_capture_v1 as source

    if output.exists():
        raise FileExistsError("causal worker output already exists; no retry")
    output.mkdir(parents=True)
    records = output / "records.jsonl"
    records.open("x").close()
    started = time.monotonic()
    active = "setup"

    def budget() -> None:
        elapsed = time.monotonic() - started
        rss = source.current_rss_bytes()
        if not math.isfinite(elapsed) or elapsed > MAX_SECONDS:
            raise TimeoutError("causal worker elapsed cap")
        if rss > MAX_RSS_BYTES:
            raise MemoryError("causal worker RSS cap")
        if _output_bytes(output) > MAX_OUTPUT_BYTES:
            raise RuntimeError("causal worker output cap")

    _write_row(records, {
        "type": "start", "schema": "acoustic_fit_noise_causal_worker_v1",
        "receipt_sha256": _sha(PREFLIGHT), "worker_go_sha256": _sha(WORKER_GO),
        "source_sha256": _sha(Path(__file__)), "protocol_sha256": _sha(PROTOCOL),
        "selected_case": causal.CASE, "selected_arms": list(causal.ARMS),
        "start_utc": time.time(),
    })
    try:
        budget()
        torch.set_num_threads(4)
        if model is None or processor is None:
            if model is not None or processor is not None:
                raise ValueError("model and processor must be supplied together")
            model, processor = _load_pinned_model()
        clips = pilot.stimuli()[causal.CASE]
        selected = causal._selected_rows()
        baseline: dict[str, dict[str, Any]] = {}
        for arm in BASELINE_ARMS:
            active = f"baseline:{arm}"
            budget()
            pcm = clips[arm]
            values = source.normalized_input(pcm, mode="S", stats={},
                                             processor=processor)
            if (values.dtype != np.float32 or len(values) != len(pcm)
                    or not np.isfinite(values).all()):
                raise ValueError("causal normalized input invalid")
            input_tensor = torch.from_numpy(values.copy())[None]
            states, counts = pilot.wav2vec2_model_forward_v2(
                model, input_tensor, len(pcm))
            state, logits = states["feature_projection"], states["logits"]
            output_value = _decode(processor, logits)
            saved = selected[arm]
            if (causal.array_sha(state) != causal.STATE_HASHES[arm]
                    or causal.array_sha(logits) !=
                    saved["metrics"]["logits"]["sha256"]
                    or output_value != saved["output"]
                    or any(v != 1 for v in counts.values())):
                raise ValueError("causal baseline state/logits/output parity failed")
            state_file = _save_array(output, f"{arm}_state", state)
            logits_file = _save_array(output, f"{arm}_logits", logits)
            baseline[arm] = {"state": state, "logits": logits,
                             "input": input_tensor, "output": output_value}
            _write_row(records, {
                "type": "baseline", "case": causal.CASE, "arm": arm,
                "pcm_sha256": causal.array_sha(pcm), "state": state_file,
                "logits": logits_file, "output": output_value,
                "hook_counts": counts,
                "elapsed_seconds": time.monotonic() - started,
                "rss_bytes": source.current_rss_bytes(),
            })
            budget()
        for recipient_arm in BASELINE_ARMS:
            donor_arm = next(arm for arm in BASELINE_ARMS
                             if arm != recipient_arm)
            recipient, donor = baseline[recipient_arm], baseline[donor_arm]
            for arm in causal.ARMS:
                active = f"patch:{recipient_arm}:{arm}"
                budget()
                patched, edit = causal.construct_patch(
                    recipient["state"], donor["state"],
                    recipient_arm=recipient_arm, arm=arm)
                logits, hook_count = causal.patched_forward(
                    model, recipient["input"],
                    recipient_state=recipient["state"], patched_state=patched)
                output_value = _decode(processor, logits)
                logits_file = _save_array(
                    output, f"{recipient_arm}_{arm}_logits", logits)
                self_equal = arm == "self_W" and np.array_equal(
                    logits, recipient["logits"])
                full_equal = arm == "full_donor" and np.array_equal(
                    logits, donor["logits"])
                _write_row(records, {
                    "type": "patch", "case": causal.CASE,
                    "recipient_arm": recipient_arm, "donor_arm": donor_arm,
                    "arm": arm, "edit": edit, "logits": logits_file,
                    "output": output_value, "hook_count": hook_count,
                    "self_logits_exact": self_equal if arm == "self_W" else None,
                    "full_donor_logits_exact": full_equal if arm == "full_donor" else None,
                    "elapsed_seconds": time.monotonic() - started,
                    "rss_bytes": source.current_rss_bytes(),
                })
                if arm == "self_W" and not self_equal:
                    raise ValueError("causal self-logits parity failed")
                if arm == "full_donor" and not full_equal:
                    raise ValueError("causal full-donor logits parity failed")
                budget()
        _write_row(records, {
            "type": "complete", "baseline_rows": 2, "patch_rows": 16,
            "control_gate_pass": True,
            "elapsed_seconds": time.monotonic() - started,
        })
        budget()
    except BaseException as exc:
        _write_row(records, {
            "type": "failure", "active": active,
            "error_type": type(exc).__name__, "detail": str(exc),
            "elapsed_seconds": time.monotonic() - started,
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate-only", "run"))
    parser.add_argument("--receipt", type=Path, default=PREFLIGHT)
    parser.add_argument("--worker-review", type=Path, default=WORKER_GO)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "validate-only":
        current = causal.no_weight_preflight()
        print(pilot.canonical({
            "schema": "acoustic_fit_noise_causal_worker_source_v1",
            "source_sha256": _sha(Path(__file__)),
            "protocol_sha256": _sha(PROTOCOL),
            "preflight_matches": pilot.canonical(
                json.loads(PREFLIGHT.read_text())) == pilot.canonical(current),
            "worker_go_present": WORKER_GO.is_file(),
        }))
        return
    if args.output is None:
        parser.error("run requires --output")
    if args.receipt.resolve() != PREFLIGHT.resolve() or args.worker_review.resolve() != WORKER_GO.resolve():
        raise ValueError("causal worker run requires pinned receipt/review paths")
    verify_worker_go(args.receipt, args.worker_review)
    import acoustic_fit_noise_causal_parent_v1 as parent
    parent.verify_parent_go()
    if args.output.resolve() != (parent.ATTEMPT / "worker_output").resolve():
        raise ValueError("causal worker output must be the pinned parent attempt")
    launch = json.loads((parent.ATTEMPT / "launch_start.json").read_text())
    if (launch.get("source_binding") != parent.source_binding()
            or launch.get("worker_go_sha256") != _sha(WORKER_GO)
            or launch.get("parent_go_sha256") != _sha(parent.PARENT_GO)):
        raise ValueError("causal worker parent launch identity differs")
    execute_after_admission(args.output)


if __name__ == "__main__":
    main()
